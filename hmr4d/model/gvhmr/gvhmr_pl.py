from typing import Any, Dict
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from einops import rearrange, einsum
from hmr4d.configs import MainStore, builds
import cv2
import os

from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points, look_at_rotation
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo.augment_noisy_pose import (
    get_wham_aug_kp3d,
    get_visible_mask,
    get_invisible_legs_mask,
    randomly_occlude_lower_half,
    randomly_modify_hands_legs,
)
from hmr4d.utils.geo.hmr_cam import perspective_projection, normalize_kp2d, safely_render_x3d_K, get_bbx_xys, create_camera_sensor

from hmr4d.utils.vis.cv2_utils import draw_bbx_xys_on_image_batch, draw_coco17_skeleton_batch
from hmr4d.utils.geo.flip_utils import flip_smplx_params, avg_smplx_aa
from hmr4d.model.gvhmr.utils.postprocess import pp_static_joint, pp_static_joint_cam, process_ik

SMPL_BODY_KEYS = ("body_pose", "betas", "global_orient", "transl")


def _body_smpl_params(params):
    return {k: v for k, v in params.items() if k in SMPL_BODY_KEYS}




class GvhmrPL(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        optimizer=None,
        scheduler_cfg=None,
        ignored_weights_prefix=["smplx", "pipeline.endecoder"],
        freeze_backbone=False,
        freeze_exo_head=False,
        freeze_ego_head=True,
        copy_exo_to_ego=True,
        copy_exo_to_ego_after_load=False,
        copy_exo_to_ego_mode="pose",
        backbone_lr_scale=1.0,
        unfreeze_last_n_blocks=0,
        vis_every_n_steps=100,
        val_vis_every_n_batches=50,  # 验证集可视化间隔（每隔 N 个 batch 可视化一次）
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.copy_exo_to_ego_after_load = copy_exo_to_ego_after_load
        self.copy_exo_to_ego_mode = copy_exo_to_ego_mode
        if copy_exo_to_ego:
            self._copy_exo_to_ego(mode=copy_exo_to_ego_mode)

        self._set_freeze(
            freeze_backbone=freeze_backbone,
            freeze_exo_head=freeze_exo_head,
            freeze_ego_head=freeze_ego_head,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg

        # Options
        self.ignored_weights_prefix = ignored_weights_prefix
        self.backbone_lr_scale = backbone_lr_scale
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks

        # The test step is the same as validation
        self.test_step = self.predict_step = self.validation_step

        # SMPLX (lite 版本用于训练)
        self.smplx = make_smplx("supermotion_v437coco17")
        
        # SMPLX (完整版本用于可视化，与 demo.py 保持一致)
        self.smplx_full = make_smplx("supermotion")

        # SMPLX to SMPL 转换矩阵和 J_regressor（与 demo.py 保持一致）
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
        
        # 将转换矩阵注册为 buffer，但不移动到设备（在可视化时动态移动到对应设备）

        self.vis_every_n_steps = vis_every_n_steps
        self.val_vis_every_n_batches = val_vis_every_n_batches
        self._val_vis_step = 0
        self._last_train_reproj_vis_step = -1
        self._last_val_reproj_vis_step = -1
    
    def _tb_step(self, tag_prefix):
        return self._val_vis_step if str(tag_prefix).startswith("val") else self.global_step

    def _add_tb_image(self, tag, image, tag_prefix):
        self.logger.experiment.add_image(tag, image, self._tb_step(tag_prefix))

    def _add_tb_figure(self, tag, fig, tag_prefix):
        self.logger.experiment.add_figure(tag, fig, self._tb_step(tag_prefix))

    def _copy_exo_to_ego(self, mode="pose"):
        den = self.pipeline.denoiser3d
        if not hasattr(den, "final_layer_ego") or den.final_layer_ego is None:
            Log.warning("final_layer_ego is None, skip copy_exo_to_ego. Set dual_head=True in network config to enable.")
            return

        def _copy_mlp_prefix(dst, src, out_dim):
            if dst is None or src is None:
                return 0
            if hasattr(dst, "fc1") and hasattr(src, "fc1") and dst.fc1.weight.shape == src.fc1.weight.shape:
                dst.fc1.load_state_dict(src.fc1.state_dict())
            n = min(int(out_dim), dst.fc2.weight.shape[0], src.fc2.weight.shape[0])
            with torch.no_grad():
                dst.fc2.weight[:n].copy_(src.fc2.weight[:n])
                dst.fc2.bias[:n].copy_(src.fc2.bias[:n])
            return n

        mode = str(mode or "pose")
        if mode == "none":
            return
        if mode == "pose":
            # Shared semantic part: body_pose r6d (0:126) + betas (126:136).
            # Do not copy exo global_orient_gv/local_vel into ego root_trans/residual slots.
            copied = _copy_mlp_prefix(den.final_layer_ego, den.final_layer, 136)
        elif mode in ("compatible", "prefix"):
            copied = _copy_mlp_prefix(
                den.final_layer_ego,
                den.final_layer,
                min(den.final_layer_ego.fc2.weight.shape[0], den.final_layer.fc2.weight.shape[0]),
            )
        else:
            raise ValueError(f"Unknown copy_exo_to_ego_mode={mode}")
        Log.info(f"[Init] Copied exo final_layer -> ego final_layer prefix dims: {copied} (mode={mode})")

        if mode != "pose":
            if hasattr(den, "pred_cam_head_ego") and den.pred_cam_head_ego is not None and den.pred_cam_head:
                den.pred_cam_head_ego.load_state_dict(den.pred_cam_head.state_dict())
            if hasattr(den, "static_conf_head_ego") and den.static_conf_head_ego is not None and den.static_conf_head:
                den.static_conf_head_ego.load_state_dict(den.static_conf_head.state_dict())
    
    def _set_freeze(self, freeze_backbone=False, freeze_exo_head=False, freeze_ego_head=True, unfreeze_last_n_blocks=0):
        den = self.pipeline.denoiser3d

        def _freeze_module(m, freeze):
            if m is None:
                return
            # 处理 nn.Parameter 和 nn.Module
            if isinstance(m, nn.Parameter):
                m.requires_grad = not freeze
            else:
                for p in m.parameters():
                    p.requires_grad = not freeze

        if freeze_backbone:
            _freeze_module(den.learned_pos_linear, True)
            _freeze_module(den.learned_pos_params, True)
            _freeze_module(den.embed_noisyobs, True)
            _freeze_module(getattr(den, "cliffcam_embedder", None), True)
            _freeze_module(getattr(den, "cam_angvel_embedder", None), True)
            _freeze_module(getattr(den, "imgseq_embedder", None), True)
            _freeze_module(den.blocks, True)
            # Keep newly introduced ego condition embedders trainable so stage-1
            # ego adaptation can learn to use CPF/head/PV cues while preserving
            # the original GVHMR exo backbone.
            _freeze_module(getattr(den, "ego_imgseq_embedder", None), False)
            _freeze_module(getattr(den, "cam_trans_vel_embedder", None), False)
            _freeze_module(getattr(den, "gravity_embedder", None), False)
            _freeze_module(getattr(den, "ego_head_embedder", None), False)
            _freeze_module(getattr(den, "ego_hand_embedder", None), False)
            _freeze_module(getattr(den, "interaction_embedder", None), False)
            _freeze_module(getattr(den, "cross_extrinsic_embedder", None), False)
            _freeze_module(getattr(den, "cross_attn", None), False)
            _freeze_module(getattr(den, "cross_q_norm", None), False)
            _freeze_module(getattr(den, "cross_kv_norm", None), False)
            _freeze_module(getattr(den, "cross_gate", None), False)
            _freeze_module(getattr(den, "cross_interactee_type", None), False)
            _freeze_module(getattr(den, "cross_wearer_type", None), False)
            n_unfreeze = int(unfreeze_last_n_blocks or 0)
            if n_unfreeze > 0:
                blocks = getattr(den, "blocks", None)
                try:
                    n_blocks = len(blocks)
                    for block in list(blocks)[max(0, n_blocks - n_unfreeze):]:
                        _freeze_module(block, False)
                    Log.info(f"[Freeze] Unfroze last {min(n_unfreeze, n_blocks)}/{n_blocks} shared transformer blocks")
                except Exception as e:
                    Log.warning(f"[Freeze] Failed to unfreeze last transformer blocks: {e}")

        _freeze_module(den.final_layer, freeze_exo_head)
        _freeze_module(getattr(den, "pred_cam_head", None), freeze_exo_head)
        _freeze_module(getattr(den, "static_conf_head", None), freeze_exo_head)

        _freeze_module(getattr(den, "final_layer_ego", None), freeze_ego_head)
        _freeze_module(getattr(den, "pred_cam_head_ego", None), freeze_ego_head)
        _freeze_module(getattr(den, "static_conf_head_ego", None), freeze_ego_head)

    def _effective_branch_mode(self, batch):
        mode = self.pipeline.args.get("branch_mode", "both")
        is_paired = "exo" in batch and "ego" in batch
        if mode == "auto":
            return "ego" if is_paired else "exo"
        if mode in ("ego", "both") and not is_paired:
            return "exo"
        return mode

    def _default_loss_log_role(self, batch):
        role = batch.get("_supervise_role", None)
        if role == "both" and self.pipeline.args.get("loss_role_alias", "ego_exo") == "wearer_partner":
            return "teacher"
        if role not in ("ego", "exo"):
            role = batch.get("_input_role", self.pipeline.args.get("input_role", None))
        return role if role in ("ego", "exo") else "exo"

    def _loss_component_tag(self, key, default_role):
        alias = self.pipeline.args.get("loss_role_alias", "ego_exo")
        ego_role = "wearer" if alias == "wearer_partner" else "ego"
        exo_role = "partner" if alias == "wearer_partner" else "exo"
        if key == "loss":
            return default_role, "loss"
        if key.endswith("_ego"):
            return ego_role, key[:-4]
        if key.startswith("ego_"):
            return ego_role, key[4:]
        return exo_role, key

    def _log_loss_outputs(self, tag_prefix, outputs, log_kwargs, default_role):
        if "loss" in outputs:
            self.log(f"{tag_prefix}/loss", outputs["loss"], **log_kwargs)
            self.log(f"{tag_prefix}/{default_role}/loss", outputs["loss"], **log_kwargs)
        for k, v in outputs.items():
            if k.startswith("cross_view_") and torch.is_tensor(v) and v.numel() == 1:
                self.log(f"{tag_prefix}/cross_view/{k[len('cross_view_'):]}", v, **log_kwargs)
                continue
            if "_loss" not in k:
                continue
            role, name = self._loss_component_tag(k, default_role)
            self.log(f"{tag_prefix}/{role}/{name}", v, **log_kwargs)

    def _slice_batch_value(self, value, index, batch_size):
        if torch.is_tensor(value):
            if value.ndim > 0 and value.shape[0] == batch_size:
                return value.index_select(0, index.to(value.device))
            return value
        if isinstance(value, dict):
            return {k: self._slice_batch_value(v, index, batch_size) for k, v in value.items()}
        if isinstance(value, list) and len(value) == batch_size:
            idx_cpu = index.detach().cpu().tolist()
            return [value[i] for i in idx_cpu]
        if isinstance(value, tuple) and len(value) == batch_size:
            idx_cpu = index.detach().cpu().tolist()
            return tuple(value[i] for i in idx_cpu)
        return value

    def _slice_batch(self, batch, index):
        batch_size = int(batch["ego"]["smpl_params_c"]["body_pose"].shape[0]) if "ego" in batch else int(batch["smpl_params_c"]["body_pose"].shape[0])
        out = {k: self._slice_batch_value(v, index, batch_size) for k, v in batch.items()}
        out["B"] = int(index.numel())
        return out

    def _training_step_mixed_batch(self, batch, batch_idx):
        B = int(batch["ego"]["smpl_params_c"]["body_pose"].shape[0])
        ego_prob = float(self.pipeline.args.get("train_ego_prob", 0.5))
        if B <= 1:
            role = "ego" if torch.rand((), device=self.device).item() < ego_prob else "exo"
            batch["_force_input_role"] = role
            batch["_force_supervise_role"] = role
            return self.training_step(batch, batch_idx)

        role_mask = torch.rand(B, device=self.device) < ego_prob
        if not role_mask.any():
            role_mask[torch.randint(B, (), device=self.device)] = True
        if role_mask.all():
            role_mask[torch.randint(B, (), device=self.device)] = False

        outputs_by_role = []
        for role, mask in (("ego", role_mask), ("exo", ~role_mask)):
            index = torch.nonzero(mask, as_tuple=False).flatten()
            if index.numel() == 0:
                continue
            sub_batch = self._slice_batch(batch, index)
            sub_batch["_force_input_role"] = role
            sub_batch["_force_supervise_role"] = role
            outputs = self.training_step(sub_batch, batch_idx)
            outputs_by_role.append((role, int(index.numel()), outputs))

        mixed_loss = sum(outputs["loss"] * (n / B) for _, n, outputs in outputs_by_role)
        log_kwargs = {
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "batch_size": B,
            "sync_dist": True,
        }
        self.log("train/mixed/loss", mixed_loss, **log_kwargs)
        self.log("train/mixed/ego_batch", torch.as_tensor(int(role_mask.sum()), device=self.device, dtype=mixed_loss.dtype), **log_kwargs)
        self.log("train/mixed/exo_batch", torch.as_tensor(int((~role_mask).sum()), device=self.device, dtype=mixed_loss.dtype), **log_kwargs)
        return {"loss": mixed_loss}

    def training_step(self, batch, batch_idx):
        forced_input_role = batch.pop("_force_input_role", None)
        forced_supervise_role = batch.pop("_force_supervise_role", None)
        is_paired = "exo" in batch and "ego" in batch
        branch_mode = self._effective_branch_mode(batch)
        input_role = branch_mode
        if forced_input_role is None and is_paired and branch_mode == "both":
            train_input_role = self.pipeline.args.get("train_input_role", self.pipeline.args.get("input_role", "exo"))
            if train_input_role == "mixed":
                return self._training_step_mixed_batch(batch, batch_idx)
            if train_input_role == "alternate":
                input_role = "ego" if (int(self.trainer.global_step) % 2 == 0) else "exo"
            elif train_input_role == "random":
                ego_prob = float(self.pipeline.args.get("train_ego_prob", 0.5))
                input_role = "ego" if torch.rand((), device=self.device).item() < ego_prob else "exo"
            elif train_input_role in ("ego", "exo"):
                input_role = train_input_role
            else:
                input_role = "exo"
        elif forced_input_role in ("ego", "exo"):
            input_role = forced_input_role
        if is_paired:
            batch.setdefault("K_exo", batch["K_fullimg"])
            supervise_role = forced_supervise_role
            if supervise_role not in ("ego", "exo", "both"):
                supervise_role = self.pipeline.args.get("supervise_role", None)
            if supervise_role not in ("ego", "exo", "both"):
                supervise_role = input_role if branch_mode == "both" else branch_mode
            batch["_input_role"] = input_role
            batch["_supervise_role"] = supervise_role
            if input_role == "ego":
                primary = batch["ego"]
                batch["smpl_params_c"] = primary["smpl_params_c"]
                batch["smpl_params_w"] = primary["smpl_params_w"]
                batch["interactee_smpl_params_c"] = primary["smpl_params_c"]
                batch["interactee_smpl_params_w"] = primary["smpl_params_w"]
                batch["bbx_xys"] = primary.get("bbx_body_xys", primary["bbx_xys"])
                batch["kp2d"] = primary.get("kp2d_body", primary.get("kp2d", torch.zeros_like(batch["bbx_xys"][:, :, None].expand(-1, -1, 17, -1))))
                batch["f_imgseq"] = primary.get("f_body_imgseq", primary["f_imgseq"])
                batch["K_fullimg"] = batch.get("K_ego", batch["K_fullimg"])
                if "ego_imgname" in batch:
                    batch["imgname"] = batch["ego_imgname"]
                batch["mask"]["valid"] = batch["mask"].get("ego_valid", batch["mask"]["valid"])
            else:
                primary = batch["exo"]
                batch["smpl_params_c"] = primary["smpl_params_c"]
                batch["smpl_params_w"] = primary["smpl_params_w"]
                batch["interactee_smpl_params_c"] = primary["smpl_params_c"]
                batch["interactee_smpl_params_w"] = primary["smpl_params_w"]
                batch["bbx_xys"] = primary["bbx_xys"]
                batch["kp2d"] = primary.get("kp2d", torch.zeros_like(primary["bbx_xys"][:, :, None].expand(-1, -1, 17, -1)))
                batch["f_imgseq"] = primary["f_imgseq"]
                batch["K_fullimg"] = batch.get("K_exo", batch["K_fullimg"])
                if "exo_imgname" in batch:
                    batch["imgname"] = batch["exo_imgname"]
                batch["mask"]["valid"] = batch["mask"].get("exo_valid", batch["mask"]["valid"])
            valid_mask = batch["mask"]["valid"]
            batch["mask"].setdefault("spv_incam_only", torch.zeros_like(valid_mask, dtype=torch.bool))
            batch["mask"].setdefault("vitpose", torch.zeros_like(valid_mask, dtype=torch.bool))
            batch["mask"].setdefault("bbx_xys", torch.ones_like(valid_mask, dtype=torch.bool))
            batch["mask"].setdefault("f_imgseq", torch.ones_like(valid_mask, dtype=torch.bool))

        B, F = batch["smpl_params_c"]["body_pose"].shape[:2]

        # Create augmented noisy-obs : gt_j3d(coco17)
        with torch.no_grad():
            gt_verts437, gt_j3d = self.smplx(**_body_smpl_params(batch["interactee_smpl_params_c"]))
            root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
            batch["gt_j3d"] = gt_j3d
            batch["gt_cr_coco17"] = gt_j3d - root_
            batch["gt_c_verts437"] = gt_verts437
            batch["gt_cr_verts437"] = gt_verts437 - root_
            if is_paired:
                ego_verts437, ego_j3d = self.smplx(**_body_smpl_params(batch["ego"]["smpl_params_c"]))
                ego_root = ego_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
                batch["ego_gt_j3d"] = ego_j3d
                batch["ego_gt_cr_verts437"] = ego_verts437 - ego_root

        if input_role == "ego" and is_paired:
            obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
            obs[~batch["mask"]["valid"]] = 0
            batch["obs"] = obs
        else:
            # bbx_xys
            i_x2d = safely_render_x3d_K(gt_verts437, batch["K_fullimg"], thr=0.3)
            bbx_xys = get_bbx_xys(i_x2d, do_augment=True)
            if False:  # trust image bbx_xys seems better
                batch["bbx_xys"] = bbx_xys
            else:
                mask_bbx_xys = batch["mask"]["bbx_xys"]
                batch["bbx_xys"][~mask_bbx_xys] = bbx_xys[~mask_bbx_xys]
            # noisy_j3d -> project to i_j2d -> compute a bbx -> normalized kp2d [-1, 1]  让观测空间更像真实世界，不用真实的detector，而是从gt生成可控的伪检测；在val的时候就是真实的detector获得的2d关键点了
            noisy_j3d = gt_j3d + get_wham_aug_kp3d(gt_j3d.shape[:2])
            batch["noisy_j3d"] = noisy_j3d
            if True:
                noisy_j3d = randomly_modify_hands_legs(noisy_j3d)
            obs_i_j2d = perspective_projection(noisy_j3d, batch["K_fullimg"])  # (B, L, J, 2)
            j2d_visible_mask = get_visible_mask(gt_j3d.shape[:2]).cuda()  # (B, L, J)
            j2d_visible_mask[noisy_j3d[..., 2] < 0.3] = False  # Set close-to-image-plane points as invisible
            if True:  # Set both legs as invisible for a period
                legs_invisible_mask = get_invisible_legs_mask(gt_j3d.shape[:2]).cuda()  # (B, L, J)
                j2d_visible_mask[legs_invisible_mask] = False
            obs_kp2d = torch.cat([obs_i_j2d, j2d_visible_mask[:, :, :, None].float()], dim=-1)  # (B, L, J, 3)
            obs = normalize_kp2d(obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)       主要是为了学习人体姿态，要避免摄像机距离、图像分辨率等的干扰
            obs[~j2d_visible_mask] = 0  # if not visible, set to (0,0,0)
            batch["obs"] = obs
            # batch["kp2d"] = torch.zeros(B, 17, 3)   # 训练阶段不用
            if True:  # Use some detected vitpose (presave data)这个后面可以用VitPose检测之后再加入训练
                prob = 0.5
                mask_real_vitpose = (torch.rand(B, 1, device=obs_kp2d.device) < prob) & batch["mask"]["vitpose"].bool()
                batch["obs"][mask_real_vitpose] = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])[mask_real_vitpose]

            # Set untrusted frames to False
            batch["obs"][~batch["mask"]["valid"]] = 0

        # f_imgseq: apply random aug on offline extracted features
        # f_imgseq = batch["f_imgseq"] + torch.randn_like(batch["f_imgseq"]) * 0.1
        # f_imgseq[~batch["mask"]["f_imgseq"]] = 0
        # batch["f_imgseq"] = f_imgseq.clone()

        # Forward and get loss
        outputs = self.pipeline.forward(batch, train=True)

        # ========================================================
        # 训练过程可视化 (TensorBoard)
        # ========================================================
        # 使用 self.trainer.global_step 而不是 self.global_step
        current_step = int(self.trainer.global_step)
        vis_interval = max(int(self.vis_every_n_steps), 1)
        input_role = batch.get("_input_role", branch_mode)
        train_vis_due = current_step % vis_interval == 0
        train_reproj_due = (
            is_paired
            and input_role == "ego"
            and (self._last_train_reproj_vis_step < 0 or current_step - self._last_train_reproj_vis_step >= vis_interval)
        )
        if self.logger is not None and (train_vis_due or train_reproj_due):
            Log.info(f"[Vis] Triggering lightweight train visualization at step {current_step}")
            try:
                if is_paired and input_role == "ego":
                    self._visualize_ego_pv_input(batch, tag_prefix="train", vis_frames=4)
                    self._visualize_ego_pv_reprojection(batch, outputs, tag_prefix="train", vis_frames=4)
                    self._last_train_reproj_vis_step = current_step
                if train_vis_due and ((not is_paired) or input_role == "exo"):
                    self._visualize_model_output(batch, outputs, tag_prefix="train")
            except Exception as e:
                Log.warning(f"[Vis] Visualization failed: {e}")
                import traceback
                Log.warning(traceback.format_exc())

        # Log
        log_kwargs = {
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "batch_size": B,
            "sync_dist": True,
        }
        self._log_loss_outputs("train", outputs, log_kwargs, default_role=self._default_loss_log_role(batch))

        return outputs

    def _resolve_image_path(self, imgname_item):
        """将 batch["imgname"] 中的路径项解析为绝对路径，支持相对路径和绝对路径"""
        img_path = str(imgname_item)
        if not Path(img_path).is_absolute():
            possible_roots = [
                Path("/public/home/wenxin/egobody"),
                Path.cwd(),
            ]
            for root in possible_roots:
                full_path = root / img_path
                if full_path.exists():
                    return str(full_path)
        return img_path

    def _visualize_model_output(self, batch, outputs, tag_prefix="train"):
        """可视化模型预测输出: incam overlay（interactee 的 2D 可视化）
        
        Args:
            batch: 数据batch
            outputs: 模型输出
            tag_prefix: TensorBoard 日志路径前缀，训练阶段用 "train"，验证阶段用 "val"
        """
        B, F = batch["smpl_params_c"]["body_pose"].shape[:2]
        device = batch["smpl_params_c"]["body_pose"].device

        pred_smpl_params_incam = outputs.get("pred_smpl_params_incam", None)
        if pred_smpl_params_incam is None:
            return
        
        exo_incam_gt_key = "interactee_smpl_params_c" 
        if exo_incam_gt_key not in batch or batch.get(exo_incam_gt_key) is None:
            return
        
        with torch.no_grad():
            # 预测
            smpl_params_reshaped = {k: v.reshape(B * F, -1) for k, v in pred_smpl_params_incam.items()}
            smplx_out_pred = self.smplx_full(**_body_smpl_params(smpl_params_reshaped))
            pred_verts = smplx_out_pred.vertices.reshape(B, F, -1, 3)
            # 关键点使用 437 模型（只有 17 个 COCO 关节点，用于 draw_coco17_skeleton_batch）
            _, pred_j3d = self.smplx(**_body_smpl_params(smpl_params_reshaped))
            pred_j3d = pred_j3d.reshape(B, F, -1, 3)
            
            # GT
            gt_params_reshaped = {k: v.reshape(B * F, -1) for k, v in batch[exo_incam_gt_key].items()}
            gt_smplx_out = self.smplx_full(**_body_smpl_params(gt_params_reshaped))
            gt_verts = gt_smplx_out.vertices.reshape(B, F, -1, 3)
            _, gt_j3d = self.smplx(**_body_smpl_params(batch[exo_incam_gt_key]))
            gt_j3d = gt_j3d.reshape(B, F, -1, 3)
            
            # # Debug: 检查 GT 数据
            # Log.info(f"[Vis] GT verts range: [{gt_verts.min():.3f}, {gt_verts.max():.3f}]")
            # Log.info(f"[Vis] GT j3d range: [{gt_j3d.min():.3f}, {gt_j3d.max():.3f}]")
            # Log.info(f"[Vis] GT transl[0,0]: {batch[exo_incam_gt_key]['transl'][0, 0]}")
            
            # 渲染 incam overlay
            self._render_incam_overlay(
                batch, pred_verts, pred_j3d, gt_verts, gt_j3d, 
                tag_prefix=tag_prefix,
                vis_frames=2
            )

    def _render_incam_overlay(self, batch, pred_verts, pred_joints, gt_verts, gt_j3d, tag_prefix="train", vis_frames=8):
        """
            可视化多帧:
                1. pred 
                2. gt 
                3. pred/gt mesh compare
            
            Args:
                tag_prefix: TensorBoard 日志路径前缀，训练阶段用 "train"，验证阶段用 "val"
                vis_frames: 可视化的帧数
        """
        from hmr4d.utils.vis.renderer import Renderer
        
        imgname_list = batch.get("imgname", None)
        if imgname_list is None or len(imgname_list) == 0:
            Log.warning("[Vis] imgname_list is empty")
            return

        B, F = pred_verts.shape[:2]
        
        # 获取相机内参和 bbox (所有帧)
        K_fullimg = batch["K_fullimg"]  # (B, F, 3, 3) 或 (B, F, 3, 3)
        bbx_xys = batch["bbx_xys"]  # (B, F, 4) 或 (B, F, 4)
        
        # 选择要可视化的帧索引
        frame_indices = np.linspace(0, F - 1, vis_frames, dtype=int)
        
        # Collect the two exo visualizations we keep: keypoints and mesh overlay.
        kp_compare_overlays = []
        mesh_compare_overlays = []
        
        for idx_in_vis, frame_idx in enumerate(frame_indices):
            # 读取当前帧的图像
            if isinstance(imgname_list, list) and len(imgname_list) > 0:
                if isinstance(imgname_list[0], list):
                    # imgname_list 是嵌套列表 [batch][frame]
                    if len(imgname_list[0]) > frame_idx:
                        img_path = self._resolve_image_path(imgname_list[0][frame_idx])
                    else:
                        img_path = self._resolve_image_path(imgname_list[0][0])
                else:
                    # imgname_list 是扁平列表，需要计算索引
                    if len(imgname_list) > frame_idx:
                        img_path = self._resolve_image_path(imgname_list[frame_idx])
                    else:
                        img_path = self._resolve_image_path(imgname_list[0])
            else:
                Log.warning(f"[Vis] Cannot get image for frame {frame_idx}")
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                Log.warning(f"[Vis] Failed to read image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]
            
            # 获取当前帧的相机内参
            if K_fullimg.ndim == 4:
                K = K_fullimg[0, frame_idx]  # (B, F, 3, 3) -> (3, 3)
            elif K_fullimg.ndim == 3:
                K = K_fullimg[frame_idx]  # (F, 3, 3) -> (3, 3)
            else:
                K = K_fullimg[0]  # fallback
            
            # 获取当前帧的 bbox
            if bbx_xys.ndim == 3:
                bbx = bbx_xys[0, frame_idx]  # (B, F, 4) -> (4,)
            elif bbx_xys.ndim == 2:
                bbx = bbx_xys[frame_idx]  # (F, 4) -> (4,)
            else:
                bbx = bbx_xys[0]  # fallback
            bbx = bbx.cpu().numpy()
            
            # 创建渲染器
            renderer = Renderer(W, H, device="cpu", faces=self.smplx_full.faces, K=K)
            K_cpu = K.cpu() if isinstance(K, torch.Tensor) else K
            
            # 取当前帧的顶点和关节点 (batch 0)
            pred_verts_f = pred_verts[0, frame_idx].detach().cpu().float()
            pred_joints_f = pred_joints[0, frame_idx].detach().cpu().float()

            pred_joints_tensor = pred_joints_f if isinstance(pred_joints_f, torch.Tensor) else torch.from_numpy(pred_joints_f)
            pred_j2d = perspective_projection(pred_joints_tensor.unsqueeze(0).unsqueeze(0), K_cpu.unsqueeze(0).unsqueeze(0))
            pred_j2d = pred_j2d.squeeze(0).squeeze(0).numpy()

            # Render GT and prediction comparisons.
            if gt_verts is not None:
                gt_verts_f = gt_verts[0, frame_idx].detach().cpu().float()
                gt_joints_f = gt_j3d[0, frame_idx].detach().cpu().float()
                gt_img = renderer.render_mesh(gt_verts_f, background=img_rgb.copy(), colors=[1.0, 0.3, 0.3])
                gt_joints_tensor = gt_joints_f if isinstance(gt_joints_f, torch.Tensor) else torch.from_numpy(gt_joints_f)
                gt_j2d = perspective_projection(gt_joints_tensor.unsqueeze(0).unsqueeze(0), K_cpu.unsqueeze(0).unsqueeze(0))
                gt_j2d = gt_j2d.squeeze(0).squeeze(0).numpy()
                
                if idx_in_vis == 0:
                    Log.info(f"[Vis] gt_j2d range: [{gt_j2d[:, 0].min():.1f}, {gt_j2d[:, 0].max():.1f}] x [{gt_j2d[:, 1].min():.1f}, {gt_j2d[:, 1].max():.1f}]")
                    Log.info(f"[Vis] gt_joints z range: [{gt_joints_tensor[:, 2].min():.3f}, {gt_joints_tensor[:, 2].max():.3f}]")
                
                def _put_legend(img, entries):
                    y = 28
                    for text, color in entries:
                        cv2.putText(img, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                        y += 30
                    cv2.putText(img, f"frame {int(frame_idx)}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                    return img

                def _draw_skeleton(img, pts, color):
                    edges = [
                        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
                        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
                        (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
                    ]
                    for a, b in edges:
                        if a < len(pts) and b < len(pts):
                            xa, ya = pts[a, :2]
                            xb, yb = pts[b, :2]
                            if np.isfinite([xa, ya, xb, yb]).all():
                                cv2.line(
                                    img,
                                    (int(round(xa)), int(round(ya))),
                                    (int(round(xb)), int(round(yb))),
                                    color,
                                    3,
                                    lineType=cv2.LINE_AA,
                                )
                    for x, y in pts[:, :2]:
                        if np.isfinite(x) and np.isfinite(y):
                            cv2.circle(img, (int(round(x)), int(round(y))), 5, color, -1, lineType=cv2.LINE_AA)
                    return img

                # unified kp compare: GT and pred skeletons on the original image.
                kp_compare = img_rgb.copy()
                kp_compare = _draw_skeleton(kp_compare, gt_j2d, (0, 220, 0))
                kp_compare = _draw_skeleton(kp_compare, pred_j2d, (255, 160, 0))
                kp_compare = draw_bbx_xys_on_image_batch([bbx], [kp_compare])[0]
                kp_compare = _put_legend(kp_compare, [("exo GT kp", (0, 220, 0)), ("exo pred kp", (255, 160, 0))])
                kp_compare_overlays.append(kp_compare)

                # unified mesh compare: GT and pred meshes on the original image.
                mesh_compare = renderer.render_mesh(gt_verts_f, background=img_rgb.copy(), colors=[0.1, 0.8, 0.25])
                mesh_compare = renderer.render_mesh(pred_verts_f, background=mesh_compare, colors=[1.0, 0.55, 0.05])
                mesh_compare = _put_legend(mesh_compare, [("exo GT mesh", (0, 220, 0)), ("exo pred mesh", (255, 160, 0))])
                mesh_compare_overlays.append(mesh_compare)

            
            del renderer
        
        # 将多帧图像拼接成网格 (2行 x N列 或 N行 x 2列)
        def stack_images_horizontal(image_list):
            """水平拼接图像列表"""
            if len(image_list) == 0:
                return np.zeros((H, W, 3), dtype=np.uint8)
            return np.concatenate(image_list, axis=1)
        
        def stack_images_vertical(image_list):
            """垂直拼接图像列表"""
            if len(image_list) == 0:
                return np.zeros((H, W, 3), dtype=np.uint8)
            return np.concatenate(image_list, axis=0)
        
        # 对于多帧，我们创建一个网格布局
        # 如果帧数 <= 4，使用 2x2 或 1xN 布局
        # 如果帧数 > 4，使用多行布局
        
        n_frames = len(kp_compare_overlays)
        if n_frames <= 4:
            if kp_compare_overlays:
                kp_compare_combined = stack_images_horizontal(kp_compare_overlays)
            if mesh_compare_overlays:
                mesh_compare_combined = stack_images_horizontal(mesh_compare_overlays)
        else:
            cols = 4
            rows = (n_frames + cols - 1) // cols
            kp_compare_rows = []
            mesh_compare_rows = []
            for r in range(rows):
                start_idx = r * cols
                end_idx = min(start_idx + cols, n_frames)
                if kp_compare_overlays:
                    kp_compare_rows.append(stack_images_horizontal(kp_compare_overlays[start_idx:end_idx]))
                if mesh_compare_overlays:
                    mesh_compare_rows.append(stack_images_horizontal(mesh_compare_overlays[start_idx:end_idx]))
            if kp_compare_rows:
                kp_compare_combined = stack_images_vertical(kp_compare_rows)
            if mesh_compare_rows:
                mesh_compare_combined = stack_images_vertical(mesh_compare_rows)

        if kp_compare_overlays:
            kp_tb = torch.from_numpy(kp_compare_combined).permute(2, 0, 1)
            self._add_tb_image(f"{tag_prefix}/exo_incam_kp_gt_pred_multi", kp_tb, tag_prefix)
        if mesh_compare_overlays:
            mesh_tb = torch.from_numpy(mesh_compare_combined).permute(2, 0, 1)
            self._add_tb_image(f"{tag_prefix}/exo_incam_mesh_gt_pred_multi", mesh_tb, tag_prefix)


    def _project_smpl_joints_to_kp2d(self, smpl_params, K):
        with torch.no_grad():
            first = next(iter(smpl_params.values()))
            has_batch = first.ndim == 3
            if has_batch:
                B, F = first.shape[:2]
                flat_params = {k: v.reshape(B * F, -1) for k, v in smpl_params.items()}
            else:
                F = first.shape[0]
                B = 1
                flat_params = {k: v.reshape(F, -1) for k, v in smpl_params.items()}
            _, j3d = self.smplx(**_body_smpl_params(flat_params))
            j3d = j3d.reshape(B, F, -1, 3)
            if K.ndim == 3:
                K = K[None]
            j2d = perspective_projection(j3d, K)
            conf = (j3d[..., 2:3] > 0.1).float()
            return torch.cat([j2d, conf], dim=-1)

    def _params_to_smplx_verts_batched(self, smpl_params):
        if smpl_params is None:
            return None
        with torch.no_grad():
            first = next(iter(smpl_params.values()))
            if first.ndim == 3:
                B, F = first.shape[:2]
                flat_params = {k: v.reshape(B * F, -1) for k, v in smpl_params.items()}
                verts = self.smplx_full(**_body_smpl_params(flat_params)).vertices.reshape(B, F, -1, 3)
            else:
                F = first.shape[0]
                flat_params = {k: v.reshape(F, -1) for k, v in smpl_params.items()}
                verts = self.smplx_full(**_body_smpl_params(flat_params)).vertices.reshape(1, F, -1, 3)
            return verts.float()

    def _train_verts_to_raw_world(self, verts_train, ego_cond):
        if verts_train is None:
            return None
        T_raw_to_train = None
        if isinstance(ego_cond, dict):
            T_raw_to_train = ego_cond.get("T_raw_to_train_world", None)
        if T_raw_to_train is None:
            # Legacy fallback for old y/z-flipped worlds.
            verts_raw = verts_train.clone()
            verts_raw[..., 1] = -verts_raw[..., 1]
            verts_raw[..., 2] = -verts_raw[..., 2]
            return verts_raw
        if T_raw_to_train.ndim == 3:
            T_raw_to_train = T_raw_to_train[None]
        B = min(verts_train.shape[0], T_raw_to_train.shape[0])
        F = min(verts_train.shape[1], T_raw_to_train.shape[1])
        verts_train = verts_train[:B, :F]
        T_raw_to_train = T_raw_to_train[:B, :F].to(verts_train.device, dtype=verts_train.dtype)
        R = T_raw_to_train[..., :3, :3]
        t = T_raw_to_train[..., :3, 3]
        return torch.einsum("bfij,bfvj->bfvi", R.mT, verts_train - t[:, :, None])

    def _world_verts_to_ego_pv_cam(self, verts_world, T_world_pv):
        if verts_world is None or T_world_pv is None:
            return None
        verts_world = verts_world.float()
        T_world_pv = T_world_pv.to(device=verts_world.device).float()
        if T_world_pv.ndim == 3:
            T_world_pv = T_world_pv[None]
        B = min(verts_world.shape[0], T_world_pv.shape[0])
        F = min(verts_world.shape[1], T_world_pv.shape[1])
        verts_world = verts_world[:B, :F]
        T_world_pv = T_world_pv[:B, :F]
        R_world_pv = T_world_pv[..., :3, :3]
        t_world_pv = T_world_pv[..., :3, 3]
        verts_pv = torch.einsum("bfvj,bfjk->bfvk", verts_world - t_world_pv[:, :, None], R_world_pv)
        cam_conv = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=verts_world.device, dtype=verts_world.dtype))
        return torch.einsum("bfvj,kj->bfvk", verts_pv, cam_conv)

    def _draw_coco17_colored(self, img, kp, color, conf_thr=0.5, thickness=3, radius=4):
        img = img.copy()
        skel = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
        for i, j in skel:
            if kp[i, 2] > conf_thr and kp[j, 2] > conf_thr:
                p1 = tuple(kp[i, :2].astype(int).tolist())
                p2 = tuple(kp[j, :2].astype(int).tolist())
                cv2.line(img, p1, p2, color, thickness)
        for j in range(kp.shape[0]):
            if kp[j, 2] > conf_thr:
                p = tuple(kp[j, :2].astype(int).tolist())
                cv2.circle(img, p, radius, color, -1)
        return img

    def _render_mesh_alpha(self, renderer, verts, background, color, alpha=0.45):
        rendered = renderer.render_mesh(verts, background=background.copy(), colors=color)
        diff = np.abs(rendered.astype(np.int16) - background.astype(np.int16)).sum(axis=-1)
        mask = diff > 12
        out = background.copy()
        out[mask] = (alpha * rendered[mask].astype(np.float32) + (1.0 - alpha) * background[mask].astype(np.float32)).astype(np.uint8)
        return out

    def _mesh_proj_stats(self, verts, K, width, height):
        if verts is None or not torch.is_tensor(verts) or verts.numel() == 0:
            return None
        verts = verts.detach().float().cpu()
        z = verts[:, 2]
        valid = torch.isfinite(verts).all(dim=-1) & (z > 0.05)
        if valid.sum() < 8:
            return {"z_med": float(z[torch.isfinite(z)].median()) if torch.isfinite(z).any() else float("nan"), "area": float("nan"), "valid": int(valid.sum())}
        K_cpu = K.detach().float().cpu() if torch.is_tensor(K) else torch.as_tensor(K).float()
        xy = verts[valid, :2] / z[valid, None].clamp_min(1e-4)
        uv = xy @ K_cpu[:2, :2].T + K_cpu[:2, 2]
        in_img = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        uv_clip = uv.clone()
        uv_clip[:, 0].clamp_(0, width - 1)
        uv_clip[:, 1].clamp_(0, height - 1)
        wh = uv_clip.max(0)[0] - uv_clip.min(0)[0]
        return {
            "z_med": float(z[valid].median()),
            "area": float((wh[0] * wh[1]) / max(width * height, 1)),
            "valid": int(valid.sum()),
            "in_ratio": float(in_img.float().mean()),
        }

    def _visualize_ego_pv_reprojection(self, batch, outputs, tag_prefix="val", vis_frames=4):
        if self.logger is None or "ego_imgname" not in batch or "ego" not in batch:
            return
        imgname_list = batch.get("ego_imgname")
        if not imgname_list:
            return
        img_paths = imgname_list[0] if isinstance(imgname_list, list) and len(imgname_list) > 0 and isinstance(imgname_list[0], list) else imgname_list
        ego = batch["ego"]
        gt_kp = ego.get("kp2d_body", None)
        input_role = batch.get("_input_role", "ego" if self._effective_branch_mode(batch) == "ego" else "exo")
        pred_params_incam = outputs.get("pred_smpl_params_incam") if input_role == "ego" else None
        pred_params_incam = pred_params_incam or outputs.get("frozen_ego_image_exo_incam", None)
        if gt_kp is None or pred_params_incam is None:
            return

        K = batch.get("K_ego", batch.get("K_fullimg"))
        exo_for_pv = batch.get("exo", {})
        ego_cond_for_pv = batch.get("ego_cond", {})

        gt_verts_world = self._params_to_smplx_verts_batched(exo_for_pv.get("smpl_params_w_raw", exo_for_pv.get("smpl_params_w")))
        T_world_pv_raw = ego_cond_for_pv.get("T_world_pv_raw", ego_cond_for_pv.get("T_world_pv", None))
        gt_verts_cam = self._world_verts_to_ego_pv_cam(gt_verts_world, T_world_pv_raw)

        pred_verts_cam = self._params_to_smplx_verts_batched(pred_params_incam)
        if pred_verts_cam is None:
            pred_world_params = outputs.get("frozen_ego_image_exo_world_from_pv", outputs.get("frozen_ego_image_exo_kinect_from_incam", None))
            pred_verts_world = self._params_to_smplx_verts_batched(pred_world_params)
            pred_verts_raw = self._train_verts_to_raw_world(pred_verts_world, ego_cond_for_pv)
            pred_verts_cam = self._world_verts_to_ego_pv_cam(pred_verts_raw, T_world_pv_raw)
        if pred_verts_cam is None and "frozen_ego_image_exo_world_from_pv" in outputs:
            pred_verts_cam = self._world_verts_to_ego_pv_cam(
                self._params_to_smplx_verts_batched(outputs["frozen_ego_image_exo_world_from_pv"]),
                ego_cond_for_pv.get("T_world_pv", None),
            )

        F = gt_kp.shape[1] if gt_kp.ndim == 4 else gt_kp.shape[0]
        frame_indices = np.linspace(0, max(F - 1, 0), min(vis_frames, F), dtype=int)
        overlays = []
        for frame_idx in frame_indices:
            path_idx = min(int(frame_idx), len(img_paths) - 1)
            img = cv2.imread(self._resolve_image_path(img_paths[path_idx]))
            if img is None:
                continue
            overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = overlay.shape[:2]
            if K.ndim == 4:
                K_f = K[0, frame_idx]
            elif K.ndim == 3:
                K_f = K[frame_idx]
            else:
                K_f = K
            source_device = None
            if torch.is_tensor(K_f):
                source_device = K_f.device
            elif gt_verts_cam is not None and torch.is_tensor(gt_verts_cam):
                source_device = gt_verts_cam.device
            render_device = source_device if source_device is not None and source_device.type == "cuda" else torch.device("cpu")
            K_render = K_f.detach().to(render_device).float() if torch.is_tensor(K_f) else torch.as_tensor(K_f, device=render_device).float()
            renderer = Renderer(W, H, device=str(render_device), faces=self.smplx_full.faces, K=K_render)
            for attr in ("K", "K_full", "R", "T", "bboxes"):
                value = getattr(renderer, attr, None)
                if torch.is_tensor(value):
                    setattr(renderer, attr, value.to(device=render_device, dtype=torch.float32))
            renderer.cameras = renderer.create_camera()
            gt_stats = None
            pred_stats = None
            try:
                autocast_device = "cuda" if render_device.type == "cuda" else "cpu"
                with torch.autocast(device_type=autocast_device, enabled=False):
                    if gt_verts_cam is not None:
                        gt_mesh_cpu = gt_verts_cam[0, min(int(frame_idx), gt_verts_cam.shape[1] - 1)].detach().cpu().float()
                        gt_stats = self._mesh_proj_stats(gt_mesh_cpu, K_f, W, H)
                        if torch.isfinite(gt_mesh_cpu).all() and (gt_mesh_cpu[:, 2] > 0.05).any():
                            gt_mesh = gt_mesh_cpu.to(render_device, dtype=torch.float32)
                            overlay = renderer.render_mesh(gt_mesh, background=overlay.copy(), colors=[0.0, 0.8, 0.15])
                    if pred_verts_cam is not None:
                        pred_mesh_cpu = pred_verts_cam[0, min(int(frame_idx), pred_verts_cam.shape[1] - 1)].detach().cpu().float()
                        pred_stats = self._mesh_proj_stats(pred_mesh_cpu, K_f, W, H)
                        if torch.isfinite(pred_mesh_cpu).all() and (pred_mesh_cpu[:, 2] > 0.05).any():
                            pred_mesh = pred_mesh_cpu.to(render_device, dtype=torch.float32)
                            overlay = renderer.render_mesh(pred_mesh, background=overlay.copy(), colors=[1.0, 0.1, 0.1])
            except Exception as e:
                Log.warning(f"[Vis] ego PV mesh overlay failed at frame {frame_idx}: {e}")
            finally:
                del renderer

            cv2.putText(overlay, "GT mesh", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
            cv2.putText(overlay, "exo_pred_aux", (24, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 64, 64), 2)
            cv2.putText(overlay, f"frame {int(frame_idx)}", (24, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if gt_stats is not None and pred_stats is not None:
                diag = (
                    f"z gt/pred {gt_stats['z_med']:.2f}/{pred_stats['z_med']:.2f} "
                    f"area {gt_stats['area']:.2f}/{pred_stats['area']:.2f} "
                    f"in {gt_stats.get('in_ratio', 0):.2f}/{pred_stats.get('in_ratio', 0):.2f}"
                )
                cv2.putText(overlay, diag, (24, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            overlays.append(overlay)
        if not overlays:
            return
        H = min(img.shape[0] for img in overlays)
        resized = []
        for img in overlays:
            if img.shape[0] != H:
                scale = H / img.shape[0]
                img = cv2.resize(img, (int(img.shape[1] * scale), H))
            resized.append(img)
        combined = np.concatenate(resized, axis=1)
        tb = torch.from_numpy(combined).permute(2, 0, 1)
        self._add_tb_image(f"{tag_prefix}/ego_pv_reprojection_gt_vs_exo", tb, tag_prefix)

    def _add_world_state_figure(self, batch, outputs, tag_prefix="val", vis_frames=4):
        if self.logger is None or "ego" not in batch or "exo" not in batch:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            Log.warning(f"[Vis] matplotlib unavailable for world state figure: {e}")
            return

        def _joints_from_params(params):
            if params is None:
                return None
            with torch.no_grad():
                first = next(iter(params.values()))
                if first.ndim == 3:
                    B, F = first.shape[:2]
                    flat = {k: v.reshape(B * F, -1) for k, v in params.items()}
                    joints = self.pipeline.endecoder.fk_v2(**_body_smpl_params(flat)).reshape(B, F, -1, 3)[0]
                else:
                    F = first.shape[0]
                    flat = {k: v.reshape(F, -1) for k, v in params.items()}
                    joints = self.pipeline.endecoder.fk_v2(**_body_smpl_params(flat)).reshape(F, -1, 3)
                return joints.detach().float().cpu()

        def _filter_finite_joints(joints):
            if joints is None or joints.numel() == 0:
                return None
            finite = torch.isfinite(joints).all(dim=(-1, -2))
            if not finite.any():
                return None
            return joints[finite]

        input_role = batch.get("_input_role", self._effective_branch_mode(batch))
        exo_pred_label = "exo_pred"
        items = [
            ("exo_gt", _joints_from_params(outputs.get("exo_gt_smpl_params_w_aligned", batch.get("exo", {}).get("smpl_params_w", batch.get("interactee_smpl_params_w", batch.get("smpl_params_w"))))), "tab:blue"),
            ("ego_gt", _joints_from_params(outputs.get("ego_gt_smpl_params_w_aligned", batch.get("ego", {}).get("smpl_params_w"))), "tab:green"),
            ("ego_pred", _joints_from_params(outputs.get("pred_smpl_params_global_ego")), "tab:orange"),
            (exo_pred_label, _joints_from_params(outputs.get("pred_smpl_params_kinect_from_incam")), "tab:purple"),
            ("exo_pred_aux", _joints_from_params(outputs.get("frozen_ego_image_exo_world_from_pv", outputs.get("frozen_ego_image_exo_kinect_from_incam"))), "tab:red"),
        ]
        items = [(n, _filter_finite_joints(j), c) for n, j, c in items]
        items = [(n, j, c) for n, j, c in items if j is not None and j.numel() > 0]
        if not items:
            return
        F = max(j.shape[0] for _, j, _ in items)
        frame_indices = np.linspace(0, max(F - 1, 0), min(vis_frames, F), dtype=int)
        skel = [(15, 12), (12, 9), (16, 13), (13, 9), (9, 6), (6, 3), (3, 0), (14, 11), (11, 8), (8, 5), (5, 2), (2, 0), (10, 7), (7, 4), (4, 1), (1, 0)]
        fig = plt.figure(figsize=(4 * len(frame_indices), 4), dpi=120)
        all_pts = torch.cat([j[min(int(fi), j.shape[0]-1)] for _, j, _ in items for fi in frame_indices], dim=0)
        all_pts = all_pts[torch.isfinite(all_pts).all(dim=-1)]
        if all_pts.numel() == 0:
            plt.close(fig)
            return
        center = all_pts.mean(0)
        radius = float((all_pts - center).abs().max()) + 1e-4
        if not np.isfinite(radius):
            plt.close(fig)
            return
        for plot_i, frame_idx in enumerate(frame_indices, start=1):
            ax = fig.add_subplot(1, len(frame_indices), plot_i, projection="3d")
            for name, joints, color in items:
                ji = joints[min(int(frame_idx), joints.shape[0] - 1)]
                root = ji[0]
                ax.scatter(root[0], root[1], root[2], color=color, s=24, label=name)
                for a, b in skel:
                    if a < ji.shape[0] and b < ji.shape[0]:
                        ax.plot([ji[a, 0], ji[b, 0]], [ji[a, 1], ji[b, 1]], [ji[a, 2], ji[b, 2]], color=color, linewidth=1.5)
            ax.set_title(f"frame {int(frame_idx)}")
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            if plot_i == 1:
                ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        self._add_tb_figure(f"{tag_prefix}/world_ego_exo_state", fig, tag_prefix)
        plt.close(fig)

    def _params_to_smpl_verts(self, params):
        if params is None:
            return None
        with torch.no_grad():
            first = next(iter(params.values()))
            device = first.device
            if first.ndim == 3:
                B, F = first.shape[:2]
                flat = {k: v.reshape(B * F, -1) for k, v in params.items()}
                out = self.smplx_full(**_body_smpl_params(flat))
                verts_x = out.vertices.reshape(B, F, -1, 3)[0].float()
            else:
                F = first.shape[0]
                flat = {k: v.reshape(F, -1) for k, v in params.items()}
                out = self.smplx_full(**_body_smpl_params(flat))
                verts_x = out.vertices.reshape(F, -1, 3).float()
            smplx2smpl = self.smplx2smpl.to(device)
            if smplx2smpl.is_sparse:
                smplx2smpl = smplx2smpl.to_dense()
            smplx2smpl = smplx2smpl.float()
            return torch.einsum("sv,fvc->fsc", smplx2smpl, verts_x)

    def _batch_world_coord_name(self, batch):
        meta = batch.get("meta", None)
        if isinstance(meta, list) and len(meta) > 0:
            meta = meta[0]
        if isinstance(meta, dict):
            return str(meta.get("world_coord", "")).lower()
        return ""

    def _to_gvhmr_vis_world(self, x, world_coord=""):
        """Convert dataset camera-style world coords to GVHMR y-up coords for rendering only."""
        if x is None:
            return None
        out = x.clone()
        world_coord = str(world_coord).lower()
        if world_coord in ("gvhmr_yup", "yup", "canonical"):
            return out
        if world_coord in ("kinect12", "pv", "ego_pv", "holo", ""):
            out[..., 1] = -out[..., 1]
            out[..., 2] = -out[..., 2]
        return out

    def _project_world_points_for_vis(self, points_w, cam_R, cam_T, K):
        points_w = points_w.to(device=cam_R.device, dtype=cam_R.dtype)
        points_c = (cam_R @ points_w.T).T + cam_T[None]
        z = points_c[:, 2].clamp(min=1e-4)
        u = K[0, 0] * points_c[:, 0] / z + K[0, 2]
        v = K[1, 1] * points_c[:, 1] / z + K[1, 2]
        return torch.stack([u, v], dim=-1), points_c[:, 2]

    def _make_world_vis_camera(self, verts_f, width, height, device, K=None, padding=0.16):
        pts = verts_f.detach().float().reshape(-1, 3).cpu()
        finite = torch.isfinite(pts).all(dim=-1)
        pts = pts[finite]
        if pts.numel() == 0:
            target = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            radius = 2.0
        else:
            vmin = pts.min(0)[0]
            vmax = pts.max(0)[0]
            target = (vmin + vmax) * 0.5
            target[1] = max(float(vmin[1] + 0.85), 0.85)
            radius = max(float(torch.norm(vmax - vmin)) * 0.5, 1.0)

        view_dir = torch.tensor([0.75, 0.32, 0.92], dtype=torch.float32)
        view_dir = view_dir / view_dir.norm()
        target_dev = target.to(device).float()
        pts_dev = pts.to(device).float()
        K_dev = K.to(device).float() if K is not None else None

        distance = max(radius * 1.25, 2.2)
        max_distance = max(radius * 5.0, 8.0)
        best = None
        while distance <= max_distance:
            position = target_dev + view_dir.to(device) * distance
            position[1] = max(float(target[1] + radius * 0.35), float(position[1]))
            rotation = look_at_rotation(position[None].cpu(), target[None]).mT[0].to(device).float()
            translation = -(rotation @ position[:, None]).squeeze(-1)
            best = (rotation, translation)
            if K_dev is None or pts_dev.numel() == 0:
                break
            uv, z = self._project_world_points_for_vis(pts_dev, rotation, translation, K_dev)
            valid = z > 1e-3
            if valid.any():
                uv_valid = uv[valid]
                x0, y0 = uv_valid.min(dim=0)[0]
                x1, y1 = uv_valid.max(dim=0)[0]
                margin_x = width * padding
                margin_y = height * padding
                if x0 >= margin_x and x1 <= width - margin_x and y0 >= margin_y and y1 <= height - margin_y:
                    break
            distance *= 1.12
        if best is None:
            position = target_dev + view_dir.to(device) * max(radius * 2.2, 4.0)
            rotation = look_at_rotation(position[None].cpu(), target[None]).mT[0].to(device).float()
            translation = -(rotation @ position[:, None]).squeeze(-1)
            best = (rotation, translation)
        return best

    def _add_world_mesh_floor_image(self, batch, outputs, tag_prefix="val", vis_frames=4):
        if self.logger is None:
            return
        device = batch["smpl_params_w"]["body_pose"].device if "smpl_params_w" in batch else next(iter(batch["ego"]["smpl_params_w"].values())).device
        world_coord = self._batch_world_coord_name(batch)
        input_role = batch.get("_input_role", self._effective_branch_mode(batch))
        exo_pred_label = "exo_pred"
        main_mesh_items = [
            ("exo_gt", self._params_to_smpl_verts(outputs.get("exo_gt_smpl_params_w_aligned", batch.get("exo", {}).get("smpl_params_w", batch.get("interactee_smpl_params_w", batch.get("smpl_params_w"))))), torch.tensor([0.15, 0.45, 1.0], device=device)),
            ("ego_gt", self._params_to_smpl_verts(outputs.get("ego_gt_smpl_params_w_aligned", batch.get("ego", {}).get("smpl_params_w"))), torch.tensor([0.1, 0.75, 0.25], device=device)),
            ("ego_pred", self._params_to_smpl_verts(outputs.get("pred_smpl_params_global_ego")), torch.tensor([1.0, 0.55, 0.05], device=device)),
            (exo_pred_label, self._params_to_smpl_verts(outputs.get("pred_smpl_params_kinect_from_incam")), torch.tensor([0.55, 0.15, 0.85], device=device)),
        ]
        aux_mesh_items = [
            ("exo_pred_aux", self._params_to_smpl_verts(outputs.get("frozen_ego_image_exo_world_from_pv", outputs.get("frozen_ego_image_exo_kinect_from_incam"))), torch.tensor([1.0, 0.1, 0.1], device=device)),
        ]
        main_mesh_items = [(n, v, c) for n, v, c in main_mesh_items if v is not None and torch.isfinite(v).all()]
        aux_mesh_items = [(n, v, c) for n, v, c in aux_mesh_items if v is not None and torch.isfinite(v).all()]
        if not main_mesh_items:
            return

        main_mesh_items = [(n, self._to_gvhmr_vis_world(v, world_coord), c) for n, v, c in main_mesh_items]
        aux_mesh_items = [(n, self._to_gvhmr_vis_world(v, world_coord), c) for n, v, c in aux_mesh_items]
        cam_traj, cam_forward = self._camera_traj_from_batch(batch)
        cam_traj = self._to_gvhmr_vis_world(cam_traj, world_coord) if cam_traj is not None else None
        cam_forward = self._to_gvhmr_vis_world(cam_forward, world_coord) if cam_forward is not None else None
        ground_mesh_items = [(n, v, c) for n, v, c in main_mesh_items if n.endswith("_gt")] or main_mesh_items
        ground_y = torch.cat([v[..., 1].reshape(-1).detach().float().cpu() for _, v, _ in ground_mesh_items]).min()
        main_mesh_items = [(n, v.clone(), c) for n, v, c in main_mesh_items]
        aux_mesh_items = [(n, v.clone(), c) for n, v, c in aux_mesh_items]
        for _, verts, _ in main_mesh_items + aux_mesh_items:
            verts[..., 1] = verts[..., 1] - ground_y.to(verts.device, verts.dtype)
        if cam_traj is not None:
            cam_traj = cam_traj.clone()
            cam_traj[..., 1] = cam_traj[..., 1] - ground_y.to(cam_traj.device, cam_traj.dtype)

        F = max(v.shape[0] for _, v, _ in main_mesh_items)
        frame_indices = np.linspace(0, max(F - 1, 0), min(vis_frames, F), dtype=int)
        all_verts = torch.cat([v.detach().float().cpu() for _, v, _ in main_mesh_items], dim=0)
        J_regressor = self.J_regressor.to(all_verts.device).float()
        roots = einsum(J_regressor, all_verts, "j v, f v c -> f j c")[:, 0]
        scale, cx, cz = get_ground_params_from_points(roots, all_verts)
        scale = max(float(scale), 2.0)

        if aux_mesh_items:
            ref_root = einsum(J_regressor, main_mesh_items[0][1].detach().float().cpu(), "j v, f v c -> f j c")[:, 0]
            kept_aux = []
            for name, verts, color in aux_mesh_items:
                aux_root = einsum(J_regressor, verts.detach().float().cpu(), "j v, f v c -> f j c")[:, 0]
                d = torch.norm(aux_root - ref_root[: aux_root.shape[0]], dim=-1).median().item()
                if np.isfinite(d) and d < 6.0:
                    kept_aux.append((name, verts, color))
                else:
                    Log.warning(f"[Vis] skip {name} in world_mesh_floor: median root distance to exo_gt is {d:.2f}m")
            aux_mesh_items = kept_aux
        mesh_items = main_mesh_items + aux_mesh_items

        width, height = 960, 720
        images = []
        renderer = None
        try:
            _, _, K = create_camera_sensor(width, height, 24)
            K = K.to(device).float()
            faces_smpl = torch.as_tensor(make_smplx("smpl").faces, device=device).long()
            renderer = Renderer(width, height, device=str(device), faces=faces_smpl, K=K, bin_size=0)
            renderer.set_ground(scale * 3.0, cx, cz)
            light_verts = torch.cat([v.detach().float().cpu() for _, v, _ in main_mesh_items], dim=0)
            _, _, global_lights = get_global_cameras_static(
                light_verts, beta=3.2, cam_height_degree=25, target_center_height=1.0, device=str(device)
            )
            with torch.autocast(device_type="cuda", enabled=False):
                for frame_idx in frame_indices:
                    verts_f = []
                    colors_f = []
                    for _, verts, color in mesh_items:
                        fi = min(int(frame_idx), verts.shape[0] - 1)
                        verts_f.append(verts[fi].to(device).float())
                        colors_f.append(color.float())
                    verts_f = torch.stack(verts_f, dim=0)
                    colors_f = torch.stack(colors_f, dim=0)
                    fit_verts = torch.cat([v for v in verts_f], dim=0)
                    cam_R, cam_T = self._make_world_vis_camera(fit_verts, width, height, device, K=K, padding=0.16)
                    cameras = renderer.create_camera(cam_R, cam_T)
                    img = renderer.render_with_ground(verts_f, colors_f, cameras, global_lights)
                    img = np.ascontiguousarray(img)
                    if cam_traj is not None and cam_traj.numel() > 0:
                        ci = min(int(frame_idx), cam_traj.shape[0] - 1)
                        cam_p = cam_traj[ci].to(device).float()
                        marker_pts = [cam_p]
                        if cam_forward is not None and cam_forward.shape[0] > ci:
                            cam_f = cam_forward[ci].to(device).float()
                            cam_f = cam_f / cam_f.norm().clamp(min=1e-6)
                            marker_pts.append(cam_p + cam_f * 0.35)
                        marker_pts = torch.stack(marker_pts, dim=0)
                        uv, z = self._project_world_points_for_vis(marker_pts, cam_R, cam_T, K)
                        uv_np = uv.detach().cpu().numpy()
                        z_np = z.detach().cpu().numpy()
                        if z_np[0] > 1e-3 and np.isfinite(uv_np[0]).all():
                            p0 = tuple(np.round(uv_np[0]).astype(int).tolist())
                            if 0 <= p0[0] < width and 0 <= p0[1] < height:
                                cv2.circle(img, p0, 7, (0, 0, 0), -1)
                                cv2.circle(img, p0, 9, (255, 255, 255), 2)
                                cv2.putText(img, "cam", (p0[0] + 10, p0[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            if len(marker_pts) > 1 and z_np[1] > 1e-3 and np.isfinite(uv_np[1]).all():
                                p1 = tuple(np.round(uv_np[1]).astype(int).tolist())
                                cv2.line(img, p0, p1, (0, 0, 0), 3)
                                cv2.line(img, p0, p1, (255, 255, 255), 1)
                    y = 24
                    for name, _, color in mesh_items:
                        rgb = tuple((color.detach().cpu().numpy() * 255).astype(np.uint8).tolist())
                        cv2.putText(img, name, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rgb, 2)
                        y += 22
                    cv2.putText(img, f"frame {int(frame_idx)}", (width - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
                    images.append(img)
        except Exception as e:
            import traceback
            Log.warning(f"[Vis] world mesh-floor render failed, fallback to matplotlib: {e}")
            Log.warning(traceback.format_exc())
            fallback_images = self._render_world_mesh_floor_matplotlib(mesh_items, frame_indices, width=width, height=height)
            if fallback_images:
                fallback = np.concatenate(fallback_images, axis=1)
                fallback_tb = torch.from_numpy(fallback).permute(2, 0, 1)
                self._add_tb_image(f"{tag_prefix}/world_mesh_floor_fallback", fallback_tb, tag_prefix)
            return
        finally:
            if renderer is not None:
                del renderer
        if not images:
            Log.warning("[Vis] world_mesh_floor produced no images")
            return
        combined = np.concatenate(images, axis=1)
        tb = torch.from_numpy(combined).permute(2, 0, 1)
        self._add_tb_image(f"{tag_prefix}/world_mesh_floor", tb, tag_prefix)

    def _render_world_mesh_floor_matplotlib(self, mesh_items, frame_indices, width=960, height=720):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            Log.warning(f"[Vis] matplotlib fallback unavailable: {e}")
            return []
        images = []
        colors = {name: color.detach().cpu().numpy().clip(0, 1) for name, _, color in mesh_items}
        for frame_idx in frame_indices:
            fig = plt.figure(figsize=(width / 120, height / 120), dpi=120)
            ax = fig.add_subplot(111, projection="3d")
            pts_all = []
            for name, verts, _ in mesh_items:
                fi = min(int(frame_idx), verts.shape[0] - 1)
                v = verts[fi].detach().float().cpu()
                if v.numel() == 0 or not torch.isfinite(v).all():
                    continue
                pts_all.append(v)
                step = max(v.shape[0] // 1200, 1)
                vv = v[::step]
                c = colors[name]
                ax.scatter(vv[:, 0], vv[:, 2], vv[:, 1], s=1.0, color=c, alpha=0.85, label=name)
            if not pts_all:
                plt.close(fig)
                continue
            pts = torch.cat(pts_all, dim=0)
            center = pts.mean(0)
            radius = max(float((pts - center).abs().max()), 1.0)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[2] - radius, center[2] + radius)
            ax.set_zlim(0, max(float(pts[:, 1].max()) + 0.3, 2.0))
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_zlabel("y")
            ax.view_init(elev=18, azim=-55)
            ax.legend(loc="upper left", fontsize=7)
            ax.set_title(f"frame {int(frame_idx)}")
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            images.append(img)
            plt.close(fig)
        return images

    def _visualize_ego_pv_input(self, batch, tag_prefix="val", vis_frames=4):
        if self.logger is None or "ego_imgname" not in batch or "ego" not in batch:
            return
        imgname_list = batch.get("ego_imgname")
        if not imgname_list:
            return
        if isinstance(imgname_list, list) and len(imgname_list) > 0 and isinstance(imgname_list[0], list):
            img_paths = imgname_list[0]
        else:
            img_paths = imgname_list
        if len(img_paths) == 0:
            return

        ego = batch["ego"]
        bbx_xys = ego.get("bbx_body_xys", None)
        kp2d = ego.get("kp2d_body", None)
        if bbx_xys is None or kp2d is None:
            return
        F = bbx_xys.shape[1] if bbx_xys.ndim == 3 else bbx_xys.shape[0]
        frame_indices = np.linspace(0, max(F - 1, 0), min(vis_frames, F), dtype=int)
        overlays = []
        for frame_idx in frame_indices:
            path_idx = min(int(frame_idx), len(img_paths) - 1)
            img_path = self._resolve_image_path(img_paths[path_idx])
            img = cv2.imread(img_path)
            if img is None:
                Log.warning(f"[Vis] Failed to read ego PV image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if bbx_xys.ndim == 3:
                bbx = bbx_xys[0, frame_idx].detach().cpu().numpy()
            else:
                bbx = bbx_xys[frame_idx].detach().cpu().numpy()
            if kp2d.ndim == 4:
                kp = kp2d[0, frame_idx].detach().cpu().numpy()
            else:
                kp = kp2d[frame_idx].detach().cpu().numpy()
            overlay = draw_coco17_skeleton_batch([img_rgb], [kp], conf_thr=0.5)[0]
            overlay = draw_bbx_xys_on_image_batch([bbx], [overlay])[0]
            cv2.putText(overlay, "ego PV body input", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(overlay, f"frame {int(frame_idx)}", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            overlays.append(overlay)
        if not overlays:
            return
        H = min(img.shape[0] for img in overlays)
        resized = []
        for img in overlays:
            if img.shape[0] != H:
                scale = H / img.shape[0]
                img = cv2.resize(img, (int(img.shape[1] * scale), H))
            resized.append(img)
        combined = np.concatenate(resized, axis=1)
        tb = torch.from_numpy(combined).permute(2, 0, 1)
        self._add_tb_image(f"{tag_prefix}/ego_pv_body_input", tb, tag_prefix)

    def _camera_traj_from_batch(self, batch):
        T = None
        ego_cond = batch.get("ego_cond", None)
        if isinstance(ego_cond, dict):
            T = ego_cond.get("T_world_pv", None)
        if T is None:
            T = batch.get("T_world_cam", batch.get("T_world_exo_cam", None))
        if T is None:
            return None, None
        if T.ndim == 3:
            T = T[None]
        T = T[0].detach().float().cpu()
        pos = T[:, :3, 3]
        forward = -T[:, :3, 2]
        return pos, forward

    def _add_world_traj_figure(self, batch, outputs, tag_prefix="val"):
        if self.logger is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            Log.warning(f"[Vis] matplotlib unavailable for world trajectory figure: {e}")
            return

        def _traj_from_params(params):
            if params is None or "transl" not in params:
                return None
            t = params["transl"]
            if t.ndim == 2:
                t = t[None]
            return t[0].detach().float().cpu()

        def _filter_finite_traj(traj):
            if traj is None or traj.numel() == 0:
                return None
            if traj.shape[-1] != 3:
                return None
            traj = traj.reshape(-1, 3)
            finite = torch.isfinite(traj).all(dim=-1)
            if not finite.any():
                return None
            return traj[finite]

        world_coord = self._batch_world_coord_name(batch)
        curves = []
        cam_traj, cam_forward = self._camera_traj_from_batch(batch)
        cam_traj = self._to_gvhmr_vis_world(cam_traj, world_coord) if cam_traj is not None else None
        cam_forward = self._to_gvhmr_vis_world(cam_forward, world_coord) if cam_forward is not None else None
        if cam_traj is not None and cam_traj.numel() > 0:
            cam_traj = cam_traj.reshape(-1, 3)
            cam_finite = torch.isfinite(cam_traj).all(dim=-1)
            if cam_forward is not None and cam_forward.shape[0] == cam_traj.shape[0]:
                cam_forward = cam_forward.reshape(-1, 3)
                cam_finite = cam_finite & torch.isfinite(cam_forward).all(dim=-1)
                cam_forward = cam_forward[cam_finite] if cam_finite.any() else None
            else:
                cam_forward = None
            cam_traj = cam_traj[cam_finite] if cam_finite.any() else None
        exo_gt = _filter_finite_traj(self._to_gvhmr_vis_world(_traj_from_params(outputs.get("exo_gt_smpl_params_w_aligned", batch.get("exo", {}).get("smpl_params_w", batch.get("interactee_smpl_params_w", batch.get("smpl_params_w"))))), world_coord))
        ego_gt = _filter_finite_traj(self._to_gvhmr_vis_world(_traj_from_params(outputs.get("ego_gt_smpl_params_w_aligned", batch.get("ego", {}).get("smpl_params_w"))), world_coord))
        ego_pred = _filter_finite_traj(self._to_gvhmr_vis_world(_traj_from_params(outputs.get("pred_smpl_params_global_ego")), world_coord))
        exo_pred_incam = _filter_finite_traj(self._to_gvhmr_vis_world(_traj_from_params(outputs.get("pred_smpl_params_kinect_from_incam")), world_coord))
        frozen = _filter_finite_traj(self._to_gvhmr_vis_world(_traj_from_params(outputs.get("frozen_ego_image_exo_world_from_pv", outputs.get("frozen_ego_image_exo_kinect_from_incam"))), world_coord))
        input_role = batch.get("_input_role", self._effective_branch_mode(batch))
        exo_pred_label = "exo_pred"
        for name, traj, color in [
            ("exo_gt", exo_gt, "tab:blue"),
            ("ego_gt", ego_gt, "tab:green"),
            (exo_pred_label, exo_pred_incam, "tab:purple"),
            ("ego_pred", ego_pred, "tab:orange"),
            ("exo_pred_aux", frozen, "tab:red"),
        ]:
            if traj is not None and traj.numel() > 0:
                curves.append((name, traj, color))
        if not curves:
            return

        fig = plt.figure(figsize=(6, 5), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        extent_traj = [traj for _, traj, _ in curves]
        if cam_traj is not None and cam_traj.numel() > 0:
            extent_traj.append(cam_traj)
        all_traj = torch.cat(extent_traj, dim=0)
        all_traj = all_traj[torch.isfinite(all_traj).all(dim=-1)]
        if all_traj.numel() == 0:
            plt.close(fig)
            return
        center = all_traj.mean(0)
        radius = max(float((all_traj - center).abs().max()), 1.0)
        if not np.isfinite(radius):
            plt.close(fig)
            return
        for name, traj, color in curves:
            # Match world_mesh_floor view: horizontal plane is x-z, vertical axis is y.
            ax.plot(traj[:, 0], traj[:, 2], traj[:, 1], color=color, label=name, linewidth=2)
            ax.scatter(traj[:1, 0], traj[:1, 2], traj[:1, 1], color=color, marker="o", s=20)
            ax.scatter(traj[-1:, 0], traj[-1:, 2], traj[-1:, 1], color=color, marker="x", s=25)
        if cam_traj is not None and cam_traj.numel() > 0:
            cam_color = "black"
            ax.plot(cam_traj[:, 0], cam_traj[:, 2], cam_traj[:, 1], color=cam_color, label="input_cam", linewidth=1.5, linestyle="--")
            ax.scatter(cam_traj[:1, 0], cam_traj[:1, 2], cam_traj[:1, 1], color=cam_color, marker="^", s=35)
            stride = max(cam_traj.shape[0] // 8, 1)
            if cam_forward is not None and cam_forward.shape[0] == cam_traj.shape[0]:
                p = cam_traj[::stride]
                f = cam_forward[::stride]
                f_norm = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                q = p + f_norm * 0.25
                for pi, qi in zip(p, q):
                    ax.plot([pi[0], qi[0]], [pi[2], qi[2]], [pi[1], qi[1]], color=cam_color, linewidth=1.0, alpha=0.8)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(max(0.0, float(all_traj[:, 1].min()) - 0.2), max(float(all_traj[:, 1].max()) + 0.2, 2.0))
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.view_init(elev=18, azim=-55)
        ax.legend(loc="best", fontsize=7)
        if cam_traj is not None and cam_traj.numel() > 0:
            cam_y_min = float(cam_traj[:, 1].min()) - 0.2
            cam_y_max = float(cam_traj[:, 1].max()) + 0.2
            if np.isfinite(cam_y_min) and np.isfinite(cam_y_max):
                ax.set_zlim(min(ax.get_zlim()[0], cam_y_min), max(ax.get_zlim()[1], cam_y_max))
        ax.set_title("World root trajectories + input camera (x-z ground, y up)")
        self._add_tb_figure(f"{tag_prefix}/world_root_trajectories", fig, tag_prefix)
        plt.close(fig)

    def _visualize_validation(self, batch, outputs, include_reprojection=False):
        """TensorBoard validation/test visuals without video rendering."""
        input_role = batch.get("_input_role", self._effective_branch_mode(batch))
        self._visualize_ego_pv_input(batch, tag_prefix="val", vis_frames=4)
        if include_reprojection:
            self._visualize_ego_pv_reprojection(batch, outputs, tag_prefix="val", vis_frames=4)
        self._add_world_traj_figure(batch, outputs, tag_prefix="val")
        self._add_world_mesh_floor_image(batch, outputs, tag_prefix="val", vis_frames=4)
        if ("exo" not in batch and "ego" not in batch) or input_role == "exo":
            self._visualize_model_output(batch, outputs, tag_prefix="val")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Options & Check
        do_postproc = self.trainer.state.stage == "test"  # Only apply postproc in test
        do_flip_test = "flip_test" in batch
        do_postproc_not_flip_test = do_postproc and not do_flip_test  # later pp when flip_test
        assert batch["B"] == 1, "Only support batch size 1 in evalution."

        is_paired = "exo" in batch and "ego" in batch
        branch_mode = self._effective_branch_mode(batch)
        active_exo = branch_mode in ("exo", "both")
        val_input_role = "ego" if branch_mode == "ego" else "exo"
        if branch_mode == "both":
            cfg_input_role = self.pipeline.args.get("val_input_role", self.pipeline.args.get("input_role", None))
            if cfg_input_role in ("ego", "exo"):
                val_input_role = cfg_input_role
        val_supervise_role = self.pipeline.args.get("supervise_role", None)
        if val_supervise_role not in ("ego", "exo", "both"):
            val_supervise_role = val_input_role if branch_mode == "both" else branch_mode
        if is_paired:
            batch.setdefault("K_exo", batch["K_fullimg"])
            batch["_input_role"] = val_input_role
            batch["_supervise_role"] = val_supervise_role
            if val_input_role == "ego":
                primary = batch["ego"]
                batch["smpl_params_c"] = primary["smpl_params_c"]
                batch["smpl_params_w"] = primary["smpl_params_w"]
                batch["interactee_smpl_params_c"] = primary["smpl_params_c"]
                batch["interactee_smpl_params_w"] = primary["smpl_params_w"]
                batch["bbx_xys"] = primary.get("bbx_body_xys", primary["bbx_xys"])
                batch["kp2d"] = primary.get("kp2d_body", primary["kp2d"])
                batch["f_imgseq"] = primary.get("f_body_imgseq", primary["f_imgseq"])
                batch["K_fullimg"] = batch.get("K_ego", batch["K_fullimg"])
                if "ego_imgname" in batch:
                    batch["imgname"] = batch["ego_imgname"]
                batch["mask"]["valid"] = batch["mask"].get("ego_valid", batch["mask"]["valid"])
            else:
                primary = batch["exo"]
                batch["smpl_params_c"] = primary["smpl_params_c"]
                batch["smpl_params_w"] = primary["smpl_params_w"]
                batch["interactee_smpl_params_c"] = primary["smpl_params_c"]
                batch["interactee_smpl_params_w"] = primary["smpl_params_w"]
                batch["bbx_xys"] = primary["bbx_xys"]
                batch["kp2d"] = primary["kp2d"]
                batch["f_imgseq"] = primary["f_imgseq"]
                if "exo_imgname" in batch:
                    batch["imgname"] = batch["exo_imgname"]
                batch["mask"]["valid"] = batch["mask"].get("exo_valid", batch["mask"]["valid"])

        # ROPE inference
        obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
        if "mask" in batch:
            obs[0, ~batch["mask"]["valid"][0]] = 0

        batch_ = {
            "length": batch["length"],
            "obs": obs,
            "bbx_xys": batch["bbx_xys"],
            "K_fullimg": batch["K_fullimg"],
            "K_exo": batch.get("K_exo", batch["K_fullimg"]),
            "K_ego": batch.get("K_ego", batch["K_fullimg"]),
            "cam_angvel": batch["cam_angvel"],
            "R_c2gv": batch["R_c2gv"],
            "f_imgseq": batch["f_imgseq"],
            "mask": batch.get("mask", {}),
        }
        for key in ("T_world_cam", "T_world_exo_cam"):
            if key in batch:
                batch_[key] = batch[key]
        if is_paired:
            batch_["_input_role"] = val_input_role
            batch_["_supervise_role"] = val_supervise_role
        if is_paired:
            batch_["exo"] = batch["exo"]
            batch_["ego"] = batch["ego"]
            batch_["ego_cond"] = batch.get("ego_cond", {})

        # The train loss path expects GT geometry cached by training_step.
        # Validation constructs a lighter inference batch, so recreate those
        # derived tensors before logging val losses.
        with torch.no_grad():
            gt_verts437, gt_j3d = self.smplx(**_body_smpl_params(batch["interactee_smpl_params_c"]))
            root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
            batch_["gt_j3d"] = gt_j3d
            batch_["gt_cr_coco17"] = gt_j3d - root_
            batch_["gt_c_verts437"] = gt_verts437
            batch_["gt_cr_verts437"] = gt_verts437 - root_

        # Validation losses are logged separately from train losses. We run the
        # training loss path under no-grad, then run the eval path for rendering
        # outputs. This keeps TensorBoard tags clean: val/loss vs train/loss.
        val_loss_outputs = self.pipeline.forward(batch_, train=True)
        val_log_kwargs = {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": False,
            "logger": True,
            "batch_size": int(batch["B"]),
            "sync_dist": True,
        }
        self._log_loss_outputs("val", val_loss_outputs, val_log_kwargs, default_role=val_supervise_role)
        if is_paired and branch_mode == "both":
            if val_input_role != "ego":
                ego_primary = batch["ego"]
                ego_bbx = ego_primary.get("bbx_body_xys", ego_primary["bbx_xys"])
                ego_kp2d = ego_primary.get("kp2d_body", ego_primary["kp2d"])
                ego_obs = normalize_kp2d(ego_kp2d, ego_bbx)
                ego_mask = dict(batch.get("mask", {}))
                ego_mask["valid"] = ego_mask.get("ego_valid", ego_mask.get("valid"))
                ego_obs[0, ~ego_mask["valid"][0]] = 0
                batch_ego = {
                    "length": batch["length"],
                    "obs": ego_obs,
                    "bbx_xys": ego_bbx,
                    "K_fullimg": batch.get("K_ego", batch["K_fullimg"]),
                    "cam_angvel": batch["cam_angvel"],
                    "R_c2gv": batch["R_c2gv"],
                    "f_imgseq": ego_primary.get("f_body_imgseq", ego_primary["f_imgseq"]),
                    "mask": ego_mask,
                    "_input_role": "ego",
                    "_supervise_role": "ego",
                    "exo": batch["exo"],
                    "ego": batch["ego"],
                    "ego_cond": batch.get("ego_cond", {}),
                }
                for key in ("T_world_cam", "T_world_exo_cam"):
                    if key in batch:
                        batch_ego[key] = batch[key]
                with torch.no_grad():
                    ego_verts437, ego_j3d = self.smplx(**_body_smpl_params(ego_primary["smpl_params_c"]))
                    ego_root = ego_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
                    batch_ego["gt_j3d"] = ego_j3d
                    batch_ego["gt_cr_coco17"] = ego_j3d - ego_root
                    batch_ego["gt_c_verts437"] = ego_verts437
                    batch_ego["gt_cr_verts437"] = ego_verts437 - ego_root
                    val_ego_outputs = self.pipeline.forward(batch_ego, train=True)
                self._log_loss_outputs("val", val_ego_outputs, val_log_kwargs, default_role="ego")

        outputs = self.pipeline.forward(batch_, train=False, postproc=do_postproc_not_flip_test)
        if "pred_smpl_params_global" in outputs:
            outputs["pred_smpl_params_global"] = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}
        if "pred_smpl_params_incam" in outputs:
            outputs["pred_smpl_params_incam"] = {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()}
        if "pred_smpl_params_kinect_from_incam" in outputs:
            outputs["pred_smpl_params_kinect_from_incam"] = {k: v[0] for k, v in outputs["pred_smpl_params_kinect_from_incam"].items()}
        
        # 处理 ego 输出（如果存在）
        if "pred_smpl_params_global_ego" in outputs:
            outputs["pred_smpl_params_global_ego"] = {k: v[0] for k, v in outputs["pred_smpl_params_global_ego"].items()}
        if "pred_smpl_params_global_ego_vel" in outputs:
            outputs["pred_smpl_params_global_ego_vel"] = {k: v[0] for k, v in outputs["pred_smpl_params_global_ego_vel"].items()}
        if "ego_gt_smpl_params_w_aligned" in outputs:
            outputs["ego_gt_smpl_params_w_aligned"] = {k: v[0] for k, v in outputs["ego_gt_smpl_params_w_aligned"].items()}
        if "exo_gt_smpl_params_w_aligned" in outputs:
            outputs["exo_gt_smpl_params_w_aligned"] = {k: v[0] for k, v in outputs["exo_gt_smpl_params_w_aligned"].items()}
        if "pred_smpl_params_incam_ego" in outputs:
            outputs["pred_smpl_params_incam_ego"] = {k: v[0] for k, v in outputs["pred_smpl_params_incam_ego"].items()}
        if "frozen_ego_image_exo_global" in outputs:
            outputs["frozen_ego_image_exo_global"] = {k: v[0] for k, v in outputs["frozen_ego_image_exo_global"].items()}
        if "frozen_ego_image_exo_kinect_from_incam" in outputs:
            outputs["frozen_ego_image_exo_kinect_from_incam"] = {k: v[0] for k, v in outputs["frozen_ego_image_exo_kinect_from_incam"].items()}
        if "frozen_ego_image_exo_world_from_pv" in outputs:
            outputs["frozen_ego_image_exo_world_from_pv"] = {k: v[0] for k, v in outputs["frozen_ego_image_exo_world_from_pv"].items()}
        if "frozen_ego_image_exo_incam" in outputs:
            outputs["frozen_ego_image_exo_incam"] = {k: v[0] for k, v in outputs["frozen_ego_image_exo_incam"].items()}

        if do_flip_test and active_exo:
            flip_test = batch["flip_test"]
            obs = normalize_kp2d(flip_test["kp2d"], flip_test["bbx_xys"])
            if "mask" in batch:
                obs[0, ~batch["mask"]["valid"][0]] = 0

            batch_ = {
                "length": batch["length"],
                "obs": obs,
                "bbx_xys": flip_test["bbx_xys"],
                "K_fullimg": batch["K_fullimg"],
                "cam_angvel": flip_test["cam_angvel"],
                "f_imgseq": flip_test["f_imgseq"],
            }
            flipped_outputs = self.pipeline.forward(batch_, train=False)

            # First update incam results
            flipped_outputs["pred_smpl_params_incam"] = {
                k: v[0] for k, v in flipped_outputs["pred_smpl_params_incam"].items()
            }
            smpl_params1 = outputs["pred_smpl_params_incam"]
            smpl_params2 = flip_smplx_params(flipped_outputs["pred_smpl_params_incam"])

            smpl_params_avg = smpl_params1.copy()
            smpl_params_avg["betas"] = (smpl_params1["betas"] + smpl_params2["betas"]) / 2
            smpl_params_avg["body_pose"] = avg_smplx_aa(smpl_params1["body_pose"], smpl_params2["body_pose"])
            smpl_params_avg["global_orient"] = avg_smplx_aa(
                smpl_params1["global_orient"], smpl_params2["global_orient"]
            )
            outputs["pred_smpl_params_incam"] = smpl_params_avg

            # Then update global results
            outputs["pred_smpl_params_global"]["betas"] = smpl_params_avg["betas"]
            outputs["pred_smpl_params_global"]["body_pose"] = smpl_params_avg["body_pose"]

            # Finally, apply postprocess
            if do_postproc:
                # temporarily recover the original batch-dim
                outputs["pred_smpl_params_global"] = {k: v[None] for k, v in outputs["pred_smpl_params_global"].items()}
                outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(outputs, self.pipeline.endecoder)
                body_pose = process_ik(outputs, self.pipeline.endecoder)
                outputs["pred_smpl_params_global"] = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}

                outputs["pred_smpl_params_global"]["body_pose"] = body_pose[0]
                # outputs["pred_smpl_params_incam"]["body_pose"] = body_pose[0]

        # ========================================================
        # Validation/Test 可视化 (TensorBoard)
        # 使用 val_vis_every_n_batches 控制频率，每个 epoch 可视化不同样本
        # ========================================================
        if self.logger is not None and batch_idx % self.val_vis_every_n_batches == 0:
            current_step = int(self.trainer.global_step)
            reproj_interval = max(int(self.vis_every_n_steps), 1)
            include_reprojection = (
                self._last_val_reproj_vis_step < 0
                or current_step - self._last_val_reproj_vis_step >= reproj_interval
            )
            self._visualize_validation(batch, outputs, include_reprojection=include_reprojection)
            if include_reprojection:
                self._last_val_reproj_vis_step = current_step
            self._val_vis_step += 1
            

        if False:  # o3d
            prog_keys = [
                "pred_smpl_progress",
                "pred_localjoints_progress",
                "pred_incam_localjoints_progress",
            ]
            for k in prog_keys:
                if k in outputs_window:
                    seq_out = torch.cat(
                        [v[:, :l] for v, l in zip(outputs_window[k], length)], dim=1
                    )  # (B, P, L, J, 3) -> (P, L, J, 3) -> (P, CL, J, 3)
                    outputs[k] = seq_out[None]

        return outputs

    def configure_optimizers(self):
        slow_prefixes = (
            "denoiser3d.learned_pos_linear",
            "denoiser3d.learned_pos_params",
            "denoiser3d.embed_noisyobs",
            "denoiser3d.cliffcam_embedder",
            "denoiser3d.cam_angvel_embedder",
            "denoiser3d.imgseq_embedder",
            "denoiser3d.blocks",
        )
        main_params = []
        slow_params = []
        for name, param in self.pipeline.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(slow_prefixes):
                slow_params.append(param)
            else:
                main_params.append(param)

        backbone_lr_scale = float(self.backbone_lr_scale)
        opt_keywords = getattr(self.optimizer, "keywords", {}) or {}
        base_lr = opt_keywords.get("lr", None)
        if slow_params and backbone_lr_scale != 1.0 and base_lr is not None:
            params = [
                {"params": main_params},
                {"params": slow_params, "lr": float(base_lr) * backbone_lr_scale},
            ]
            Log.info(
                f"[Optimizer] main params={len(main_params)}, slow backbone params={len(slow_params)}, "
                f"backbone_lr={float(base_lr) * backbone_lr_scale:g} ({backbone_lr_scale:g}x)"
            )
        else:
            params = main_params + slow_params
            if slow_params and backbone_lr_scale != 1.0:
                Log.warning("[Optimizer] backbone_lr_scale ignored because base lr is unavailable from optimizer config")
        optimizer = self.optimizer(params=params)

        if self.scheduler_cfg["scheduler"] is None:
            return optimizer

        scheduler_cfg = dict(self.scheduler_cfg)
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)
        return [optimizer], [scheduler_cfg]

    # ============== Utils ================= #
    def on_save_checkpoint(self, checkpoint) -> None:
        for ig_keys in self.ignored_weights_prefix:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    # Log.info(f"Remove key `{ig_keys}' from checkpoint.")
                    checkpoint["state_dict"].pop(k)

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt: {ckpt_path}")

        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        has_ego_pose_head = any(k.startswith("pipeline.denoiser3d.final_layer_ego.") for k in state_dict)
        own_state = self.state_dict()
        skipped_shape = []
        filtered_state = {}
        for k, v in state_dict.items():
            if k in own_state and own_state[k].shape != v.shape:
                skipped_shape.append((k, tuple(v.shape), tuple(own_state[k].shape)))
                continue
            filtered_state[k] = v
        if skipped_shape:
            Log.warning(f"Skip shape-mismatched checkpoint keys: {skipped_shape}")
        missing, unexpected = self.load_state_dict(filtered_state, strict=False)
        real_missing = []
        for k in missing:
            ignored_when_saving = any(k.startswith(ig_keys) for ig_keys in self.ignored_weights_prefix)
            if not ignored_when_saving:
                real_missing.append(k)

        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")
        if getattr(self, "copy_exo_to_ego_after_load", False):
            if has_ego_pose_head:
                Log.info("[Init] Checkpoint already contains ego pose head; skip exo-to-ego copy")
            else:
                self._copy_exo_to_ego(mode=getattr(self, "copy_exo_to_ego_mode", "pose"))


gvhmr_pl = builds(
    GvhmrPL,
    pipeline="${pipeline}",
    optimizer="${optimizer}",
    scheduler_cfg="${scheduler_cfg}",
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="gvhmr_pl", node=gvhmr_pl, group="model/gvhmr")