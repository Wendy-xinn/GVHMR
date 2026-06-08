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
from hmr4d.utils.video_io_utils import get_writer
import os

from hmr4d.utils.geo_transform import compute_T_ayfz2ay, apply_T_on_points
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from hmr4d.utils.vis.renderer_tools import checkerboard_geometry
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo.augment_noisy_pose import (
    get_wham_aug_kp3d,
    get_visible_mask,
    get_invisible_legs_mask,
    randomly_occlude_lower_half,
    randomly_modify_hands_legs,
)
from hmr4d.utils.geo.hmr_cam import perspective_projection, normalize_kp2d, safely_render_x3d_K, get_bbx_xys

from hmr4d.utils.video_io_utils import save_video
from hmr4d.utils.vis.cv2_utils import draw_bbx_xys_on_image_batch, draw_coco17_skeleton_batch
from hmr4d.utils.geo.flip_utils import flip_smplx_params, avg_smplx_aa
from hmr4d.model.gvhmr.utils.postprocess import pp_static_joint, pp_static_joint_cam, process_ik
from hmr4d.model.gvhmr.utils.vis_utils import render_global_video


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
        vis_every_n_steps=100,
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        if copy_exo_to_ego:
            self._copy_exo_to_ego()

        self._set_freeze(
            freeze_backbone=freeze_backbone,
            freeze_exo_head=freeze_exo_head,
            freeze_ego_head=freeze_ego_head,
        )
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg

        # Options
        self.ignored_weights_prefix = ignored_weights_prefix

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
    
    def _copy_exo_to_ego(self):
        den = self.pipeline.denoiser3d
        # 检查 final_layer_ego 是否存在且不为 None (需要 dual_head=True)
        if not hasattr(den, "final_layer_ego") or den.final_layer_ego is None:
            Log.warning("final_layer_ego is None, skip copy_exo_to_ego. Set dual_head=True in network config to enable.")
            return
        den.final_layer_ego.load_state_dict(den.final_layer.state_dict())
        if hasattr(den, "pred_cam_head_ego") and den.pred_cam_head_ego is not None and den.pred_cam_head:
            den.pred_cam_head_ego.load_state_dict(den.pred_cam_head.state_dict())
        if hasattr(den, "static_conf_head_ego") and den.static_conf_head_ego is not None and den.static_conf_head:
            den.static_conf_head_ego.load_state_dict(den.static_conf_head.state_dict())
    
    def _set_freeze(self, freeze_backbone=False, freeze_exo_head=False, freeze_ego_head=True):
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
            _freeze_module(den.blocks, True)

        _freeze_module(den.final_layer, freeze_exo_head)
        _freeze_module(getattr(den, "pred_cam_head", None), freeze_exo_head)
        _freeze_module(getattr(den, "static_conf_head", None), freeze_exo_head)

        _freeze_module(getattr(den, "final_layer_ego", None), freeze_ego_head)
        _freeze_module(getattr(den, "pred_cam_head_ego", None), freeze_ego_head)
        _freeze_module(getattr(den, "static_conf_head_ego", None), freeze_ego_head)

    def training_step(self, batch, batch_idx):
        B, F = batch["smpl_params_c"]["body_pose"].shape[:2]

        # Create augmented noisy-obs : gt_j3d(coco17)
        with torch.no_grad():
            gt_verts437, gt_j3d = self.smplx(**batch["interactee_smpl_params_c"])
            root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
            batch["gt_j3d"] = gt_j3d
            batch["gt_cr_coco17"] = gt_j3d - root_
            batch["gt_c_verts437"] = gt_verts437
            batch["gt_cr_verts437"] = gt_verts437 - root_

        # bbx_xys
        i_x2d = safely_render_x3d_K(gt_verts437, batch["K_fullimg"], thr=0.3)
        bbx_xys = get_bbx_xys(i_x2d, do_augment=True)
        if False:  # trust image bbx_xys seems better
            batch["bbx_xys"] = bbx_xys
        else:
            mask_bbx_xys = batch["mask"]["bbx_xys"]
            batch["bbx_xys"][~mask_bbx_xys] = bbx_xys[~mask_bbx_xys]
        if False:  # visualize bbx_xys from an iPhone view   可视化检查（如果输入没有真实的图像）
            render_w, render_h = 120, 160  # iphone main-lens 24mm 3:4
            ratio = render_w / 1528
            offset = torch.tensor([764 - 500, 1019 - 500]).to(i_x2d)
            i_x2d_render = (i_x2d + offset).clone()
            i_x2d_render = (i_x2d_render * ratio).long().clone()
            torch.clamp_(i_x2d_render[..., 0], 0, render_w - 1)
            torch.clamp_(i_x2d_render[..., 1], 0, render_h - 1)
            bbx_xys_render = bbx_xys.clone()
            bbx_xys_render[..., :2] += offset
            bbx_xys_render *= ratio

            output_dir = Path("outputs/simulated_bbx_xys")
            output_dir.mkdir(parents=True, exist_ok=True)
            video_list = []
            for bid in range(B):
                images = torch.zeros(F, render_h, render_w, 3, device=i_x2d.device)
                for fid in range(F):
                    images[fid, i_x2d_render[bid, fid, :, 1], i_x2d_render[bid, fid, :, 0]] = 255

                images = draw_bbx_xys_on_image_batch(bbx_xys_render[bid].cpu().numpy(), images.cpu().numpy())
                images = np.stack(images).astype("uint8")  # (L, H, W, 3)
                images[:, 0, :] = np.array([255, 255, 255])
                images[:, -1, :] = np.array([255, 255, 255])
                images[:, :, 0] = np.array([255, 255, 255])
                images[:, :, -1] = np.array([255, 255, 255])
                video_list.append(images)

            # stack videos
            video_output = []
            for i in range(0, len(video_list), 4):
                if i + 4 <= len(video_list):
                    video_output.append(np.concatenate(video_list[i : i + 4], axis=2))
            video_output = np.concatenate(video_output, axis=1)
            save_video(video_output, output_dir / f"{batch_idx}.mp4", fps=30, quality=5)

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
            mask_real_vitpose = (torch.rand(B).to(obs_kp2d) < prob) * batch["mask"]["vitpose"]
            batch["obs"][mask_real_vitpose] = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])[mask_real_vitpose]

        # Set untrusted frames to False
        batch["obs"][~batch["mask"]["valid"]] = 0

        if False:  # wis3d  用于调试的可视化工具
            wis3d = make_wis3d(name="debug-aug-kp3d")
            add_motion_as_lines(gt_j3d[0], wis3d, name="gt_j3d", skeleton_type="coco17")
            add_motion_as_lines(noisy_j3d[0], wis3d, name="noisy_j3d", skeleton_type="coco17")

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
        current_step = self.trainer.global_step
        if self.logger is not None and current_step % self.vis_every_n_steps == 0:
            Log.info(f"[Vis] Triggering visualization at step {current_step}")
            try:
                # self._visualize_training(batch, outputs)
                self._visualize_model_output(batch, outputs)
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
        self.log("train/loss", outputs["loss"], **log_kwargs)
        for k, v in outputs.items():
            if "_loss" in k:
                self.log(f"train/{k}", v, **log_kwargs)

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

    def _visualize_training(self, batch, outputs):
        """将训练过程中的图像、bbox、关键点、mesh渲染等可视化到 TensorBoard"""
        # 3d可视化
        # 3d可视化 - 检查 NaN 后再可视化
        gt_j3d = batch["gt_j3d"][0][:20]
        noisy_j3d = batch["noisy_j3d"][0][:20]
        
        if not torch.isnan(gt_j3d).any():
            wis3d = make_wis3d(name="train_stage1_gt", time_postfix=True)
            add_motion_as_lines(gt_j3d, wis3d, name="gt_j3d", skeleton_type="coco17")
            del wis3d
        
        if not torch.isnan(noisy_j3d).any():
            wis3d = make_wis3d(name="train_stage1_noisy", time_postfix=True)
            add_motion_as_lines(noisy_j3d, wis3d, name="noisy_j3d", skeleton_type="coco17")
            del wis3d

        # 2. 可视化模型输出: incam (局部) 和 global (整体)
        self._visualize_model_output(batch, outputs)

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
        
        exo_incam_gt_key = "interactee_smpl_params_c" if "interactee_smpl_params_c" in batch else "smpl_params_c"
        if exo_incam_gt_key not in batch:
            return
        
        with torch.no_grad():
            # 预测
            smpl_params_reshaped = {k: v.reshape(B * F, -1) for k, v in pred_smpl_params_incam.items()}
            smplx_out_pred = self.smplx_full(**smpl_params_reshaped)
            pred_verts = smplx_out_pred.vertices.reshape(B, F, -1, 3)
            # 关键点使用 437 模型（只有 17 个 COCO 关节点，用于 draw_coco17_skeleton_batch）
            _, pred_j3d = self.smplx(**smpl_params_reshaped)
            pred_j3d = pred_j3d.reshape(B, F, -1, 3)
            
            # GT
            gt_params_reshaped = {k: v.reshape(B * F, -1) for k, v in batch[exo_incam_gt_key].items()}
            gt_smplx_out = self.smplx_full(**gt_params_reshaped)
            gt_verts = gt_smplx_out.vertices.reshape(B, F, -1, 3)
            _, gt_j3d = self.smplx(**batch[exo_incam_gt_key])
            gt_j3d = gt_j3d.reshape(B, F, -1, 3)
            
            # 渲染 incam overlay
            self._render_incam_overlay(
                batch, pred_verts, pred_j3d, gt_verts, gt_j3d, 
                tag_prefix=tag_prefix
            )

    def _render_incam_overlay(self, batch, pred_verts, pred_joints, gt_verts, gt_j3d, tag_prefix="train"):
        """
            可视化:
                1. pred 
                2. gt 
                3. pred/gt mesh compare
            
            Args:
                tag_prefix: TensorBoard 日志路径前缀，训练阶段用 "train"，验证阶段用 "val"
        """
        from hmr4d.utils.vis.renderer import Renderer
        
        imgname_list = batch.get("imgname", None)
        if imgname_list is None or len(imgname_list) == 0:
            Log.warning("[Vis] imgname_list is empty")
            return

        # 读取图像并获取相机内参
        if isinstance(imgname_list[0], list):
            img_path = self._resolve_image_path(imgname_list[0][0])
        else:
            img_path = self._resolve_image_path(imgname_list[0])
        img = cv2.imread(img_path)
        if img is None:
            Log.warning(f"[Vis] Failed to read image: {img_path}")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        K = batch["K_fullimg"][0, 0] if batch["K_fullimg"].ndim == 4 else batch["K_fullimg"][0]

        # 取预测的顶点和关节点 (batch 0, frame 0)
        # 转换为 float32，因为渲染器期望 float32 类型
        pred_verts_0 = pred_verts[0, 0].detach().cpu().float()
        pred_joints_0 = pred_joints[0, 0].detach().cpu().float()

        # 创建渲染器
        renderer = Renderer(W, H, device="cpu", faces=self.smplx_full.faces, K=K)
        bbx = batch["bbx_xys"][0]
        if bbx.ndim > 1:
            bbx = bbx[0]
        bbx = bbx.cpu().numpy()

        # 渲染预测 Mesh (蓝色)
        pred_img = renderer.render_mesh(pred_verts_0, background=img_rgb.copy(), colors=[0.3, 0.5, 1.0])
        # 绘制预测关键点 (使用 draw_coco17_skeleton_batch)
        # 使用原模型的 perspective_projection 函数
        # 确保 K 在 CPU 上，并将 joints 转为 (1, 1, J, 3) 形状
        K_cpu = K.cpu() if isinstance(K, torch.Tensor) else K
        pred_joints_tensor = pred_joints_0 if isinstance(pred_joints_0, torch.Tensor) else torch.from_numpy(pred_joints_0)
        pred_j2d = perspective_projection(pred_joints_tensor.unsqueeze(0).unsqueeze(0), K_cpu.unsqueeze(0).unsqueeze(0))
        pred_j2d = pred_j2d.squeeze(0).squeeze(0).numpy()
        kp2d_pred = np.concatenate([pred_j2d, np.ones((pred_j2d.shape[0], 1))], axis=-1)
        pred_overlay = draw_coco17_skeleton_batch([pred_img], [kp2d_pred])[0]
        pred_overlay = draw_bbx_xys_on_image_batch([bbx], [pred_overlay])[0]

        # 渲染 GT (红色，如果存在)
        if gt_verts is not None:
            gt_verts_0 = gt_verts[0, 0].detach().cpu().float()
            gt_joints_0 = gt_j3d[0, 0].detach().cpu().float()
            gt_img = renderer.render_mesh(gt_verts_0, background=img_rgb.copy(), colors=[1.0, 0.3, 0.3])
            gt_joints_tensor = gt_joints_0 if isinstance(gt_joints_0, torch.Tensor) else torch.from_numpy(gt_joints_0)
            gt_j2d = perspective_projection(gt_joints_tensor.unsqueeze(0).unsqueeze(0), K_cpu.unsqueeze(0).unsqueeze(0))
            gt_j2d = gt_j2d.squeeze(0).squeeze(0).numpy()
            kp2d_gt = np.concatenate([gt_j2d, np.ones((gt_j2d.shape[0], 1))], axis=-1)
            gt_overlay = draw_coco17_skeleton_batch([gt_img], [kp2d_gt])[0]
            gt_overlay = draw_bbx_xys_on_image_batch([bbx], [gt_overlay])[0]
            
            pred_gt = None
            bg = np.ones((H, W, 3), dtype=np.uint8) * 255
            # pred mesh
            pred_mesh_img = renderer.render_mesh(
                pred_verts_0,
                background=bg.copy(),
                colors=[0.2, 0.4, 1.0],
            )
            # gt mesh
            gt_mesh_img = renderer.render_mesh(
                gt_verts_0,
                background=pred_mesh_img,
                colors=[1.0, 0.2, 0.2],
            )
            pred_gt = gt_mesh_img
            # pred_gt = cv2.addWeighted(
            #     pred_mesh_img,
            #     0.5,
            #     gt_mesh_img,
            #     0.5,
            #     0,
            # )
    
        pred_tb = torch.from_numpy(pred_overlay).permute(2, 0, 1)
        self.logger.experiment.add_image(f"{tag_prefix}/incam_pred", pred_tb, self.global_step)
        if gt_overlay is not None:
            gt_tb = torch.from_numpy(gt_overlay).permute(2, 0, 1)
            self.logger.experiment.add_image(f"{tag_prefix}/incam_gt", gt_tb, self.global_step)
        if pred_gt is not None:
            pred_gt_tb = torch.from_numpy(pred_gt).permute(2, 0, 1)
            self.logger.experiment.add_image(f"{tag_prefix}/incam_pred_gt_overlay", pred_gt_tb, self.global_step)

    def _visualize_validation(self, batch, outputs):
        """可视化 validation/test 阶段: ego/exo global 视频 + exo incam overlay"""
        B, F = batch["smpl_params_c"]["body_pose"].shape[:2]
        output_dir = self.trainer.default_root_dir
        global_step = self.trainer.global_step
        
        # 1. Ego global 视频
        pred_smpl_params_global_ego = outputs.get("pred_smpl_params_global_ego", None)
        if pred_smpl_params_global_ego is not None and "smpl_params_w" in batch:
            with torch.no_grad():
                pred_reshaped = {k: v.reshape(B * F, -1) for k, v in pred_smpl_params_global_ego.items()}
                gt_reshaped = {k: v.reshape(B * F, -1) for k, v in batch["smpl_params_w"].items()}
                smplx_out_pred = self.smplx_full(**pred_reshaped)
                smplx_out_gt = self.smplx_full(**gt_reshaped)
                # Reshape 回 (B, F, V, 3)
                for out in [smplx_out_pred, smplx_out_gt]:
                    out.vertices = out.vertices.reshape(B, F, -1, 3)
                    out.joints = out.joints.reshape(B, F, -1, 3)
                render_global_video(smplx_out_pred, smplx_out_gt, global_step, output_dir, tag_prefix="ego", vis_frames=30)
        
        # 2. Exo global 视频
        pred_smpl_params_global = outputs.get("pred_smpl_params_global", None)
        if pred_smpl_params_global is not None and "interactee_smpl_params_w" in batch:
            with torch.no_grad():
                pred_reshaped = {k: v.reshape(B * F, -1) for k, v in pred_smpl_params_global.items()}
                gt_reshaped = {k: v.reshape(B * F, -1) for k, v in batch["interactee_smpl_params_w"].items()}
                smplx_out_pred = self.smplx_full(**pred_reshaped)
                smplx_out_gt = self.smplx_full(**gt_reshaped)
                for out in [smplx_out_pred, smplx_out_gt]:
                    out.vertices = out.vertices.reshape(B, F, -1, 3)
                    out.joints = out.joints.reshape(B, F, -1, 3)
                render_global_video(smplx_out_pred, smplx_out_gt, global_step, output_dir, tag_prefix="exo", vis_frames=30)
        
        # 3. Exo incam overlay
        self._visualize_model_output(batch, outputs, tag_prefix="val")

    # 利用wis3d进行可视化，可以用来debug  
    def _visualize_val_global(self, batch, smplx_out_pred, smplx_out_gt, smplx_model):
        """可视化 validation/test 阶段的 global 预测结果"""
        wis3d = make_wis3d(name="val_global_motion", time_postfix=True)
        
        # gender = batch["gender"][0]
        # T_w2ay = batch["T_w2ay"][0]
        device = smplx_out_gt.vertices.device
        B = batch["smpl_params_c"]["body_pose"].shape[0]
        
        # EgoBody 世界坐标系：Y 轴向上（gravity_vec = [0, -1, 0]）
        # GVHMR ay 坐标系：Y 轴向上
        # 两者坐标系一致，T_w2ay 为单位矩阵
        T_w2ay = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        
        # 仅可视化第一个batch
        vis_frames = min(smplx_out_pred.vertices.shape[1], 30)
        # for i in range(len(smplx_out_pred.vertices)):
        # for i in range(vis_frames):
        #     wis3d.set_scene_id(i)
        #     wis3d.add_mesh(smplx_out_pred.vertices[0, i], smplx_model.bm.faces, name=f"pred-smplx-global_{i}")
    
        # # GT (w)
        # smplx_models = {
        #     "male": make_smplx("rich-smplx", gender="male").cuda(),
        #     "female": make_smplx("rich-smplx", gender="female").cuda(),
        # }
        # gt_smpl_params = {k: v[0, windows[0]] for k, v in batch["gt_smpl_params"].items()}
        # gt_smplx_out = smplx_models[gender](**gt_smpl_params)

        # GT (ayfz)：ay是将数据集的重力都统一为y=重力；ayfz是将数据集的重力统一为y=重力，并且人体面朝z方向，这样可以更好地观察人体的运动细节，而不受全局旋转的干扰？
        smplx_verts_ay = apply_T_on_points(smplx_out_gt.vertices, T_w2ay)
        smplx_joints_ay = apply_T_on_points(smplx_out_gt.joints, T_w2ay)
        # 取第一帧计算 ayfz 变换 (B, J, 3)
        T_ay2ayfz = compute_T_ayfz2ay(smplx_joints_ay[:, 0], inverse=True)  # (B, 4, 4)
        smplx_verts_ayfz = apply_T_on_points(smplx_verts_ay, T_ay2ayfz)  # (B, F, V, 3)
        
        # Pred 也在 ay 坐标系，需要同样应用 T_ay2ayfz 转换到 ayfz
        T_ay2ayfz_pred = compute_T_ayfz2ay(smplx_out_pred.joints[:, 0], inverse=True)  # (B, 4, 4)
        smplx_pred_verts_ayfz = apply_T_on_points(smplx_out_pred.vertices, T_ay2ayfz_pred)  # (B, F, V, 3)

        for i in range(vis_frames):
            wis3d.set_scene_id(i)
            ground_v, ground_f, ground_vc, _ = checkerboard_geometry(
                length=10,
                c1=0,
                c2=0,
                up="y",
            )
            wis3d.add_mesh(ground_v, ground_f, ground_vc, name="ground")
            wis3d.add_mesh(smplx_pred_verts_ayfz[0, i], smplx_model.bm.faces, name=f"pred-smplx-ayfz_{i}")
            wis3d.add_mesh(smplx_verts_ayfz[0, i], smplx_model.bm.faces, name=f"gt-smplx-ayfz_{i}")
        
        del wis3d

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Options & Check
        do_postproc = self.trainer.state.stage == "test"  # Only apply postproc in test
        do_flip_test = "flip_test" in batch
        do_postproc_not_flip_test = do_postproc and not do_flip_test  # later pp when flip_test
        assert batch["B"] == 1, "Only support batch size 1 in evalution."

        # ROPE inference
        obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
        if "mask" in batch:
            obs[0, ~batch["mask"]["valid"][0]] = 0

        batch_ = {
            "length": batch["length"],
            "obs": obs,
            "bbx_xys": batch["bbx_xys"],
            "K_fullimg": batch["K_fullimg"],
            "cam_angvel": batch["cam_angvel"],
            "f_imgseq": batch["f_imgseq"],
        }
        outputs = self.pipeline.forward(batch_, train=False, postproc=do_postproc_not_flip_test)
        outputs["pred_smpl_params_global"] = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}
        outputs["pred_smpl_params_incam"] = {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()}
        
        # 处理 ego 输出（如果存在）
        if "pred_smpl_params_global_ego" in outputs:
            outputs["pred_smpl_params_global_ego"] = {k: v[0] for k, v in outputs["pred_smpl_params_global_ego"].items()}
        if "pred_smpl_params_incam_ego" in outputs:
            outputs["pred_smpl_params_incam_ego"] = {k: v[0] for k, v in outputs["pred_smpl_params_incam_ego"].items()}

        if do_flip_test:
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
        # ========================================================
        if self.logger is not None and batch_idx == 0:
            self._visualize_validation(batch, outputs)
            

        if False:  # wis3d
            wis3d = make_wis3d(name="debug-rich-cap")
            smplx_model = make_smplx("rich-smplx", gender="neutral").cuda()
            gender = batch["gender"][0]
            T_w2ay = batch["T_w2ay"][0]

            # Prediction
            # add_motion_as_lines(outputs_window["pred_ayfz_motion"][bid], wis3d, name="pred_ayfz_motion")

            smplx_out = smplx_model(**pred_smpl_params_global)
            for i in range(len(smplx_out.vertices)):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(smplx_out.vertices[i], smplx_model.bm.faces, name=f"pred-smplx-global")

            # GT (w)
            smplx_models = {
                "male": make_smplx("rich-smplx", gender="male").cuda(),
                "female": make_smplx("rich-smplx", gender="female").cuda(),
            }
            gt_smpl_params = {k: v[0, windows[0]] for k, v in batch["gt_smpl_params"].items()}
            gt_smplx_out = smplx_models[gender](**gt_smpl_params)

            # GT (ayfz)：ay是将数据集的重力都统一为y=重力；ayfz是将数据集的重力统一为y=重力，并且人体面朝z方向，这样可以更好地观察人体的运动细节，而不受全局旋转的干扰？
            smplx_verts_ay = apply_T_on_points(gt_smplx_out.vertices, T_w2ay)
            smplx_joints_ay = apply_T_on_points(gt_smplx_out.joints, T_w2ay)
            T_ay2ayfz = compute_T_ayfz2ay(smplx_joints_ay[:1], inverse=True)[0]  # (4, 4)
            smplx_verts_ayfz = apply_T_on_points(smplx_verts_ay, T_ay2ayfz)  # (F, 22, 3)

            for i in range(len(smplx_verts_ayfz)):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(smplx_verts_ayfz[i], smplx_models[gender].bm.faces, name=f"gt-smplx-ayfz")

            breakpoint()

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
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                params.append(v)
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
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            ignored_when_saving = any(k.startswith(ig_keys) for ig_keys in self.ignored_weights_prefix)
            if not ignored_when_saving:
                real_missing.append(k)

        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")


gvhmr_pl = builds(
    GvhmrPL,
    pipeline="${pipeline}",
    optimizer="${optimizer}",
    scheduler_cfg="${scheduler_cfg}",
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="gvhmr_pl", node=gvhmr_pl, group="model/gvhmr")
