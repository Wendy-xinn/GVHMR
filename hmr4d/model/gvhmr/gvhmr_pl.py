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
        val_vis_every_n_steps=200,  # 验证集可视化间隔（每隔 N 个 step 可视化一次）
        test_vis_every_n_batches=1,  # 测试阶段每隔 N 个 batch 可视化一次
        test_vis_max_batches=2,  # 测试阶段最多可视化多少个 batch；0 表示关闭
        zero_f_imgseq=False,  # 实验：将 f_imgseq 置零
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
        self.zero_f_imgseq = zero_f_imgseq

        # Test/predict reuse validation logic; explicit hooks are defined below.

        # SMPLX (lite 版本用于训练)
        self.smplx = make_smplx("supermotion_v437coco17")
        
        # SMPLX (完整版本用于可视化，与 demo.py 保持一致)
        self.smplx_full = make_smplx("supermotion")

        # SMPLX to SMPL 转换矩阵和 J_regressor（与 demo.py 保持一致）
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
        
        # 将转换矩阵注册为 buffer，但不移动到设备（在可视化时动态移动到对应设备）

        self.vis_every_n_steps = vis_every_n_steps
        self.val_vis_every_n_steps = val_vis_every_n_steps
        self.test_vis_every_n_batches = test_vis_every_n_batches
        self.test_vis_max_batches = test_vis_max_batches
    
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

        # 实验：将 f_imgseq 置零，验证 ego 头是否可以仅从 obs/cliffcam/cam_angvel 学习
        if self.zero_f_imgseq:
            batch["f_imgseq"] = torch.zeros_like(batch["f_imgseq"])

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
        self._visualize_model_output(batch, outputs, tag_prefix="train")

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
        
        # 收集所有帧的图像
        pred_overlays = []
        gt_overlays = []
        pred_gt_overlays = []
        
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

            # 渲染预测 Mesh (蓝色)
            pred_img = renderer.render_mesh(pred_verts_f, background=img_rgb.copy(), colors=[0.3, 0.5, 1.0])
            # 绘制预测关键点
            pred_joints_tensor = pred_joints_f if isinstance(pred_joints_f, torch.Tensor) else torch.from_numpy(pred_joints_f)
            pred_j2d = perspective_projection(pred_joints_tensor.unsqueeze(0).unsqueeze(0), K_cpu.unsqueeze(0).unsqueeze(0))
            pred_j2d = pred_j2d.squeeze(0).squeeze(0).numpy()
            kp2d_pred = np.concatenate([pred_j2d, np.ones((pred_j2d.shape[0], 1))], axis=-1)
            pred_overlay = draw_coco17_skeleton_batch([pred_img], [kp2d_pred])[0]
            pred_overlay = draw_bbx_xys_on_image_batch([bbx], [pred_overlay])[0]
            pred_overlays.append(pred_overlay)
            
            # 渲染 GT (红色，如果存在)
            if gt_verts is not None:
                gt_verts_f = gt_verts[0, frame_idx].detach().cpu().float()
                gt_joints_f = gt_j3d[0, frame_idx].detach().cpu().float()
                gt_img = renderer.render_mesh(gt_verts_f, background=img_rgb.copy(), colors=[1.0, 0.3, 0.3])
                gt_joints_tensor = gt_joints_f if isinstance(gt_joints_f, torch.Tensor) else torch.from_numpy(gt_joints_f)
                gt_j2d = perspective_projection(gt_joints_tensor.unsqueeze(0).unsqueeze(0), K_cpu.unsqueeze(0).unsqueeze(0))
                gt_j2d = gt_j2d.squeeze(0).squeeze(0).numpy()
                
                # if idx_in_vis == 0:
                #     Log.info(f"[Vis] gt_j2d range: [{gt_j2d[:, 0].min():.1f}, {gt_j2d[:, 0].max():.1f}] x [{gt_j2d[:, 1].min():.1f}, {gt_j2d[:, 1].max():.1f}]")
                #     Log.info(f"[Vis] gt_joints z range: [{gt_joints_tensor[:, 2].min():.3f}, {gt_joints_tensor[:, 2].max():.3f}]")
                
                kp2d_gt = np.concatenate([gt_j2d, np.ones((gt_j2d.shape[0], 1))], axis=-1)
                gt_overlay = draw_coco17_skeleton_batch([gt_img], [kp2d_gt])[0]
                gt_overlay = draw_bbx_xys_on_image_batch([bbx], [gt_overlay])[0]
                gt_overlays.append(gt_overlay)
                
                # pred/gt mesh compare
                bg = np.ones((H, W, 3), dtype=np.uint8) * 255
                pred_mesh_img = renderer.render_mesh(pred_verts_f, background=bg.copy(), colors=[0.2, 0.4, 1.0])
                gt_mesh_img = renderer.render_mesh(gt_verts_f, background=pred_mesh_img, colors=[1.0, 0.2, 0.2])
                pred_gt_overlays.append(gt_mesh_img)
            
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
        
        n_frames = len(pred_overlays)
        if n_frames <= 4:
            # 水平拼接所有帧
            pred_combined = stack_images_horizontal(pred_overlays)
            if gt_overlays:
                gt_combined = stack_images_horizontal(gt_overlays)
            if pred_gt_overlays:
                pred_gt_combined = stack_images_horizontal(pred_gt_overlays)
        else:
            # 分成多行，每行 4 帧
            cols = 4
            rows = (n_frames + cols - 1) // cols
            
            pred_rows = []
            gt_rows = []
            pred_gt_rows = []
            
            for r in range(rows):
                start_idx = r * cols
                end_idx = min(start_idx + cols, n_frames)
                pred_rows.append(stack_images_horizontal(pred_overlays[start_idx:end_idx]))
                if gt_overlays:
                    gt_rows.append(stack_images_horizontal(gt_overlays[start_idx:end_idx]))
                if pred_gt_overlays:
                    pred_gt_rows.append(stack_images_horizontal(pred_gt_overlays[start_idx:end_idx]))
            
            pred_combined = stack_images_vertical(pred_rows)
            if gt_rows:
                gt_combined = stack_images_vertical(gt_rows)
            if pred_gt_rows:
                pred_gt_combined = stack_images_vertical(pred_gt_rows)
    
        pred_tb = torch.from_numpy(pred_combined).permute(2, 0, 1)
        self.logger.experiment.add_image(f"{tag_prefix}/incam_pred_multi", pred_tb, self.global_step)
        if gt_overlays:
            gt_tb = torch.from_numpy(gt_combined).permute(2, 0, 1)
            self.logger.experiment.add_image(f"{tag_prefix}/incam_gt_multi", gt_tb, self.global_step)
        if pred_gt_overlays:
            pred_gt_tb = torch.from_numpy(pred_gt_combined).permute(2, 0, 1)
            self.logger.experiment.add_image(f"{tag_prefix}/incam_pred_gt_overlay_multi", pred_gt_tb, self.global_step)

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
                render_global_video(smplx_out_pred, smplx_out_gt, global_step, output_dir, tag_prefix="ego", vis_frames=60)
                # # Wis3d 可视化 (ego)
                # self._visualize_val_global(batch, smplx_out_pred, smplx_out_gt, self.smplx_full, tag_prefix="ego")
        
        # 2. Exo global 视频
        pred_smpl_params_global = outputs.get("pred_smpl_params_global", None)
        interactee_smpl_params_w = batch.get("interactee_smpl_params_w", None)
        if pred_smpl_params_global is not None and interactee_smpl_params_w is not None:
            with torch.no_grad():
                pred_reshaped = {k: v.reshape(B * F, -1) for k, v in pred_smpl_params_global.items()}
                gt_reshaped = {k: v.reshape(B * F, -1) for k, v in interactee_smpl_params_w.items()}
                smplx_out_pred = self.smplx_full(**pred_reshaped)
                smplx_out_gt = self.smplx_full(**gt_reshaped)
                for out in [smplx_out_pred, smplx_out_gt]:
                    out.vertices = out.vertices.reshape(B, F, -1, 3)
                    out.joints = out.joints.reshape(B, F, -1, 3)
                render_global_video(smplx_out_pred, smplx_out_gt, global_step, output_dir, tag_prefix="exo", vis_frames=60)
                # # Wis3d 可视化 (exo)
                # self._visualize_val_global(batch, smplx_out_pred, smplx_out_gt, self.smplx_full, tag_prefix="exo")
        
        # 3. Exo incam overlay
        self._visualize_model_output(batch, outputs, tag_prefix="val")

    # 利用wis3d进行可视化，可以用来debug  
    def _visualize_val_global(self, batch, smplx_out_pred, smplx_out_gt, smplx_model, tag_prefix="exo"):
        """可视化 validation/test 阶段的 global 预测结果
        
        使用脚底对齐地面的可视化方式，保存 ego/exo 的 gt 和 pred
        """
        wis3d = make_wis3d(name=f"val_global_motion_{tag_prefix}", time_postfix=True)
        
        device = smplx_out_gt.vertices.device
        B, F = smplx_out_gt.vertices.shape[:2]
        
        # 获取顶点和关节点 (第一个 batch)
        pred_verts = smplx_out_pred.vertices[0].float()  # (F, V_smplx, 3)
        gt_verts = smplx_out_gt.vertices[0].float()  # (F, V_smplx, 3)
        # pred_joints = smplx_out_pred.joints[0].float()  # (F, J, 3)
        # gt_joints = smplx_out_gt.joints[0].float()  # (F, J, 3)
        
        # 转换为 SMPL 顶点（J_regressor 是针对 SMPL 的）
        smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to(device).to_dense().float()
        J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").to(device).float()
        pred_verts_smpl = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in pred_verts])  # (F, V_smpl, 3)
        gt_verts_smpl = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in gt_verts])  # (F, V_smpl, 3)
        
        # 脚底对齐地面的对齐函数（和 demo.py 一致）
        def move_to_start_point_face_z(verts):
            """XZ 到原点，脚底对齐地面，面朝 Z 方向"""
            verts = verts.clone()
            offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
            offset[1] = verts[:, :, [1]].min()
            verts = verts - offset
            
            # 面朝方向
            T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
            verts = apply_T_on_points(verts, T_ay2ayfz)
            
            return verts
        
        # 对齐 pred 和 gt（各自对齐到各自的 ayfz 坐标系）
        pred_verts_aligned = move_to_start_point_face_z(pred_verts_smpl)
        gt_verts_aligned = move_to_start_point_face_z(gt_verts_smpl)
        gt_joints_aligned = einsum(J_regressor, gt_verts_aligned, "j v, l v i -> l j i")  # (F, J, 3)
        
        # 仅可视化第一个batch
        vis_frames = min(F, 30)
        
        # 获取地面参数（使用 gt 的根关节点）
        gt_root_points = gt_joints_aligned[:, 0].cpu()  # (F, 3)
        from hmr4d.utils.vis.renderer import get_ground_params_from_points
        scale, cx, cz = get_ground_params_from_points(gt_root_points, gt_verts_aligned.cpu())
        
        # 生成 checkerboard 地面几何体
        ground_v, ground_f, ground_vc, _ = checkerboard_geometry(
            length=scale * 3,  # 地面大小
            c1=cx,
            c2=cz,
            up="y",
        )
        
        # 获取 SMPL faces
        smpl_faces = make_smplx("smpl").faces
        
        for i in range(vis_frames):
            wis3d.set_scene_id(i)
            
            # 添加地面
            wis3d.add_mesh(ground_v, ground_f, ground_vc, name="ground")
            
            # 添加 pred mesh (蓝色)
            wis3d.add_mesh(
                pred_verts_aligned[i].cpu().numpy(), 
                smpl_faces, 
                name=f"pred-smplx-ayfz_{i}",
                vertex_colors=np.tile([0.3, 0.5, 1.0, 1.0], (pred_verts_aligned.shape[1], 1)).astype(np.float32)
            )
            
            # 添加 gt mesh (红色)
            wis3d.add_mesh(
                gt_verts_aligned[i].cpu().numpy(), 
                smpl_faces, 
                name=f"gt-smplx-ayfz_{i}",
                vertex_colors=np.tile([1.0, 0.3, 0.3, 1.0], (gt_verts_aligned.shape[1], 1)).astype(np.float32)
            )
        
        Log.info(f"[Vis Val Global] Saved {tag_prefix} global visualization with {vis_frames} frames")
        del wis3d

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

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

        # batch_ = {
        #     "length": batch["length"],
        #     "obs": obs,
        #     "bbx_xys": batch["bbx_xys"],
        #     "K_fullimg": batch["K_fullimg"],
        #     "cam_angvel": batch["cam_angvel"],
        #     "f_imgseq": batch["f_imgseq"],
        # }
        # outputs = self.pipeline.forward(batch_, train=False, postproc=do_postproc_not_flip_test)
        batch["obs"] = obs
        with torch.no_grad():
            gt_verts437, gt_j3d = self.smplx(**batch["interactee_smpl_params_c"])
            root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
            batch["gt_j3d"] = gt_j3d
            batch["gt_cr_coco17"] = gt_j3d - root_
            batch["gt_c_verts437"] = gt_verts437
            batch["gt_cr_verts437"] = gt_verts437 - root_
        outputs = self.pipeline.forward(batch, train=False, postproc=do_postproc_not_flip_test)
        outputs["pred_smpl_params_global"] = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}
        outputs["pred_smpl_params_incam"] = {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()}
        
        # 计算并记录 val/test loss
        B, F = batch["smpl_params_c"]["body_pose"].shape[:2]
        log_prefix = "test" if self.trainer.state.stage == "test" else "val"
        log_kwargs = {
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "batch_size": B,
            "sync_dist": True,
        }
        self.log(f"{log_prefix}/loss", outputs["loss"], **log_kwargs)
        for k, v in outputs.items():
            if "_loss" in k:
                self.log(f"{log_prefix}/{k}", v, **log_kwargs)
        
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
        if self.logger is not None:
            if self.trainer.state.stage == "test":
                do_test_vis = (
                    self.test_vis_max_batches > 0
                    and batch_idx < self.test_vis_max_batches
                    and batch_idx % self.test_vis_every_n_batches == 0
                )
                if do_test_vis:
                    self._visualize_validation(batch, outputs)
            else:
                current_step = self.trainer.global_step
                do_val_vis = self.val_vis_every_n_steps > 0 and current_step % self.val_vis_every_n_steps == 0 and batch_idx < 2
                if do_val_vis:
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