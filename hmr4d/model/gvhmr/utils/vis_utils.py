"""
Ego/Exo 可视化工具
- Global 3D 可视化：ego 和 exo 分开，GT 和 pred 同视频左右分屏对比
"""
import torch
import numpy as np
from pathlib import Path
from einops import einsum
from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import get_writer
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from hmr4d.utils.geo_transform import compute_T_ayfz2ay, apply_T_on_points
from hmr4d.utils.smplx_utils import make_smplx
import cv2


def render_global_video(smplx_out_pred, smplx_out_gt, global_step, output_dir, tag_prefix="ego", vis_frames=30):
    """
    渲染 global 3D 可视化视频：GT 和 pred 放在同一个视频里对比（左右分屏）
    
    Args:
        smplx_out_pred: 预测的 SMPLX 输出（包含 vertices, joints），形状 (B, F, V, 3)
        smplx_out_gt: GT 的 SMPLX 输出，形状 (B, F, V, 3)
        global_step: 当前全局步数
        output_dir: 输出目录
        tag_prefix: 视频文件名前缀（"ego" 或 "exo"）
        vis_frames: 可视化帧数
    """
    device = smplx_out_gt.vertices.device
    
    # 获取第一个 batch 的数据
    pred_verts = smplx_out_pred.vertices[0].float()  # (F, V_smplx, 3)
    gt_verts = smplx_out_gt.vertices[0].float()  # (F, V_smplx, 3)
    
    # 转换为 SMPL 顶点（使用 torch.load 加载，和原代码一致）
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to(device).to_dense().float()
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").to(device).float()
    
    pred_verts_smpl = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in pred_verts])  # (F, V_smpl, 3)
    gt_verts_smpl = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in gt_verts])  # (F, V_smpl, 3)
    
    # 计算关节点
    pred_joints = einsum(J_regressor, pred_verts_smpl, "j v, l v i -> l j i")  # (F, J, 3)
    gt_joints = einsum(J_regressor, gt_verts_smpl, "j v, l v i -> l j i")  # (F, J, 3)
    
    # 对齐到起点（面朝 Z 方向）
    def move_to_start_point_face_z(verts_smpl, joints):
        """XZ 到原点，脚底对齐地面，面朝 Z 方向"""
        verts_smpl = verts_smpl.clone()
        offset = joints[0, 0].clone()  # (3)
        offset[1] = verts_smpl[:, :, 1].min()  # 脚底高度
        verts_smpl = verts_smpl - offset
        
        # 面朝方向
        joints_aligned = joints[[0]]  # (1, J, 3)
        T_ay2ayfz = compute_T_ayfz2ay(joints_aligned, inverse=True)  # (1, 4, 4)
        verts_smpl = apply_T_on_points(verts_smpl, T_ay2ayfz)
        joints = apply_T_on_points(joints, T_ay2ayfz)
        
        return verts_smpl, joints
    
    pred_verts_aligned, pred_joints_aligned = move_to_start_point_face_z(pred_verts_smpl, pred_joints)
    gt_verts_aligned, gt_joints_aligned = move_to_start_point_face_z(gt_verts_smpl, gt_joints)
    
    # 获取地面参数
    pred_root_points = pred_joints_aligned[:, 0].cpu()  # (F, 3)
    scale, cx, cz = get_ground_params_from_points(pred_root_points, pred_verts_aligned.cpu())
    
    # 创建 Renderer
    length = pred_verts_aligned.shape[0]
    width, height = 1920, 1080
    K = torch.tensor([[1.0, 0, width/2], [0, 1.0, height/2], [0, 0, 1]]).float()
    faces_smpl = make_smplx("smpl").faces
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)
    renderer.set_ground(scale * 1.5, cx, cz)
    
    # 获取相机参数
    global_R, global_T, global_lights = get_global_cameras_static(
        pred_verts_aligned.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )
    
    # 创建输出目录
    output_dir = Path(output_dir) / "outputs" / 'videos' / f"vis_{tag_prefix}_global_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GT 和 pred 放在同一个视频里（左右分屏）
    video_path = output_dir / f"{tag_prefix}_global_{global_step}.mp4"
    writer = get_writer(str(video_path), fps=10)
    
    vis_frames = min(vis_frames, length)
    color_pred = torch.tensor([0.3, 0.5, 1.0]).float().cuda()  # 蓝色
    color_gt = torch.tensor([1.0, 0.3, 0.3]).float().cuda()  # 红色
    
    # 禁用 autocast，确保渲染在 FP32 下进行（PyTorch3D 不支持 Half）
    with torch.autocast(device_type="cuda", enabled=False):
        for i in range(vis_frames):
            cameras = renderer.create_camera(global_R[i], global_T[i])
            
            # 渲染 pred（左半部分）
            pred_img = renderer.render_with_ground(
                pred_verts_aligned[[i]].cuda().float(), 
                color_pred[None], 
                cameras, 
                global_lights
            )
            
            # 渲染 GT（右半部分）
            gt_img = renderer.render_with_ground(
                gt_verts_aligned[[i]].cuda().float(), 
                color_gt[None], 
                cameras, 
                global_lights
            )
            
            # 左右拼接
            combined_img = np.concatenate([pred_img, gt_img], axis=1)
            writer.write_frame(combined_img)
    
    writer.close()
    
    Log.info(f"Saved {tag_prefix} global video: {video_path}")
    
    del renderer
