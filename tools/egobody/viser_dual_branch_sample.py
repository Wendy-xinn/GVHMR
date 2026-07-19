import argparse
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from omegaconf import open_dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from hmr4d.configs import register_store_gvhmr
from hmr4d.datamodule.mocap_trainX_testY import collate_fn
from hmr4d.dataset.egobody.egobody_egoexo_v1 import EgoBodyEgoExoV1Dataset
from hmr4d.utils.net_utils import load_pretrained_model
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.smpl_root_transform import transform_smpl_root
from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import (
    _ego_cpf_to_smpl_params_world,
    _make_cam_angvel_cpf_ego_cond,
    _make_fixed_observer_ego_cond,
    _transform_root_to_local,
)
from tools.egobody.render_dual_branch_sample import move_to_device, make_exo_input_batch, make_ego_input_batch
from hmr4d.utils.video_io_utils import save_video
from hmr4d.utils.geo.hmr_global import get_local_transl_vel


COLORS = {
    "exo_gt": (38, 115, 255),
    "ego_gt": (26, 191, 64),
    "exo_pred": (140, 38, 217),
    "ego_pred_from_exo_input": (255, 140, 13),
    "ego_pred_direct": (255, 210, 40),
    "ego_pred_vel": (255, 120, 10),
    "input_camera": (255, 170, 20),
    "gt_camera": (255, 255, 255),
    "gt_ego_pv_camera": (80, 220, 255),
}


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _invert_T(T):
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    out = torch.zeros_like(T)
    out[..., :3, :3] = R.mT
    out[..., :3, 3] = -(R.mT @ t[..., None]).squeeze(-1)
    out[..., 3, 3] = 1
    return out


def _expand_T_to_params(T, params):
    if T is None or params is None:
        return T
    B, L = params["transl"].shape[:2]
    if T.ndim == 3:
        T = T.unsqueeze(0)
    if T.shape[0] == 1 and B > 1:
        T = T.expand(B, -1, -1, -1)
    if T.shape[1] == 1 and L > 1:
        T = T.expand(-1, L, -1, -1)
    return T[:, :L]


def _transform_smpl_params_world(params, T_old_to_new):
    if params is None or T_old_to_new is None:
        return params
    T_old_to_new = _expand_T_to_params(T_old_to_new, params)
    global_orient, transl = transform_smpl_root(
        params["global_orient"], params["transl"], T_old_to_new, params.get("betas")
    )
    return {**params, "global_orient": global_orient, "transl": transl}


def _first_camera_inverse(eval_batch):
    T = eval_batch.get("T_world_cam", eval_batch.get("T_world_exo_cam", None))
    if T is None:
        return None
    if T.ndim == 3:
        T = T.unsqueeze(0)
    return _invert_T(T[:, :1])


def _transform_T_seq(T_world_cam, T_old_to_new):
    if T_world_cam is None:
        return None
    if T_world_cam.ndim == 3:
        T_world_cam = T_world_cam.unsqueeze(0)
    if T_old_to_new is None:
        return T_world_cam
    if T_old_to_new.ndim == 3:
        T_old_to_new = T_old_to_new.unsqueeze(0)
    B, L = T_world_cam.shape[:2]
    if T_old_to_new.shape[0] == 1 and B > 1:
        T_old_to_new = T_old_to_new.expand(B, -1, -1, -1)
    if T_old_to_new.shape[1] == 1 and L > 1:
        T_old_to_new = T_old_to_new.expand(-1, L, -1, -1)
    return T_old_to_new[:, :L] @ T_world_cam


def _shift_T_y(T, ground_y):
    if T is None:
        return None
    T = T.detach().clone()
    T[..., 1, 3] -= ground_y.to(device=T.device, dtype=T.dtype)
    return T


def _transl_motion_stats(name, params):
    if params is None or "transl" not in params:
        return f"{name}: missing"
    transl = params["transl"]
    if transl.ndim == 2:
        transl = transl.unsqueeze(0)
    rel = transl - transl[:, :1]
    per_frame = rel.norm(dim=-1)
    step = (transl[:, 1:] - transl[:, :-1]).norm(dim=-1) if transl.shape[1] > 1 else transl.new_zeros(transl.shape[:2])
    return (
        f"{name}: dim={tuple(transl.shape)}, "
        f"motion_max={float(per_frame.max().detach().cpu()):.4f}, "
        f"motion_mean={float(per_frame.mean().detach().cpu()):.4f}, "
        f"step_mean={float(step.mean().detach().cpu()):.4f}"
    )


def _transl_error_stats(name, pred, gt):
    if pred is None or gt is None or "transl" not in pred or "transl" not in gt:
        return f"{name}: missing"
    p = pred["transl"]
    g = gt["transl"]
    if p.ndim == 2:
        p = p.unsqueeze(0)
    if g.ndim == 2:
        g = g.unsqueeze(0)
    L = min(p.shape[1], g.shape[1])
    p = p[:, :L]
    g = g[:, :L]
    err = (p - g).norm(dim=-1)
    first = (p[:, 0] - g[:, 0]).norm(dim=-1)
    p_rel = p - p[:, :1]
    g_rel = g - g[:, :1]
    rel_err = (p_rel - g_rel).norm(dim=-1)
    return (
        f"{name}: err_mean={float(err.mean().detach().cpu()):.4f}, "
        f"err_max={float(err.max().detach().cpu()):.4f}, "
        f"first_offset={float(first.mean().detach().cpu()):.4f}, "
        f"motion_err_mean={float(rel_err.mean().detach().cpu()):.4f}"
    )



def _compute_ground_y(mesh_items, source):
    if source == "gt":
        floor_items = [(n, v, c) for n, v, c in mesh_items if n.endswith("_gt") or n.endswith("_gt_ref")]
    elif source == "pred":
        floor_items = [(n, v, c) for n, v, c in mesh_items if "pred" in n]
    elif source == "exo":
        floor_items = [(n, v, c) for n, v, c in mesh_items if n.startswith("exo")]
    elif source == "all":
        floor_items = list(mesh_items)
    else:
        raise ValueError(f"Unsupported ground source: {source}")
    floor_items = floor_items or list(mesh_items)
    ground_y = torch.cat([v[..., 1].reshape(-1).detach().float().cpu() for _, v, _ in floor_items]).min()
    return ground_y, [n for n, _, _ in floor_items]



def _camera_frustum_points(T_world_cam, scale=0.28):
    device, dtype = T_world_cam.device, T_world_cam.dtype
    d = scale * 1.8
    w = scale
    h = scale * 0.65
    local = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-w, -h, -d],
            [w, -h, -d],
            [w, h, -d],
            [-w, h, -d],
            [0.0, 0.0, -d * 1.35],
            [0.0, scale * 1.15, 0.0],
            [scale * 1.15, 0.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    return (T_world_cam[:3, :3] @ local.T).T + T_world_cam[:3, 3][None]


def _add_camera_nodes(
    server,
    prefix,
    T_world_cam_seq,
    T_world_cpf_seq=None,
    *,
    node_name="camera",
    frustum_color=(255, 170, 20),
    trajectory_color=None,
    line_width=3.0,
):
    handles_by_t = []
    F = T_world_cam_seq.shape[0]
    trajectory_color = trajectory_color or frustum_color
    cam_centers = T_world_cam_seq[:, :3, 3]
    server.scene.add_spline_catmull_rom(
        f"/{prefix}/{node_name}_trajectory",
        _to_numpy(cam_centers),
        line_width=max(2.0, line_width),
        color=trajectory_color,
    )
    for t in range(F):
        handles = []
        visible = t == 0
        T_cam = T_world_cam_seq[t]
        pts = _camera_frustum_points(T_cam, scale=0.24)
        frustum_edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        for i, (a, b) in enumerate(frustum_edges):
            handles.append(
                server.scene.add_spline_catmull_rom(
                    f"/{prefix}/t{t}/{node_name}/frustum_{i}",
                    _to_numpy(torch.stack([pts[a], pts[b]], dim=0)),
                    line_width=line_width,
                    color=frustum_color,
                    visible=visible,
                )
            )
        axis_specs = [
            ("forward", 0, 5, (20, 210, 255)),
            ("up", 0, 6, (60, 220, 60)),
            ("right", 0, 7, (255, 80, 40)),
        ]
        for name, a, b, color in axis_specs:
            handles.append(
                server.scene.add_spline_catmull_rom(
                    f"/{prefix}/t{t}/{node_name}/{name}",
                    _to_numpy(torch.stack([pts[a], pts[b]], dim=0)),
                    line_width=line_width + 1.0,
                    color=color,
                    visible=visible,
                )
            )
        handles.append(
            server.scene.add_point_cloud(
                f"/{prefix}/t{t}/{node_name}/center",
                points=_to_numpy(T_cam[:3, 3][None]),
                colors=np.array([frustum_color], dtype=np.uint8),
                point_size=0.055,
                point_shape="circle",
                visible=visible,
            )
        )
        if T_world_cpf_seq is not None:
            T_cpf = T_world_cpf_seq[t]
            handles.append(
                server.scene.add_spline_catmull_rom(
                    f"/{prefix}/t{t}/{node_name}_to_virtual_cpf",
                    _to_numpy(torch.stack([T_cam[:3, 3], T_cpf[:3, 3]], dim=0)),
                    line_width=4.0,
                    color=(255, 190, 30),
                    visible=visible,
                )
            )
            handles.append(
                server.scene.add_point_cloud(
                    f"/{prefix}/t{t}/{node_name}/virtual_cpf_marker",
                    points=_to_numpy(T_cpf[:3, 3][None]),
                    colors=np.array([[255, 190, 30]], dtype=np.uint8),
                    point_size=0.06,
                    point_shape="circle",
                    visible=visible,
                )
            )
        handles_by_t.append(handles)
    return handles_by_t


def _set_visible(handles_by_t, t):
    for i, handles in enumerate(handles_by_t):
        visible = i == t
        for h in handles:
            h.visible = visible


def _render_client_frame(client, height, width):
    image = client.get_render(height=height, width=width)
    if hasattr(image, "result"):
        image = image.result()
    if hasattr(image, "__await__"):
        import asyncio

        image = asyncio.run(image)
    if isinstance(image, (bytes, bytearray)):
        import imageio.v3 as iio
        import io

        image = iio.imread(io.BytesIO(image))
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/egobody_egoexo_v1/egobody_egoexo_stage2_both_continue/checkpoints/e099-s004900.ckpt")
    parser.add_argument("--exp", default="gvhmr/egobody_egoexo_stage2_both")
    parser.add_argument("--data-root", default="/public/home/wenxin/GVHMR/data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--input-role", choices=("exo", "ego"), default="exo")
    parser.add_argument("--motion-frames", type=int, default=128)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--virtual-ego-head-camera-y-offset", type=float, default=0.5)
    parser.add_argument("--virtual-ego-back-offset", type=float, default=0.6)
    parser.add_argument("--ground-source", choices=("gt", "pred", "exo", "all"), default="pred")
    parser.add_argument("--axis-length", type=float, default=1.0)
    parser.add_argument("--ego-pred-source", choices=("direct", "vel", "both"), default="direct", help="Which ego prediction to display: direct=main root_trans_cpf trajectory, vel=velocity-rollout debug trajectory.")
    parser.add_argument("--ego-vel-override", choices=("none", "gt-root"), default="none", help="Debug only: replace ego residual velocity with GT root velocity for the velocity-rollout mesh.")
    parser.add_argument("--ego-pose-override", choices=("none", "gt"), default="none", help="Debug only: render ego prediction with GT body_pose and betas while keeping predicted global_orient/transl.")
    parser.add_argument("--video-output-dir", default="outputs/viser_videos")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    args = parser.parse_args()

    try:
        import viser
    except ImportError as e:
        raise SystemExit(
            "viser is not installed in this environment. Install it first, e.g.\n"
            "  pip install viser\n"
            "or run this script in the egoallo environment that already has viser."
        ) from e

    register_store_gvhmr()
    config_dir = str((Path(__file__).resolve().parents[2] / "hmr4d" / "configs").resolve())
    with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = hydra.compose(config_name="train", overrides=[f"exp={args.exp}"])
    with open_dict(cfg.pipeline.args):
        cfg.pipeline.args.branch_mode = "both"
        cfg.pipeline.args.input_role = args.input_role
        cfg.pipeline.args.supervise_role = "none"
        cfg.pipeline.args.add_other_role_image_condition = False
        cfg.pipeline.args.enable_frozen_ego_image_exo = False
        cfg.pipeline.args.virtual_ego_from_exo = {
            "enabled": args.input_role == "exo",
            "mode": "fixed_observer",
            "head_camera_offset": [0.0, args.virtual_ego_head_camera_y_offset, args.virtual_ego_back_offset],
        }

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, args.ckpt)
    model.eval().cuda()

    dataset = EgoBodyEgoExoV1Dataset(
        output_root=args.data_root,
        split=args.split,
        motion_frames=args.motion_frames,
        world_coord="kinect12",
    )
    loader = DataLoader(torch.utils.data.Subset(dataset, [args.sample_idx]), batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    batch = move_to_device(next(iter(loader)), torch.device("cuda"))
    if args.input_role == "ego":
        eval_batch = make_ego_input_batch(batch)
    else:
        eval_batch = make_exo_input_batch(batch, fake_ego_origin=False)

    with torch.no_grad():
        outputs = model.pipeline.forward(eval_batch, train=False, postproc=False)

    device = next(iter(batch["exo"]["smpl_params_w"].values())).device
    # Rebuild the marker trajectory explicitly with the same logic as the
    # pipeline. Pipeline.forward works on a shallow-copied input dict, so the
    # generated ego_cond is not guaranteed to be written back to eval_batch.
    marker_base_ego_cond = eval_batch.get("ego_cond", {})
    if args.input_role == "ego":
        marker_ego_cond = _make_cam_angvel_cpf_ego_cond(eval_batch, marker_base_ego_cond, cfg.pipeline.args.virtual_ego_from_exo)
        exo_gt_params = outputs.get("exo_gt_smpl_params_w_aligned", batch["exo"]["smpl_params_w"])
        ego_gt_params = outputs.get("ego_gt_smpl_params_w_aligned", batch["ego"]["smpl_params_w"])
        exo_pred_params = outputs.get("pred_smpl_params_kinect_from_incam")
        ego_pred_direct_params = outputs.get("pred_smpl_params_global_ego_coarse", outputs.get("pred_smpl_params_global_ego"))
        ego_pred_vel_params = outputs.get("pred_smpl_params_global_ego_vel", outputs.get("pred_smpl_params_global_ego"))
        ego_pred_params = ego_pred_vel_params if args.ego_pred_source == "vel" else ego_pred_direct_params
        if ego_pred_params is None:
            ego_pred_params = ego_pred_direct_params if args.ego_pred_source == "vel" else ego_pred_vel_params
        vis_world_desc = "ego pv-cam0 rooted gravity-aligned camera-motion world"
    else:
        marker_ego_cond = _make_fixed_observer_ego_cond(eval_batch, marker_base_ego_cond, cfg.pipeline.args.virtual_ego_from_exo)
        T_abs_to_exo_cam0 = marker_ego_cond.get("T_abs_to_ego_world", _first_camera_inverse(eval_batch))
        exo_gt_params = _transform_smpl_params_world(batch["exo"]["smpl_params_w"], T_abs_to_exo_cam0)
        ego_gt_params = _transform_smpl_params_world(batch["ego"]["smpl_params_w"], T_abs_to_exo_cam0)
        exo_pred_params = _transform_smpl_params_world(outputs.get("pred_smpl_params_kinect_from_incam"), T_abs_to_exo_cam0)
        ego_pred_direct_params = outputs.get("pred_smpl_params_global_ego_coarse", outputs.get("pred_smpl_params_global_ego"))
        ego_pred_vel_params = outputs.get("pred_smpl_params_global_ego_vel", outputs.get("pred_smpl_params_global_ego"))
        ego_pred_params = ego_pred_vel_params if args.ego_pred_source == "vel" else ego_pred_direct_params
        if ego_pred_params is None:
            ego_pred_params = ego_pred_direct_params if args.ego_pred_source == "vel" else ego_pred_vel_params
        vis_world_desc = "exo camera frame0 rooted gravity-aligned camera-motion world; gt meshes are reference only"
    ego_decode = outputs.get("decode_dict_ego", {})
    ego_pred_gt_vel_params = None
    gt_vel_stats = None
    if args.ego_vel_override == "gt-root" and ego_decode and "T_world_cpf" in marker_ego_cond:
        try:
            gt_root_orient_cpf, gt_root_trans_cpf = _transform_root_to_local(
                ego_gt_params["global_orient"],
                ego_gt_params["transl"],
                marker_ego_cond["T_world_cpf"],
                ego_gt_params.get("betas"),
            )
            vel_orient = ego_decode.get("root_orient_cpf", gt_root_orient_cpf)
            gt_root_vel_cpf = get_local_transl_vel(gt_root_trans_cpf, vel_orient)
            decode_gt_vel = {
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in ego_decode.items()
            }
            decode_gt_vel["root_residual_vel_cpf"] = gt_root_vel_cpf.to(
                device=decode_gt_vel["root_trans_cpf"].device,
                dtype=decode_gt_vel["root_trans_cpf"].dtype,
            )
            ego_pred_gt_vel_params = _ego_cpf_to_smpl_params_world(
                decode_gt_vel, marker_ego_cond, model.pipeline.endecoder, cfg.pipeline.args, rollout_residual=True
            )
            gt_vel_stats = gt_root_vel_cpf
        except Exception as e:
            print(f"[WARN] GT root velocity override failed: {e}")
    ego_pred_gt_pose_params = None
    if args.ego_pose_override == "gt" and ego_pred_params is not None and ego_gt_params is not None:
        ego_pred_gt_pose_params = {
            **ego_pred_params,
            "body_pose": ego_gt_params["body_pose"].to(device=ego_pred_params["transl"].device, dtype=ego_pred_params["transl"].dtype),
            "betas": ego_gt_params["betas"].to(device=ego_pred_params["transl"].device, dtype=ego_pred_params["transl"].dtype),
        }

    debug_lines = [
        f"pred_x_ego_dim: {outputs.get('model_output', {}).get('pred_x_ego').shape[-1] if outputs.get('model_output', {}).get('pred_x_ego') is not None else 'missing'}",
        f"has_root_residual_vel_cpf: {'root_residual_vel_cpf' in ego_decode or 'local_transl_vel' in ego_decode}",
        _transl_motion_stats("ego_gt", ego_gt_params),
        _transl_motion_stats("ego_pred_direct", ego_pred_direct_params),
        _transl_motion_stats("ego_pred_vel_rollout", ego_pred_vel_params),
        _transl_error_stats("ego_direct_vs_gt", ego_pred_direct_params, ego_gt_params),
        _transl_error_stats("ego_vel_rollout_vs_gt", ego_pred_vel_params, ego_gt_params),
        f"ego_pose_override: {args.ego_pose_override}",
    ]
    ltv = ego_decode.get("root_residual_vel_cpf", ego_decode.get("local_transl_vel", None))
    if ltv is not None:
        debug_lines.append(
            f"root_residual_vel_cpf pred: abs_mean={float(ltv.abs().mean().detach().cpu()):.5f}, "
            f"abs_max={float(ltv.abs().max().detach().cpu()):.5f}"
        )
    if gt_vel_stats is not None:
        debug_lines.append(
            f"GT root vel override: abs_mean={float(gt_vel_stats.abs().mean().detach().cpu()):.5f}, "
            f"abs_max={float(gt_vel_stats.abs().max().detach().cpu()):.5f}"
        )

    mesh_items = [
        ("exo_gt_ref" if args.input_role == "exo" else "exo_gt", model._params_to_smpl_verts(exo_gt_params), COLORS["exo_gt"]),
        ("ego_gt_ref" if args.input_role == "exo" else "ego_gt", model._params_to_smpl_verts(ego_gt_params), COLORS["ego_gt"]),
        ("exo_pred", model._params_to_smpl_verts(exo_pred_params), COLORS["exo_pred"]),
    ]
    if args.ego_pred_source == "both":
        mesh_items.extend([
            (f"ego_pred_direct_from_{args.input_role}_input", model._params_to_smpl_verts(ego_pred_direct_params), COLORS["ego_pred_direct"]),
            (f"ego_pred_vel_rollout_from_{args.input_role}_input", model._params_to_smpl_verts(ego_pred_vel_params), COLORS["ego_pred_vel"]),
        ])
        if ego_pred_gt_vel_params is not None:
            mesh_items.append((f"ego_pred_gt_root_vel_from_{args.input_role}_input", model._params_to_smpl_verts(ego_pred_gt_vel_params), (245, 245, 245)))
        if ego_pred_gt_pose_params is not None:
            mesh_items.append((f"ego_pred_global_gt_pose_from_{args.input_role}_input", model._params_to_smpl_verts(ego_pred_gt_pose_params), (245, 245, 245)))
    else:
        label = "ego_pred_direct" if args.ego_pred_source == "direct" else "ego_pred_vel_rollout"
        mesh_items.append((f"{label}_from_{args.input_role}_input", model._params_to_smpl_verts(ego_pred_params), COLORS["ego_pred_from_exo_input"]))
        if ego_pred_gt_pose_params is not None:
            mesh_items.append((f"{label}_gt_pose_from_{args.input_role}_input", model._params_to_smpl_verts(ego_pred_gt_pose_params), (245, 245, 245)))
    mesh_items = [(n, v, c) for n, v, c in mesh_items if v is not None and torch.isfinite(v).all()]
    if not mesh_items:
        raise RuntimeError("No valid meshes to visualize")
    ground_y, ground_sources = _compute_ground_y(mesh_items, args.ground_source)
    shifted_items = []
    for name, verts, color in mesh_items:
        verts = verts.detach().clone()
        verts[..., 1] -= ground_y.to(verts.device, verts.dtype)
        shifted_items.append((name, verts.cpu().float().numpy(), color))

    # Draw only measured/data cameras for coordinate-system debugging. The
    # cam_angvel/virtual observer camera has no GT translation, so showing it
    # here makes it too easy to confuse a condition frame with a real camera.
    T_old_to_vis = marker_ego_cond.get("T_abs_to_ego_world", None)
    gt_exo_cam = _shift_T_y(_transform_T_seq(batch.get("T_world_exo_cam", batch.get("T_world_cam", None)), T_old_to_vis), ground_y)
    gt_ego_pv_cam = None
    if "ego_cond" in batch and "T_world_pv" in batch["ego_cond"]:
        gt_ego_pv_cam = _shift_T_y(_transform_T_seq(batch["ego_cond"]["T_world_pv"], T_old_to_vis), ground_y)

    gt_exo_cam = gt_exo_cam[0].cpu().float() if gt_exo_cam is not None else None
    gt_ego_pv_cam = gt_ego_pv_cam[0].cpu().float() if gt_ego_pv_cam is not None else None

    faces = np.asarray(make_smplx("smpl").faces, dtype=np.int32)
    F = max(verts.shape[0] for _, verts, _ in shifted_items)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/ground", plane="xz", cell_color=(90, 90, 90), section_color=(60, 60, 60), position=(0.0, 0.0, 0.0))
    axis_len = float(args.axis_length)
    server.scene.add_spline_catmull_rom("/world_axes/x_right", np.array([[0.0, 0.0, 0.0], [axis_len, 0.0, 0.0]], dtype=np.float32), line_width=4.0, color=(255, 80, 40))
    server.scene.add_spline_catmull_rom("/world_axes/y_up", np.array([[0.0, 0.0, 0.0], [0.0, axis_len, 0.0]], dtype=np.float32), line_width=4.0, color=(60, 220, 60))
    server.scene.add_spline_catmull_rom("/world_axes/z_back", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, axis_len]], dtype=np.float32), line_width=4.0, color=(80, 140, 255))

    handles_by_t = [[] for _ in range(F)]
    for t in range(F):
        visible = t == 0
        for name, verts, color in shifted_items:
            fi = min(t, verts.shape[0] - 1)
            handles_by_t[t].append(
                server.scene.add_mesh_simple(
                    f"/timesteps/{t}/{name}",
                    vertices=verts[fi],
                    faces=faces,
                    color=color,
                    visible=visible,
                )
            )
    if gt_exo_cam is not None:
        gt_exo_handles_by_t = _add_camera_nodes(
            server,
            "timesteps",
            gt_exo_cam,
            None,
            node_name="gt_exo_fixed_camera",
            frustum_color=COLORS["gt_camera"],
            trajectory_color=COLORS["gt_camera"],
            line_width=2.0,
        )
        for t in range(F):
            handles_by_t[t].extend(gt_exo_handles_by_t[t])
    if gt_ego_pv_cam is not None:
        gt_ego_pv_handles_by_t = _add_camera_nodes(
            server,
            "timesteps",
            gt_ego_pv_cam,
            None,
            node_name="gt_ego_pv_camera",
            frustum_color=COLORS["gt_ego_pv_camera"],
            trajectory_color=COLORS["gt_ego_pv_camera"],
            line_width=2.0,
        )
        for t in range(F):
            handles_by_t[t].extend(gt_ego_pv_handles_by_t[t])

    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=F - 1, step=1, initial_value=0)
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)
        gui_save_video = server.gui.add_button("Save Current View Video")
        gui_video_status = server.gui.add_markdown("`video: idle`")

    first_centers = {}
    for name, verts, _color in shifted_items:
        first_centers[name] = verts[0].mean(axis=0)
    ref_cam = gt_exo_cam if gt_exo_cam is not None else gt_ego_pv_cam
    cam_pos0 = _to_numpy(ref_cam[0, :3, 3]) if ref_cam is not None else np.zeros(3, dtype=np.float32)
    cam_forward0 = _to_numpy(ref_cam[0, :3, 2] * -1.0) if ref_cam is not None else np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gt_exo_pos0 = _to_numpy(gt_exo_cam[0, :3, 3]) if gt_exo_cam is not None else None
    gt_ego_pv_pos0 = _to_numpy(gt_ego_pv_cam[0, :3, 3]) if gt_ego_pv_cam is not None else None
    marker_desc = "camera debug only: white=GT exo fixed camera, cyan=GT ego PV trajectory"
    coord_lines = [
        f"visual ground/grid: xz plane at y=0 after subtracting {args.ground_source} min_y={float(ground_y):.4f}",
        f"ground source meshes: {', '.join(ground_sources)}",
        f"visual world: {vis_world_desc}",
        marker_desc,
        f"world axes: +x red, +y green/up, +z blue/back label",
        f"reference camera forward frame0: ({cam_forward0[0]:.3f}, {cam_forward0[1]:.3f}, {cam_forward0[2]:.3f})",
        *debug_lines,
    ]
    if gt_exo_pos0 is not None:
        rel_exo = gt_exo_pos0 - cam_pos0
        coord_lines.append(
            f"gt exo fixed camera pos frame0: ({gt_exo_pos0[0]:.3f}, {gt_exo_pos0[1]:.3f}, {gt_exo_pos0[2]:.3f}); "
            f"rel reference camera: x={rel_exo[0]:.3f}, y={rel_exo[1]:.3f}, z={rel_exo[2]:.3f}"
        )
    if gt_ego_pv_pos0 is not None:
        rel_pv = gt_ego_pv_pos0 - cam_pos0
        coord_lines.append(
            f"gt ego PV camera pos frame0: ({gt_ego_pv_pos0[0]:.3f}, {gt_ego_pv_pos0[1]:.3f}, {gt_ego_pv_pos0[2]:.3f}); "
            f"rel reference camera: x={rel_pv[0]:.3f}, y={rel_pv[1]:.3f}, z={rel_pv[2]:.3f}"
        )
    for name, center in first_centers.items():
        rel = center - cam_pos0
        coord_lines.append(f"{name} center0 rel reference camera: x={rel[0]:.3f}, y={rel[1]:.3f}, z={rel[2]:.3f}")
    print("\n".join(coord_lines))

    with server.gui.add_folder("Info"):
        meta = batch.get("meta", [{}])[0] if isinstance(batch.get("meta", None), list) else batch.get("meta", {})
        server.gui.add_markdown(
            f"sample_idx: `{args.sample_idx}`  \n"
            f"recording: `{meta.get('recording', '')}`  \n"
            f"window: `{meta.get('start', '')}-{meta.get('end', '')}`  \n"
            f"input_role: `{args.input_role}`  \n"
            f"ego_pred_source: `{args.ego_pred_source}`  \n"
            f"camera debug: `white=GT exo fixed camera, cyan=GT ego PV trajectory`  \n"
            + "  \n".join(f"`{line}`" for line in coord_lines)
        )

    state = {"last_t": 0}

    def update_frame(t):
        t = int(t)
        if t == state["last_t"]:
            return
        _set_visible(handles_by_t, t)
        state["last_t"] = t

    @gui_timestep.on_update
    def _(_event):
        update_frame(gui_timestep.value)

    save_state = {"running": False}

    def save_video_worker(client):
        save_state["running"] = True
        old_playing = bool(gui_playing.value)
        gui_playing.value = False
        try:
            out_dir = Path(args.video_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"viser_sample{args.sample_idx:04d}_{args.input_role}_{ts}.mp4"
            frames = []
            for t in range(F):
                gui_timestep.value = t
                update_frame(t)
                time.sleep(0.02)
                frames.append(_render_client_frame(client, args.video_height, args.video_width))
                gui_video_status.content = f"`video: recording {t + 1}/{F}`"
            save_video(np.stack(frames, axis=0), out_path, fps=int(gui_fps.value), crf=23)
            gui_video_status.content = f"`video saved: {out_path}`"
            print(f"Saved current-view video to {out_path}")
        except Exception as exc:
            gui_video_status.content = f"`video error: {exc}`"
            print(f"[Save video error] {exc}")
        finally:
            gui_playing.value = old_playing
            save_state["running"] = False

    @gui_save_video.on_click
    def _(event):
        if save_state["running"]:
            return
        client = getattr(event, "client", None)
        if client is None:
            clients = server.get_clients()
            client = next(iter(clients.values()), None)
        if client is None:
            gui_video_status.content = "`video error: open this viser page in a browser first`"
            return
        gui_video_status.content = "`video: starting`"
        threading.Thread(target=save_video_worker, args=(client,), daemon=True).start()

    print(f"Viser server running on http://{args.host}:{args.port}")
    print("Open the URL in a browser. Use the Timestep slider or Playing checkbox.")
    while True:
        if gui_playing.value:
            gui_timestep.value = (int(gui_timestep.value) + 1) % F
        time.sleep(1.0 / float(gui_fps.value))


if __name__ == "__main__":
    main()
