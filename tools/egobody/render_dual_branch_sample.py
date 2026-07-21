import argparse
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import einsum

from hmr4d.configs import register_store_gvhmr
from hmr4d.datamodule.mocap_trainX_testY import collate_fn
from hmr4d.dataset.egobody.egobody_egoexo_v1 import EgoBodyEgoExoV1Dataset
from hmr4d.utils.geo.hmr_cam import normalize_kp2d, create_camera_sensor
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_ground_params_from_points, get_global_cameras_static, look_at_rotation
from hmr4d.utils.net_utils import load_pretrained_model
from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import _make_fixed_observer_ego_cond


def compute_ground_y(mesh_items, source):
    if source == "gt":
        floor_items = [(n, v, c) for n, v, c in mesh_items if n.endswith("_gt")]
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


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    return x


def make_exo_input_batch(batch, fake_ego_origin=True):
    primary = batch["exo"]
    batch["smpl_params_c"] = primary["smpl_params_c"]
    batch["smpl_params_w"] = primary["smpl_params_w"]
    batch["interactee_smpl_params_c"] = primary["smpl_params_c"]
    batch["interactee_smpl_params_w"] = primary["smpl_params_w"]
    batch["bbx_xys"] = primary["bbx_xys"]
    batch["kp2d"] = primary["kp2d"]
    batch["f_imgseq"] = primary["f_imgseq"]
    batch["mask"]["valid"] = batch["mask"].get("exo_valid", batch["mask"]["valid"])
    batch["_input_role"] = "exo"
    batch["_supervise_role"] = "none"

    if fake_ego_origin and "ego_cond" in batch:
        B, L = batch["bbx_xys"].shape[:2]
        eye = torch.eye(4, device=batch["bbx_xys"].device, dtype=batch["bbx_xys"].dtype).reshape(1, 1, 4, 4).repeat(B, L, 1, 1)
        batch["ego_cond"]["T_world_head"] = eye.clone()
        batch["ego_cond"]["T_world_cpf"] = eye.clone()
        batch["ego_cond"]["T_world_pv"] = eye.clone()
        batch["ego_cond"]["head_valid"] = torch.ones((B, L), device=batch["bbx_xys"].device, dtype=torch.bool)
        batch["ego_cond"]["head_angvel"] = torch.zeros((B, L, 6), device=batch["bbx_xys"].device, dtype=batch["bbx_xys"].dtype)

    obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
    obs[~batch["mask"]["valid"]] = 0
    out = {
        "length": batch["length"],
        "obs": obs,
        "bbx_xys": batch["bbx_xys"],
        "K_fullimg": batch["K_fullimg"],
        "K_exo": batch.get("K_exo", batch["K_fullimg"]),
        "K_ego": batch.get("K_ego", batch["K_fullimg"]),
        "cam_angvel": batch["cam_angvel"],
        "R_c2gv": batch["R_c2gv"],
        "f_imgseq": batch["f_imgseq"],
        "mask": batch["mask"],
        "exo": batch["exo"],
        "ego": batch["ego"],
        "ego_cond": batch.get("ego_cond", {}),
        "_input_role": "exo",
        "_supervise_role": "none",
    }
    for key in ("T_world_cam", "T_world_exo_cam"):
        if key in batch:
            out[key] = batch[key]
    return out


def make_ego_input_batch(batch):
    primary = batch["ego"]
    batch["smpl_params_c"] = primary["smpl_params_c"]
    batch["smpl_params_w"] = primary["smpl_params_w"]
    batch["interactee_smpl_params_c"] = primary["smpl_params_c"]
    batch["interactee_smpl_params_w"] = primary["smpl_params_w"]
    batch["bbx_xys"] = primary.get("bbx_body_xys", primary["bbx_xys"])
    batch["kp2d"] = primary.get("kp2d_body", primary["kp2d"])
    batch["f_imgseq"] = primary.get("f_body_imgseq", primary["f_imgseq"])
    batch["K_fullimg"] = batch.get("K_ego", batch["K_fullimg"])
    batch["mask"]["valid"] = batch["mask"].get("ego_valid", batch["mask"]["valid"])
    batch["_input_role"] = "ego"
    batch["_supervise_role"] = "none"

    obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
    obs[~batch["mask"]["valid"]] = 0
    out = {
        "length": batch["length"],
        "obs": obs,
        "bbx_xys": batch["bbx_xys"],
        "K_fullimg": batch["K_fullimg"],
        "K_exo": batch.get("K_exo", batch["K_fullimg"]),
        "K_ego": batch.get("K_ego", batch["K_fullimg"]),
        "cam_angvel": batch["cam_angvel"],
        "R_c2gv": batch["R_c2gv"],
        "f_imgseq": batch["f_imgseq"],
        "mask": batch["mask"],
        "exo": batch["exo"],
        "ego": batch["ego"],
        "ego_cond": batch.get("ego_cond", {}),
        "_input_role": "ego",
        "_supervise_role": "none",
    }
    for key in ("T_world_cam", "T_world_exo_cam"):
        if key in batch:
            out[key] = batch[key]
    return out


def _project_world_points(points_w, cam_R, cam_T, K):
    points_w = points_w.to(device=cam_R.device, dtype=cam_R.dtype)
    points_c = (cam_R @ points_w.T).T + cam_T[None]
    z = points_c[:, 2].clamp(min=1e-4)
    u = K[0, 0] * points_c[:, 0] / z + K[0, 2]
    v = K[1, 1] * points_c[:, 1] / z + K[1, 2]
    return torch.stack([u, v], dim=-1), points_c[:, 2]


def make_fit_all_camera(verts_f, device, K, width, height, padding=0.16):
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
    pts_dev = pts.to(device).float()
    target_dev = target.to(device).float()

    # Start near the group, then back off only until every mesh fits with margin.
    distance = max(radius * 1.25, 2.2)
    max_distance = max(radius * 5.0, 8.0)
    best = None
    while distance <= max_distance:
        position = target_dev + view_dir.to(device) * distance
        position[1] = max(float(target[1] + radius * 0.35), float(position[1]))
        rotation = look_at_rotation(position[None].cpu(), target[None]).mT[0].to(device).float()
        translation = -(rotation @ position[:, None]).squeeze(-1)
        uv, z = _project_world_points(pts_dev, rotation, translation, K.to(device).float())
        valid = z > 1e-3
        if valid.any():
            uv_valid = uv[valid]
            x0, y0 = uv_valid.min(dim=0)[0]
            x1, y1 = uv_valid.max(dim=0)[0]
            margin_x = width * padding
            margin_y = height * padding
            best = (rotation, translation)
            if x0 >= margin_x and x1 <= width - margin_x and y0 >= margin_y and y1 <= height - margin_y:
                break
        distance *= 1.12
    if best is None:
        position = target_dev + view_dir.to(device) * max(radius * 2.2, 4.0)
        rotation = look_at_rotation(position[None].cpu(), target[None]).mT[0].to(device).float()
        translation = -(rotation @ position[:, None]).squeeze(-1)
        best = (rotation, translation)
    return best


def _camera_frustum_points(T_world_cam, scale=0.28):
    device, dtype = T_world_cam.device, T_world_cam.dtype
    # In the y-up Kinect world used here, the fixed exo camera looks along -z.
    d = scale * 1.8
    w = scale
    h = scale * 0.65
    local = torch.tensor(
        [
            [0.0, 0.0, 0.0],      # 0 camera center
            [-w, -h, -d],         # 1 image plane corners
            [w, -h, -d],          # 2
            [w, h, -d],           # 3
            [-w, h, -d],          # 4
            [0.0, 0.0, -d * 1.35],# 5 forward, cyan
            [0.0, scale * 1.15, 0.0],  # 6 up, green
            [scale * 1.15, 0.0, 0.0],  # 7 right, red/orange
        ],
        device=device,
        dtype=dtype,
    )
    return (T_world_cam[:3, :3] @ local.T).T + T_world_cam[:3, 3][None]


def _draw_polyline_3d(img, points_w, cam_R, cam_T, K, edges, color, thickness=2):
    uv, z = _project_world_points(points_w, cam_R, cam_T, K)
    uv = uv.detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    h, w = img.shape[:2]
    for i, j in edges:
        if z[i] <= 1e-3 or z[j] <= 1e-3:
            continue
        p0 = tuple(np.round(uv[i]).astype(int).tolist())
        p1 = tuple(np.round(uv[j]).astype(int).tolist())
        if all(-w <= p[0] <= 2 * w and -h <= p[1] <= 2 * h for p in (p0, p1)):
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def _draw_camera_overlay(img, T_world_cam_seq, frame_idx, cam_R, cam_T, K, T_world_virtual_cpf_seq=None):
    if T_world_cam_seq is None:
        return img
    T_seq = T_world_cam_seq.to(device=cam_R.device, dtype=cam_R.dtype)
    if T_seq.ndim == 4:
        T_seq = T_seq[0]
    if T_seq.numel() == 0:
        return img
    fi = min(int(frame_idx), T_seq.shape[0] - 1)
    orange = (255, 170, 20)
    cyan = (20, 210, 255)
    pts = _camera_frustum_points(T_seq[fi], scale=0.24)
    frustum_edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    _draw_polyline_3d(img, pts, cam_R, cam_T, K, frustum_edges, orange, thickness=2)
    _draw_polyline_3d(img, pts, cam_R, cam_T, K, [(0, 5)], cyan, thickness=3)
    _draw_polyline_3d(img, pts, cam_R, cam_T, K, [(0, 6)], (60, 220, 60), thickness=3)
    _draw_polyline_3d(img, pts, cam_R, cam_T, K, [(0, 7)], (255, 80, 40), thickness=3)

    if T_world_virtual_cpf_seq is not None:
        T_cpf = T_world_virtual_cpf_seq.to(device=cam_R.device, dtype=cam_R.dtype)
        if T_cpf.ndim == 4:
            T_cpf = T_cpf[0]
        if T_cpf.numel() > 0:
            cpf = T_cpf[min(int(frame_idx), T_cpf.shape[0] - 1), :3, 3]
            cam = T_seq[fi, :3, 3]
            link_pts = torch.stack([cam, cpf], dim=0)
            _draw_polyline_3d(img, link_pts, cam_R, cam_T, K, [(0, 1)], (255, 180, 40), thickness=3)
            marker = torch.stack(
                [
                    cpf + torch.tensor([-0.08, 0.0, 0.0], device=cpf.device, dtype=cpf.dtype),
                    cpf + torch.tensor([0.08, 0.0, 0.0], device=cpf.device, dtype=cpf.dtype),
                    cpf + torch.tensor([0.0, -0.08, 0.0], device=cpf.device, dtype=cpf.dtype),
                    cpf + torch.tensor([0.0, 0.08, 0.0], device=cpf.device, dtype=cpf.dtype),
                    cpf + torch.tensor([0.0, 0.0, -0.08], device=cpf.device, dtype=cpf.dtype),
                    cpf + torch.tensor([0.0, 0.0, 0.08], device=cpf.device, dtype=cpf.dtype),
                ],
                dim=0,
            )
            _draw_polyline_3d(img, marker, cam_R, cam_T, K, [(0, 1), (2, 3), (4, 5)], (255, 180, 40), thickness=3)

    traj = T_seq[:, :3, 3]
    if traj.shape[0] > 1:
        sample = torch.linspace(0, traj.shape[0] - 1, min(32, traj.shape[0]), device=traj.device).long()
        traj_pts = traj[sample]
        edges = [(i, i + 1) for i in range(traj_pts.shape[0] - 1)]
        _draw_polyline_3d(img, traj_pts, cam_R, cam_T, K, edges, cyan, thickness=2)
    cv2.putText(img, "exo camera", (18, img.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, orange, 2)
    return img


def render_world_mesh_floor_like_val(
    model,
    mesh_items,
    frame_indices,
    width=960,
    height=720,
    info_lines=None,
    T_world_cam_seq=None,
    T_world_virtual_cpf_seq=None,
    render_padding=0.12,
    render_focal_fullframe=70,
):
    device = mesh_items[0][2].device
    all_verts = torch.cat([v.detach().float().cpu() for _, v, _ in mesh_items], dim=0)
    J_regressor = model.J_regressor.to(all_verts.device).float()
    roots = einsum(J_regressor, all_verts, "j v, f v c -> f j c")[:, 0]
    scale, cx, cz = get_ground_params_from_points(roots, all_verts)
    scale = max(float(scale), 2.0)

    _, _, K = create_camera_sensor(width, height, render_focal_fullframe)
    K = K.to(device).float()
    faces_smpl = torch.as_tensor(make_smplx("smpl").faces, device=device).long()
    renderer = Renderer(width, height, device=str(device), faces=faces_smpl, K=K, bin_size=0)
    renderer.set_ground(scale * 3.0, cx, cz)

    F = max(v.shape[0] for _, v, _ in mesh_items)
    ref_verts_for_lights = []
    for frame_idx in range(F):
        frame_verts = []
        for _, verts, _ in mesh_items:
            fi = min(int(frame_idx), verts.shape[0] - 1)
            frame_verts.append(verts[fi].detach().float().cpu())
        ref_verts_for_lights.append(torch.cat(frame_verts, dim=0))
    ref_verts_for_lights = torch.stack(ref_verts_for_lights, dim=0)
    _, _, lights = get_global_cameras_static(
        ref_verts_for_lights, beta=3.2, cam_height_degree=25, target_center_height=1.0, device=str(device)
    )

    fit_verts = torch.cat([verts.to(device).float() for _, verts, _ in mesh_items], dim=0)
    cam_R_fixed, cam_T_fixed = make_fit_all_camera(fit_verts, device, K, width, height, padding=render_padding)

    images = []
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
            cam_R, cam_T = cam_R_fixed, cam_T_fixed
            cameras = renderer.create_camera(cam_R, cam_T)
            img = renderer.render_with_ground(verts_f, colors_f, cameras, lights)
            img = np.ascontiguousarray(img)
            img = _draw_camera_overlay(img, T_world_cam_seq, frame_idx, cam_R, cam_T, K, T_world_virtual_cpf_seq)
            y = 24
            for name, _, color in mesh_items:
                rgb = tuple((color.detach().cpu().numpy() * 255).astype(np.uint8).tolist())
                cv2.putText(img, name, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rgb, 2)
                y += 22
            if info_lines:
                for line in info_lines:
                    cv2.putText(img, str(line), (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 3)
                    cv2.putText(img, str(line), (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                    y += 18
            cv2.putText(img, f"frame {int(frame_idx)}", (width - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
            images.append(img)
    del renderer
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/egobody_egoexo_v1/egobody_egoexo_stage2_both_continue/checkpoints/e099-s004900.ckpt")
    parser.add_argument("--exp", default="gvhmr/egobody_egoexo_stage2_both")
    parser.add_argument("--data-root", default="/public/home/wenxin/GVHMR/data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--input-role", choices=("exo", "ego"), default="exo")
    parser.add_argument("--motion-frames", type=int, default=128)
    parser.add_argument("--out", default="outputs/egobody_egoexo_v1/debug/dual_branch_exo_sample.png")
    parser.add_argument("--real-ego-head", action="store_true")
    parser.add_argument("--virtual-ego-head-camera-y-offset", type=float, default=0.5)
    parser.add_argument("--virtual-ego-back-offset", type=float, default=0.6)
    parser.add_argument("--ground-source", choices=("gt", "pred", "exo", "all"), default="pred")
    parser.add_argument("--render-padding", type=float, default=0.10)
    parser.add_argument("--render-focal-fullframe", type=float, default=70.0)
    parser.add_argument("--video-out", default="", help="Optional mp4 path. Renders every frame with the same floor/camera convention as the debug image.")
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--video-stride", type=int, default=1)
    args = parser.parse_args()

    register_store_gvhmr()
    config_dir = str((Path(__file__).resolve().parents[2] / "hmr4d" / "configs").resolve())
    with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = hydra.compose(config_name="train", overrides=[f"exp={args.exp}"])
    cfg.pipeline.args.branch_mode = "both"
    cfg.pipeline.args.input_role = args.input_role
    cfg.pipeline.args.supervise_role = "none"
    cfg.pipeline.args.add_other_role_image_condition = False
    cfg.pipeline.args.enable_frozen_ego_image_exo = False
    cfg.pipeline.args.virtual_ego_from_exo = {
        "enabled": args.input_role == "exo" and not args.real_ego_head,
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
    subset = torch.utils.data.Subset(dataset, [args.sample_idx])
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    batch = move_to_device(next(iter(loader)), torch.device("cuda"))
    if args.input_role == "ego":
        eval_batch = make_ego_input_batch(batch)
    else:
        eval_batch = make_exo_input_batch(batch, fake_ego_origin=False)

    with torch.no_grad():
        outputs = model.pipeline.forward(eval_batch, train=False, postproc=False)

    device = next(iter(batch["exo"]["smpl_params_w"].values())).device

    meta = batch.get("meta", [{}])[0] if isinstance(batch.get("meta", None), list) else batch.get("meta", {})
    exo_img = batch.get("exo_imgname", [[""]])[0]
    if isinstance(exo_img, list) and exo_img:
        exo_img = exo_img[0]
    exo_img_name = Path(str(exo_img)).name
    info_lines = [
        f"sample_idx: {args.sample_idx}",
        f"recording: {meta.get('recording', '')}",
        f"start: {meta.get('start', '')}",
        f"input_role: {args.input_role}",
        f"exo_img: {exo_img_name}",
        (
            f"marker: fixed_observer camera_to_head_dy={args.virtual_ego_head_camera_y_offset:.2f} back={args.virtual_ego_back_offset:.2f}"
            if args.input_role == "exo" and not args.real_ego_head
            else "marker: real ego camera/CPF from ego_cond"
        ),
    ]

    mesh_items = [
        ("exo_gt", model._params_to_smpl_verts(batch["exo"]["smpl_params_w"]), torch.tensor([0.15, 0.45, 1.0], device=device)),
        ("ego_gt", model._params_to_smpl_verts(batch["ego"]["smpl_params_w"]), torch.tensor([0.1, 0.75, 0.25], device=device)),
        ("exo_pred", model._params_to_smpl_verts(outputs.get("pred_smpl_params_kinect_from_incam")), torch.tensor([0.55, 0.15, 0.85], device=device)),
        (f"ego_pred_from_{args.input_role}_input", model._params_to_smpl_verts(outputs.get("pred_smpl_params_global_ego")), torch.tensor([1.0, 0.55, 0.05], device=device)),
    ]
    mesh_items = [(n, v, c) for n, v, c in mesh_items if v is not None and torch.isfinite(v).all()]
    if not mesh_items:
        raise RuntimeError("No valid meshes to render")
    ground_y, ground_sources = compute_ground_y(mesh_items, args.ground_source)
    info_lines.append(f"ground_y from {args.ground_source} meshes: {float(ground_y):.4f} ({', '.join(ground_sources)})")
    mesh_items = [(n, v.clone(), c) for n, v, c in mesh_items]
    for _, verts, _ in mesh_items:
        verts[..., 1] = verts[..., 1] - ground_y.to(verts.device, verts.dtype)

    marker_ego_cond = eval_batch.get("ego_cond", {})
    if args.input_role == "exo":
        marker_ego_cond = _make_fixed_observer_ego_cond(eval_batch, marker_ego_cond, cfg.pipeline.args.virtual_ego_from_exo)
    T_world_cam = marker_ego_cond["T_world_pv"].detach().clone()
    T_world_virtual_cpf = marker_ego_cond["T_world_cpf"].detach().clone()
    T_world_cam[..., 1, 3] = T_world_cam[..., 1, 3] - ground_y.to(device=device, dtype=T_world_cam.dtype)
    T_world_virtual_cpf[..., 1, 3] = T_world_virtual_cpf[..., 1, 3] - ground_y.to(device=device, dtype=T_world_virtual_cpf.dtype)

    F = max(v.shape[0] for _, v, _ in mesh_items)
    frame_indices = np.linspace(0, max(F - 1, 0), min(4, F), dtype=int)
    images = render_world_mesh_floor_like_val(
        model,
        mesh_items,
        frame_indices,
        width=960,
        height=720,
        info_lines=info_lines,
        T_world_cam_seq=T_world_cam,
        T_world_virtual_cpf_seq=T_world_virtual_cpf,
        render_padding=args.render_padding,
        render_focal_fullframe=args.render_focal_fullframe,
    )
    if not images:
        raise RuntimeError("Renderer produced no images")
    out_img = np.concatenate(images, axis=1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(out_path)

    if args.video_out:
        stride = max(int(args.video_stride), 1)
        video_indices = np.arange(0, F, stride, dtype=int)
        video_images = render_world_mesh_floor_like_val(
            model,
            mesh_items,
            video_indices,
            width=960,
            height=720,
            info_lines=info_lines,
            T_world_cam_seq=T_world_cam,
            T_world_virtual_cpf_seq=T_world_virtual_cpf,
            render_padding=args.render_padding,
            render_focal_fullframe=args.render_focal_fullframe,
        )
        video_path = Path(args.video_out)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(args.video_fps), (960, 720))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")
        for img in video_images:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()
        print(video_path)


if __name__ == "__main__":
    main()
