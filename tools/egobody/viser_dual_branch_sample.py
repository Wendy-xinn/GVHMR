import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from hmr4d.configs import register_store_gvhmr
from hmr4d.datamodule.mocap_trainX_testY import collate_fn
from hmr4d.dataset.egobody.egobody_egoexo_v1 import EgoBodyEgoExoV1Dataset
from hmr4d.utils.net_utils import load_pretrained_model
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import _make_fixed_observer_ego_cond
from tools.egobody.render_dual_branch_sample import move_to_device, make_exo_input_batch, make_ego_input_batch


COLORS = {
    "exo_gt": (38, 115, 255),
    "ego_gt": (26, 191, 64),
    "exo_pred": (140, 38, 217),
    "ego_pred_from_exo_input": (255, 140, 13),
}


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)



def _compute_ground_y(mesh_items, source):
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


def _add_camera_nodes(server, prefix, T_world_cam_seq, T_world_cpf_seq):
    handles_by_t = []
    F = T_world_cam_seq.shape[0]
    for t in range(F):
        handles = []
        visible = t == 0
        T_cam = T_world_cam_seq[t]
        T_cpf = T_world_cpf_seq[t]
        pts = _camera_frustum_points(T_cam, scale=0.24)
        frustum_edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        for i, (a, b) in enumerate(frustum_edges):
            handles.append(
                server.scene.add_spline_catmull_rom(
                    f"/{prefix}/t{t}/camera/frustum_{i}",
                    _to_numpy(torch.stack([pts[a], pts[b]], dim=0)),
                    line_width=3.0,
                    color=(255, 170, 20),
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
                    f"/{prefix}/t{t}/camera/{name}",
                    _to_numpy(torch.stack([pts[a], pts[b]], dim=0)),
                    line_width=4.0,
                    color=color,
                    visible=visible,
                )
            )
        handles.append(
            server.scene.add_spline_catmull_rom(
                f"/{prefix}/t{t}/camera_to_virtual_cpf",
                _to_numpy(torch.stack([T_cam[:3, 3], T_cpf[:3, 3]], dim=0)),
                line_width=4.0,
                color=(255, 190, 30),
                visible=visible,
            )
        )
        handles.append(
            server.scene.add_point_cloud(
                f"/{prefix}/t{t}/virtual_cpf_marker",
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
    mesh_items = [
        ("exo_gt", model._params_to_smpl_verts(batch["exo"]["smpl_params_w"]), COLORS["exo_gt"]),
        ("ego_gt", model._params_to_smpl_verts(batch["ego"]["smpl_params_w"]), COLORS["ego_gt"]),
        ("exo_pred", model._params_to_smpl_verts(outputs.get("pred_smpl_params_kinect_from_incam")), COLORS["exo_pred"]),
        (f"ego_pred_from_{args.input_role}_input", model._params_to_smpl_verts(outputs.get("pred_smpl_params_global_ego")), COLORS["ego_pred_from_exo_input"]),
    ]
    mesh_items = [(n, v, c) for n, v, c in mesh_items if v is not None and torch.isfinite(v).all()]
    if not mesh_items:
        raise RuntimeError("No valid meshes to visualize")
    ground_y, ground_sources = _compute_ground_y(mesh_items, args.ground_source)
    shifted_items = []
    for name, verts, color in mesh_items:
        verts = verts.detach().clone()
        verts[..., 1] -= ground_y.to(verts.device, verts.dtype)
        shifted_items.append((name, verts.cpu().float().numpy(), color))

    marker_ego_cond = eval_batch.get("ego_cond", {})
    if args.input_role == "exo":
        marker_ego_cond = _make_fixed_observer_ego_cond(eval_batch, marker_ego_cond, cfg.pipeline.args.virtual_ego_from_exo)
    T_world_cam = marker_ego_cond["T_world_pv"].detach().clone()
    T_world_cpf = marker_ego_cond["T_world_cpf"].detach().clone()
    T_world_cam[..., 1, 3] -= ground_y.to(device=device, dtype=T_world_cam.dtype)
    T_world_cpf[..., 1, 3] -= ground_y.to(device=device, dtype=T_world_cpf.dtype)
    T_world_cam = T_world_cam[0].cpu().float()
    T_world_cpf = T_world_cpf[0].cpu().float()

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
    camera_handles_by_t = _add_camera_nodes(server, "timesteps", T_world_cam, T_world_cpf)
    for t in range(F):
        handles_by_t[t].extend(camera_handles_by_t[t])

    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=F - 1, step=1, initial_value=0)
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)

    first_centers = {}
    for name, verts, _color in shifted_items:
        first_centers[name] = verts[0].mean(axis=0)
    cam_pos0 = _to_numpy(T_world_cam[0, :3, 3])
    cpf_pos0 = _to_numpy(T_world_cpf[0, :3, 3])
    cam_forward0 = _to_numpy(T_world_cam[0, :3, 2] * -1.0)
    cam_to_cpf0 = cpf_pos0 - cam_pos0
    forward_dot_cpf = float(np.dot(cam_to_cpf0, cam_forward0))
    marker_desc = (
        f"fixed observer CPF uses camera->head offset y={args.virtual_ego_head_camera_y_offset:.3f}, z={args.virtual_ego_back_offset:.3f}"
        if args.input_role == "exo"
        else "real ego camera/CPF from ego_cond"
    )
    coord_lines = [
        f"visual ground/grid: xz plane at y=0 after subtracting {args.ground_source} min_y={float(ground_y):.4f}",
        f"ground source meshes: {', '.join(ground_sources)}",
        marker_desc,
        f"world axes: +x red, +y green/up, +z blue/back label",
        f"camera pos frame0: ({cam_pos0[0]:.3f}, {cam_pos0[1]:.3f}, {cam_pos0[2]:.3f})",
        f"actual camera height above grid: {cam_pos0[1]:.3f}",
        f"camera forward frame0: ({cam_forward0[0]:.3f}, {cam_forward0[1]:.3f}, {cam_forward0[2]:.3f})",
        f"virtual CPF pos frame0: ({cpf_pos0[0]:.3f}, {cpf_pos0[1]:.3f}, {cpf_pos0[2]:.3f})",
        f"dot(camera->CPF, forward): {forward_dot_cpf:.3f} (negative means behind camera)",
    ]
    for name, center in first_centers.items():
        rel = center - cam_pos0
        coord_lines.append(f"{name} center0 rel camera: x={rel[0]:.3f}, y={rel[1]:.3f}, z={rel[2]:.3f}")
    print("\n".join(coord_lines))

    with server.gui.add_folder("Info"):
        meta = batch.get("meta", [{}])[0] if isinstance(batch.get("meta", None), list) else batch.get("meta", {})
        server.gui.add_markdown(
            f"sample_idx: `{args.sample_idx}`  \n"
            f"recording: `{meta.get('recording', '')}`  \n"
            f"window: `{meta.get('start', '')}-{meta.get('end', '')}`  \n"
            f"input_role: `{args.input_role}`  \n"
            f"marker: `{marker_desc}`  \n"
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

    print(f"Viser server running on http://{args.host}:{args.port}")
    print("Open the URL in a browser. Use the Timestep slider or Playing checkbox.")
    while True:
        if gui_playing.value:
            gui_timestep.value = (int(gui_timestep.value) + 1) % F
        time.sleep(1.0 / float(gui_fps.value))


if __name__ == "__main__":
    main()
