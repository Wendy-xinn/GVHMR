#!/usr/bin/env python3
"""Visualize paired EgoBody ego/exo SMPL-X GT in a shared coordinate frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hmr4d.utils.smplx_utils import make_smplx

COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _load(out_dir: Path):
    return (
        torch.load(out_dir / 'manifest.pt', map_location='cpu'),
        torch.load(out_dir / 'smplx_gt.pt', map_location='cpu'),
        torch.load(out_dir / 'camera_head_traj.pt', map_location='cpu'),
        torch.load(out_dir / 'calibration.pt', map_location='cpu'),
    )


def _transform_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    R = T[:3, :3]
    t = T[:3, 3]
    return torch.einsum('ij,...j->...i', R, points) + t


def _points_in_coord(points_kinect: torch.Tensor, coord: str, calib: dict) -> torch.Tensor:
    if coord == 'kinect12':
        return points_kinect
    if coord == 'holo':
        return _transform_points(points_kinect, calib['T_kinect12_to_holo'])
    if coord == 'scene':
        return _transform_points(points_kinect, calib['T_kinect12_to_scene'])
    raise ValueError(coord)


def _head_in_coord(traj: dict, coord: str, calib: dict) -> torch.Tensor:
    head_holo = traj['T_holo_head'][:, :3, 3]
    if coord == 'holo':
        return head_holo
    if coord == 'kinect12':
        return _transform_points(head_holo, calib['T_holo_to_kinect12'])
    if coord == 'scene':
        return _transform_points(head_holo, calib['T_holo_to_scene'])
    raise ValueError(coord)



def _smpl_lite_inputs(params: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        'betas': params['betas'].to(device),
        'global_orient': params['global_orient'].to(device),
        'transl': params['transl'].to(device),
        'body_pose': params['body_pose'].to(device),
    }


def _compute_geometry(params: dict[str, torch.Tensor], batch_size: int, device: torch.device):
    smplx_lite = make_smplx('supermotion_v437coco17').to(device).eval()
    joints, verts = [], []
    L = params['body_pose'].shape[0]
    with torch.no_grad():
        for start in range(0, L, batch_size):
            end = min(start + batch_size, L)
            out_verts, out_joints = smplx_lite(**_smpl_lite_inputs({k: v[start:end] for k, v in params.items()}, device))
            verts.append(out_verts.cpu())
            joints.append(out_joints.cpu())
    return torch.cat(joints, dim=0), torch.cat(verts, dim=0)


def _get_geometry(role_data: dict, batch_size: int, device: torch.device):
    if 'joints17_kinect' in role_data and 'verts437_kinect' in role_data:
        return role_data['joints17_kinect'], role_data['verts437_kinect']
    return _compute_geometry(role_data['kinect12'], batch_size=batch_size, device=device)

def _draw_skeleton(ax, joints: np.ndarray, color: str, label: str) -> None:
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=18, color=color, label=label)
    for a, b in COCO17_EDGES:
        if a < len(joints) and b < len(joints):
            seg = joints[[a, b]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=2)


def _set_equal_axes(ax, pts: np.ndarray) -> None:
    center = np.nanmean(pts, axis=0)
    radius = np.nanmax(np.linalg.norm(pts - center, axis=1))
    radius = max(float(radius), 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def _parse_frames(spec: str, length: int) -> list[int]:
    if spec == 'auto':
        candidates = [0, length // 3, (2 * length) // 3, length - 1]
    else:
        candidates = [int(x) for x in spec.split(',') if x.strip()]
    return sorted(set(max(0, min(length - 1, x)) for x in candidates))


def visualize(args: argparse.Namespace) -> Path:
    out_dir = args.preprocess_dir
    manifest, smplx_gt, traj, calib = _load(out_dir)
    length = int(manifest['length'])
    frames = _parse_frames(args.frames, length)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu_smpl else 'cpu')
    exo_j_k, exo_v_k = _get_geometry(smplx_gt['exo'], args.smpl_batch_size, device)
    ego_j_k, ego_v_k = _get_geometry(smplx_gt['ego'], args.smpl_batch_size, device)
    exo_j = _points_in_coord(exo_j_k, args.coord, calib)
    ego_j = _points_in_coord(ego_j_k, args.coord, calib)
    exo_v = _points_in_coord(exo_v_k, args.coord, calib)
    ego_v = _points_in_coord(ego_v_k, args.coord, calib)
    head = _head_in_coord(traj, args.coord, calib)

    debug_dir = out_dir / 'debug'
    debug_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5.2 * len(frames), 5.2))
    for i, f in enumerate(frames):
        ax = fig.add_subplot(1, len(frames), i + 1, projection='3d')
        ej = exo_j[f].numpy()
        gj = ego_j[f].numpy()
        hv = head[f].numpy()
        if args.show_verts:
            step_exo = max(1, exo_v.shape[1] // args.max_verts)
            step_ego = max(1, ego_v.shape[1] // args.max_verts)
            ax.scatter(*exo_v[f, ::step_exo].numpy().T, s=1, alpha=0.15, color='green')
            ax.scatter(*ego_v[f, ::step_ego].numpy().T, s=1, alpha=0.15, color='royalblue')
        _draw_skeleton(ax, ej, 'green', 'exo GT')
        _draw_skeleton(ax, gj, 'royalblue', 'ego GT')
        ax.scatter([hv[0]], [hv[1]], [hv[2]], s=60, color='red', marker='x', label='tracked head')
        ax.plot(head[: f + 1, 0].numpy(), head[: f + 1, 1].numpy(), head[: f + 1, 2].numpy(), color='red', alpha=0.45, linewidth=1)
        pts = np.concatenate([ej, gj, hv[None]], axis=0)
        _set_equal_axes(ax, pts)
        ax.set_title(f'{args.coord} frame {int(manifest["frame_ids"][f])}')
        ax.legend(loc='upper right')
        ax.view_init(elev=args.elev, azim=args.azim)
    fig.tight_layout()
    out_path = debug_dir / f'smplx_gt_{args.coord}_frames.png'
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    # Also save a trajectory-level plot using root translations plus tracked head.
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    exo_root = _points_in_coord(smplx_gt['exo']['kinect12']['transl'], args.coord, calib).numpy()
    ego_root = _points_in_coord(smplx_gt['ego']['kinect12']['transl'], args.coord, calib).numpy()
    head_np = head.numpy()
    ax.plot(exo_root[:, 0], exo_root[:, 1], exo_root[:, 2], color='green', label='exo root')
    ax.plot(ego_root[:, 0], ego_root[:, 1], ego_root[:, 2], color='royalblue', label='ego root')
    ax.plot(head_np[:, 0], head_np[:, 1], head_np[:, 2], color='red', label='tracked head')
    _set_equal_axes(ax, np.concatenate([exo_root, ego_root, head_np], axis=0))
    ax.set_title(f'GT trajectories in {args.coord}')
    ax.legend()
    ax.view_init(elev=args.elev, azim=args.azim)
    fig.tight_layout()
    traj_path = debug_dir / f'smplx_gt_{args.coord}_trajectories.png'
    fig.savefig(traj_path, dpi=170)
    plt.close(fig)

    print(out_path)
    print(traj_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preprocess-dir', type=Path, required=True)
    p.add_argument('--coord', choices=['holo', 'scene', 'kinect12'], default='holo')
    p.add_argument('--frames', default='auto', help='auto or comma-separated frame indices within the processed sequence')
    p.add_argument('--show-verts', action='store_true')
    p.add_argument('--max-verts', type=int, default=220)
    p.add_argument('--elev', type=float, default=18.0)
    p.add_argument('--azim', type=float, default=-62.0)
    p.add_argument('--smpl-batch-size', type=int, default=128)
    p.add_argument('--cpu-smpl', action='store_true')
    return p


def main() -> None:
    visualize(build_parser().parse_args())


if __name__ == '__main__':
    main()
