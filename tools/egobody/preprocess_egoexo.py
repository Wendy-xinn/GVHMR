#!/usr/bin/env python3
"""Offline EgoBody preprocessing for paired ego/exo GVHMR training.

This script deliberately avoids detector-based training labels. It uses SMPL-X GT
and calibration to derive training boxes, saves image features separately, and
stores enough coordinate transforms to choose the training world frame later.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
from hmr4d.utils.smpl_root_transform import transform_smpl_root
from tqdm import tqdm

from hmr4d.utils.geo.hmr_cam import get_bbx_xys, safely_render_x3d_K
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.preproc.vitfeat_extractor import Extractor, get_batch
from hmr4d.utils.smplx_utils import make_smplx


HEAD_HAND_EYE_COLS = 861
HAND_JOINT_COUNT = 26
HAND_PALM_INDEX = 0
HAND_WRIST_INDEX = 1
HAND_INDEX_METACARPAL_INDEX = 6
HAND_LITTLE_METACARPAL_INDEX = 21
SMPLX_KEYS = [
    "betas",
    "global_orient",
    "transl",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "expression",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
]


def _save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def _load_json_transform(path: Path) -> torch.Tensor:
    data = json.loads(path.read_text())
    return torch.tensor(data["trans"], dtype=torch.float32)


def _load_kinect12_to_scene(root: Path, recording: str) -> torch.Tensor:
    calib_dir = root / "calibrations" / recording / "cal_trans" / "kinect12_to_world"
    matches = sorted(calib_dir.glob("*.json"))
    if not matches:
        raise FileNotFoundError(f"No kinect12_to_world json under {calib_dir}")
    return _load_json_transform(matches[0])


def _load_holo_to_kinect12(root: Path, recording: str) -> torch.Tensor:
    path = root / "calibrations" / recording / "cal_trans" / "holo_to_kinect12.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_json_transform(path)


def _load_exo_intrinsics(root: Path, rel_path: str) -> torch.Tensor:
    path = root / rel_path
    data = json.loads(path.read_text())
    return torch.tensor(data["camera_mtx"], dtype=torch.float32)


def _get_split(root: Path, recording: str) -> str:
    df = pd.read_csv(root / "data_splits.csv")
    for split in ("train", "val", "test"):
        if split in df.columns and recording in df[split].dropna().tolist():
            return split
    raise ValueError(f"Cannot find split for {recording}")


def _get_frame_range(root: Path, recording: str) -> Optional[tuple[int, int]]:
    path = root / "data_info_release.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    name_col = "recording_name" if "recording_name" in df.columns else "recording"
    rows = df[df[name_col] == recording]
    if len(rows) == 0:
        return None
    row = rows.iloc[0]
    if "start_frame" in row and "end_frame" in row:
        return int(row["start_frame"]), int(row["end_frame"])
    return None


def _collect_images(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def _parse_pv_image(path: Path) -> tuple[int, int]:
    timestamp, frame = path.stem.split("_frame_")
    return int(timestamp), int(frame)


def _parse_kinect_image(path: Path) -> int:
    return int(path.stem.split("frame_")[-1])


def _find_pv_dir(root: Path, recording: str) -> Path:
    rec_dir = root / "egocentric_color" / recording
    matches = sorted(rec_dir.glob("*/PV"))
    if not matches:
        raise FileNotFoundError(f"No PV dir under {rec_dir}")
    return matches[0]


def _find_pv_txt(root: Path, recording: str) -> Path:
    rec_dir = root / "egocentric_color" / recording
    matches = sorted(rec_dir.glob("*/*_pv.txt"))
    if not matches:
        raise FileNotFoundError(f"No pv.txt under {rec_dir}")
    return matches[0]


def _read_pv_txt(path: Path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    cx, cy, w, h = [float(x) for x in lines[0].split(",")]
    meta = {"cx": cx, "cy": cy, "w": w, "h": h}
    frames = {}
    for line in lines[1:]:
        parts = line.split(",")
        ts = int(parts[0].strip())
        fx, fy = float(parts[1]), float(parts[2])
        mat = np.asarray([float(x) for x in parts[3:]], dtype=np.float64).reshape(4, 4)
        frames[ts] = {"fx": fx, "fy": fy, "pv2world": mat}
    return meta, frames


def _nearest_indices(query: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(source)
    source_sorted = source[order]
    right = np.searchsorted(source_sorted, query, side="left")
    left = np.clip(right - 1, 0, len(source_sorted) - 1)
    right = np.clip(right, 0, len(source_sorted) - 1)
    diff_left = np.abs(source_sorted[left] - query)
    diff_right = np.abs(source_sorted[right] - query)
    use_right = diff_right < diff_left
    sorted_idx = np.where(use_right, right, left)
    idx = order[sorted_idx]
    matched = source[idx]
    diff = np.minimum(diff_left, diff_right)
    return idx, matched, diff


def _nearest_pv(query_ts: np.ndarray, pv_meta: dict, pv_frames: dict):
    pv_ts = np.asarray(sorted(pv_frames.keys()), dtype=np.int64)
    idx, matched, diff = _nearest_indices(query_ts, pv_ts)
    mats, Ks = [], []
    for ts in pv_ts[idx]:
        f = pv_frames[int(ts)]
        mats.append(f["pv2world"])
        Ks.append(np.asarray([[f["fx"], 0.0, pv_meta["cx"]], [0.0, f["fy"], pv_meta["cy"]], [0.0, 0.0, 1.0]], dtype=np.float32))
    return np.stack(mats).astype(np.float32), np.stack(Ks).astype(np.float32), matched, diff


def _find_head_hand_eye_csv(root: Path, recording: str) -> Path:
    matches = sorted((root / "egocentric_gaze" / recording).glob("*/*_head_hand_eye.csv"))
    if not matches:
        raise FileNotFoundError(f"No head_hand_eye csv for {recording}")
    return matches[0]


def _read_head_hand_eye(path: Path) -> dict[str, np.ndarray]:
    raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None]
    if raw.shape[1] < HEAD_HAND_EYE_COLS:
        raise ValueError(f"Expected at least {HEAD_HAND_EYE_COLS} columns in {path}, got {raw.shape[1]}")
    left_valid_col = 17
    left_start = 18
    left_end = left_start + HAND_JOINT_COUNT * 16
    right_valid_col = left_end
    right_start = right_valid_col + 1
    right_end = right_start + HAND_JOINT_COUNT * 16
    head = raw[:, 1:17].reshape(-1, 4, 4)
    left = raw[:, left_start:left_end].reshape(-1, HAND_JOINT_COUNT, 4, 4)[:, :, :3, 3]
    right = raw[:, right_start:right_end].reshape(-1, HAND_JOINT_COUNT, 4, 4)[:, :, :3, 3]
    gaze_origin = raw[:, 852:855]
    gaze_direction = raw[:, 856:859]
    gaze_distance = raw[:, 860:861]
    finite_head = np.isfinite(head).all(axis=(1, 2))
    finite_left = np.isfinite(left).all(axis=(1, 2))
    finite_right = np.isfinite(right).all(axis=(1, 2))
    finite_gaze = np.isfinite(gaze_origin).all(axis=1) & np.isfinite(gaze_direction).all(axis=1)
    return {
        "timestamps": raw[:, 0].astype(np.int64),
        "head_mats": head.astype(np.float32),
        "left_hand": left.astype(np.float32),
        "right_hand": right.astype(np.float32),
        "left_hand_valid": ((raw[:, left_valid_col] > 0.5) & finite_left).astype(bool),
        "right_hand_valid": ((raw[:, right_valid_col] > 0.5) & finite_right).astype(bool),
        "gaze_origin": gaze_origin.astype(np.float32),
        "gaze_direction": gaze_direction.astype(np.float32),
        "gaze_distance": gaze_distance.astype(np.float32),
        "gaze_valid": ((raw[:, 851] > 0.5) & finite_head & finite_gaze).astype(bool),
        "head_valid": finite_head.astype(bool),
    }


def _head_cpf_rotation_matrix(mode: str) -> np.ndarray:
    if mode == "smpl":
        return np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
    if mode == "identity":
        return np.eye(3, dtype=np.float32)
    raise ValueError(mode)


def _median_offset(x: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
    valid = valid & np.isfinite(x).all(axis=1)
    if np.count_nonzero(valid) == 0:
        return np.zeros(3, dtype=np.float32), 0
    return np.median(x[valid], axis=0).astype(np.float32), int(np.count_nonzero(valid))


def _offset_stats(offset: np.ndarray) -> dict:
    return {
        "translation": offset.astype(float).tolist(),
        "norm": float(np.linalg.norm(offset)),
    }


def _looks_like_eye_offset(offset: np.ndarray) -> bool:
    # CPF/head offsets should be small human-scale translations. Near-zero is
    # also valid when the tracked head/device origin is already at the gaze/CPF
    # origin, which is common for egocentric head-pose exports.
    norm = float(np.linalg.norm(offset))
    return norm <= 0.25


def _estimate_T_head_cpf(
    head_mats: np.ndarray,
    gaze_origin: np.ndarray,
    gaze_valid: np.ndarray,
    source: str = "auto",
) -> tuple[np.ndarray, dict]:
    R = head_mats[:, :3, :3]
    t = head_mats[:, :3, 3]

    # Case 1: EgoBody gaze_origin is expressed in the same Holo/world frame as
    # head_mats. Then head->CPF is R_head^T * (gaze_origin_world - head_t).
    rel_world = gaze_origin - t
    rel_head_from_world = np.einsum("nij,nj->ni", np.swapaxes(R, 1, 2), rel_world)
    offset_world, valid_world = _median_offset(rel_head_from_world, gaze_valid)

    # Case 2: some exports store gaze_origin directly in the head frame. In
    # that convention subtracting head_t would be wrong; the translation is the
    # local gaze origin itself.
    offset_head_local, valid_head_local = _median_offset(gaze_origin, gaze_valid)

    if source == "gaze-world":
        offset = offset_world
        selected = "gaze_world"
        valid_count = valid_world
    elif source == "gaze-head-local":
        offset = offset_head_local
        selected = "gaze_head_local"
        valid_count = valid_head_local
    elif source == "zero":
        offset = np.zeros(3, dtype=np.float32)
        selected = "head_origin_is_cpf"
        valid_count = int(np.count_nonzero(gaze_valid))
    elif source == "auto":
        world_ok = valid_world > 0 and _looks_like_eye_offset(offset_world)
        local_ok = valid_head_local > 0 and _looks_like_eye_offset(offset_head_local)
        # EgoBody's release reader stores gaze origin as a homogeneous point in
        # the same Holo/world frame as head and hand translations. Therefore the
        # world-frame candidate is the default. The head-local candidate is only
        # a fallback for nonstandard exports where the world candidate is invalid
        # or physically too large.
        if world_ok:
            offset = offset_world
            selected = "gaze_world_auto"
            valid_count = valid_world
        elif local_ok:
            offset = offset_head_local
            selected = "gaze_head_local_auto"
            valid_count = valid_head_local
        else:
            offset = np.zeros(3, dtype=np.float32)
            selected = "head_origin_fallback"
            valid_count = 0
    else:
        raise ValueError(f"Unknown cpf offset source: {source}")

    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = offset.astype(np.float32)
    return T, {
        "source": selected,
        "requested_source": source,
        "num_valid_gaze_frames": int(valid_count),
        "translation_head_cpf": offset.astype(float).tolist(),
        "candidate_gaze_world": {**_offset_stats(offset_world), "num_valid": int(valid_world)},
        "candidate_gaze_head_local": {**_offset_stats(offset_head_local), "num_valid": int(valid_head_local)},
    }


def _safe_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def _hand_summary(hand: np.ndarray) -> np.ndarray:
    palm = hand[:, HAND_PALM_INDEX]
    wrist = hand[:, HAND_WRIST_INDEX]
    index_base = hand[:, HAND_INDEX_METACARPAL_INDEX]
    little_base = hand[:, HAND_LITTLE_METACARPAL_INDEX]
    normal = _safe_normalize(np.cross(index_base - palm, little_base - palm))
    return np.concatenate([wrist, palm, normal], axis=-1).astype(np.float32)


def _load_smplx_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_pkl_index(root: Path, split: str, recording: str, role: str) -> dict[int, Path]:
    prefix = "smplx_interactee" if role == "exo" else "smplx_camera_wearer"
    base = root / f"{prefix}_{split}" / recording
    index = {}
    for p in base.glob("*/results/frame_*/000.pkl"):
        index[int(p.parent.name.replace("frame_", ""))] = p
    return index


def _stack_smpl_params(root: Path, split: str, recording: str, role: str, frame_ids: np.ndarray) -> tuple[dict, torch.Tensor, list[str]]:
    index = _build_pkl_index(root, split, recording, role)
    arrays = {k: [] for k in SMPLX_KEYS}
    valid = []
    paths = []
    for fid in frame_ids:
        p = index.get(int(fid))
        if p is None:
            valid.append(False)
            paths.append("")
            for k in SMPLX_KEYS:
                dim = 10 if k == "betas" else 3 if k in ("global_orient", "transl", "jaw_pose", "leye_pose", "reye_pose") else 63 if k == "body_pose" else 45 if "hand_pose" in k else 10
                arrays[k].append(np.zeros((dim,), dtype=np.float32))
            continue
        data = _load_smplx_pkl(p)
        valid.append(True)
        paths.append(str(p))
        for k in SMPLX_KEYS:
            if k in data:
                arrays[k].append(np.asarray(data[k]).reshape(-1).astype(np.float32))
            else:
                dim = 10 if k == "betas" else 3 if k in ("global_orient", "transl", "jaw_pose", "leye_pose", "reye_pose") else 63 if k == "body_pose" else 45 if "hand_pose" in k else 10
                arrays[k].append(np.zeros((dim,), dtype=np.float32))
    params = {k: torch.tensor(np.stack(v), dtype=torch.float32) for k, v in arrays.items()}
    params["betas"] = params["betas"][:, :10]
    params["body_pose"] = params["body_pose"][:, :63]
    return params, torch.tensor(valid, dtype=torch.bool), paths


def _transform_smpl(params: dict[str, torch.Tensor], T_c2w: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in params.items()}
    out["global_orient"], out["transl"] = transform_smpl_root(
        params["global_orient"], params["transl"], T_c2w, params.get("betas")
    )
    return out


def _relative_rot6d(T_world_cam: torch.Tensor) -> torch.Tensor:
    R = T_world_cam[:, :3, :3]
    rel = torch.eye(3, dtype=torch.float32).repeat(len(R), 1, 1)
    if len(R) > 1:
        rel[1:] = R[:-1].mT @ R[1:]
    return matrix_to_rotation_6d(rel)


def _smpl_lite_inputs(params: dict[str, torch.Tensor], sl: slice, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "betas": params["betas"][sl].to(device),
        "global_orient": params["global_orient"][sl].to(device),
        "transl": params["transl"][sl].to(device),
        "body_pose": params["body_pose"][sl].to(device),
    }


def _compute_joints_verts_and_bbx(params_kinect: dict[str, torch.Tensor], K: torch.Tensor, batch_size: int, device: torch.device):
    smplx_lite = make_smplx("supermotion_v437coco17").to(device).eval()
    verts_all, joints_all, bbx_all = [], [], []
    L = params_kinect["body_pose"].shape[0]
    K_batched = K.to(device).unsqueeze(0)
    for start in tqdm(range(0, L, batch_size), desc="SMPLX bbox", leave=False):
        end = min(start + batch_size, L)
        with torch.no_grad():
            verts, joints = smplx_lite(**_smpl_lite_inputs(params_kinect, slice(start, end), device))
            i_x2d = safely_render_x3d_K(verts[:, None], K_batched.repeat(end - start, 1, 1)[:, None], thr=0.3)
            bbx = get_bbx_xys(i_x2d, do_augment=False).squeeze(1)
        verts_all.append(verts.cpu())
        joints_all.append(joints.cpu())
        bbx_all.append(bbx.cpu())
    return torch.cat(verts_all), torch.cat(joints_all), torch.cat(bbx_all)


def _compute_ego_pv_bbx(
    verts_kinect: torch.Tensor,
    joints_kinect: torch.Tensor,
    T_kinect_to_holo: torch.Tensor,
    T_world_pv: torch.Tensor,
    K_ego: torch.Tensor,
    batch_size: int,
    device: torch.device,
):
    """Project Kinect12 SMPL-X geometry into the per-frame ego PV camera.

    EgoBody PV txt stores a per-frame PV-to-Holo/world pose. Following the
    EgoAllo reprojection utilities, world-to-camera is R^T * (x - t), followed
    by the HoloLens/OpenCV camera convention conversion diag(1, -1, -1).
    """
    bbx_all, kp2d_all = [], []
    L = verts_kinect.shape[0]
    T_kinect_to_holo = T_kinect_to_holo.to(device)
    cam_conv = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=torch.float32))
    for start in tqdm(range(0, L, batch_size), desc="Ego PV bbox", leave=False):
        end = min(start + batch_size, L)
        verts = verts_kinect[start:end].to(device)
        joints = joints_kinect[start:end].to(device)

        R_k2h = T_kinect_to_holo[:3, :3]
        t_k2h = T_kinect_to_holo[:3, 3]
        verts_holo = torch.einsum("ij,bvj->bvi", R_k2h, verts) + t_k2h[None, None]
        joints_holo = torch.einsum("ij,bvj->bvi", R_k2h, joints) + t_k2h[None, None]

        T = T_world_pv[start:end].to(device).float()
        R_world_pv = T[:, :3, :3]
        t_world_pv = T[:, :3, 3]
        verts_pv = torch.einsum("bvj,bjk->bvk", verts_holo - t_world_pv[:, None], R_world_pv)
        joints_pv = torch.einsum("bvj,bjk->bvk", joints_holo - t_world_pv[:, None], R_world_pv)
        verts_cam = torch.einsum("bvj,kj->bvk", verts_pv, cam_conv)
        joints_cam = torch.einsum("bvj,kj->bvk", joints_pv, cam_conv)

        K = K_ego[start:end].to(device)
        with torch.no_grad():
            i_x2d = safely_render_x3d_K(verts_cam[:, None], K[:, None], thr=0.3)
            bbx = get_bbx_xys(i_x2d, do_augment=False).squeeze(1)
            kp2d = _project_points_batched(joints_cam, K)
        bbx_all.append(bbx.cpu())
        kp2d_all.append(kp2d.cpu())
    return torch.cat(bbx_all), torch.cat(kp2d_all)


def _project_points_batched(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    z = points[..., 2].clamp(min=1e-4)
    x = points[..., 0] * K[:, None, 0, 0] / z + K[:, None, 0, 2]
    y = points[..., 1] * K[:, None, 1, 1] / z + K[:, None, 1, 2]
    conf = (points[..., 2] > 0.1).float()
    return torch.stack([x, y, conf], dim=-1)


def _project_points(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    z = points[..., 2].clamp(min=1e-4)
    x = points[..., 0] * K[0, 0] / z + K[0, 2]
    y = points[..., 1] * K[1, 1] / z + K[1, 2]
    conf = (points[..., 2] > 0.1).float()
    return torch.stack([x, y, conf], dim=-1)


def _extract_features_for_paths(extractor: Extractor, image_paths: list[Path], bbx_xys: torch.Tensor, chunk: int, desc: str) -> torch.Tensor:
    feats = []
    for start in tqdm(range(0, len(image_paths), chunk), desc=desc):
        end = min(start + chunk, len(image_paths))
        imgs = []
        for p in image_paths[start:end]:
            im = cv2.imread(str(p))
            if im is None:
                raise FileNotFoundError(p)
            imgs.append(im[..., ::-1])
        imgs_np = np.stack(imgs, axis=0)
        imgs_t, bbx_ds = get_batch(imgs_np, bbx_xys[start:end].float(), img_ds=1.0, path_type="np")
        feats.append(extractor.extract_video_features(imgs_t, bbx_ds))
    return torch.cat(feats, dim=0)



def _update_cpf_only(args: argparse.Namespace) -> Path:
    """Refresh CPF-related trajectory fields without recomputing SMPL boxes/features."""
    root = args.root
    recording = args.recording
    out_dir = args.output_root / recording
    manifest_path = out_dir / "manifest.pt"
    traj_path = out_dir / "camera_head_traj.pt"
    metadata_path = out_dir / "metadata.json"
    if not manifest_path.exists() or not traj_path.exists():
        raise FileNotFoundError(f"Need existing manifest.pt and camera_head_traj.pt under {out_dir}")

    manifest = torch.load(manifest_path, map_location="cpu")
    traj = torch.load(traj_path, map_location="cpu")
    query_ts = manifest["query_timestamps"].cpu().numpy().astype(np.int64)

    head_csv = _find_head_hand_eye_csv(root, recording)
    head_data = _read_head_hand_eye(head_csv)
    head_idx, matched_head_ts, head_diff = _nearest_indices(query_ts, head_data["timestamps"])
    T_holo_head_np = head_data["head_mats"][head_idx]
    gaze_origin = head_data["gaze_origin"][head_idx]
    gaze_valid = head_data["gaze_valid"][head_idx]
    T_head_cpf, cpf_stats = _estimate_T_head_cpf(T_holo_head_np, gaze_origin, gaze_valid, args.cpf_offset_source)
    T_head_cpf[:3, :3] = _head_cpf_rotation_matrix(args.head_cpf_rotation)

    T_holo_head = torch.tensor(T_holo_head_np, dtype=torch.float32)
    T_head_cpf_t = torch.tensor(T_head_cpf, dtype=torch.float32)
    T_holo_cpf = T_holo_head @ T_head_cpf_t.unsqueeze(0)

    traj["T_holo_head"] = T_holo_head
    traj["T_holo_cpf"] = T_holo_cpf
    traj["T_head_cpf"] = T_head_cpf_t
    traj["head_angvel"] = _relative_rot6d(T_holo_head)
    traj["head_valid"] = torch.tensor(head_data["head_valid"][head_idx], dtype=torch.bool)
    traj["gaze_origin_holo"] = torch.tensor(gaze_origin, dtype=torch.float32)
    traj["gaze_valid"] = torch.tensor(gaze_valid, dtype=torch.bool)
    traj["matched_head_timestamps"] = torch.tensor(matched_head_ts, dtype=torch.long)
    traj["head_timestamp_diff_ticks"] = torch.tensor(head_diff, dtype=torch.long)
    traj["cpf_stats"] = cpf_stats
    _save(traj_path, traj)

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {"recording": recording}
    metadata["cpf_stats"] = cpf_stats
    metadata["max_head_time_diff_ticks"] = int(head_diff.max())
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(json.dumps({"recording": recording, "updated": str(traj_path), "cpf_stats": cpf_stats}, indent=2))
    return out_dir


def _append_ego_body_features(args: argparse.Namespace) -> Path:
    root = args.root
    recording = args.recording
    out_dir = args.output_root / recording
    body_path = out_dir / "features_ego_body_hmr2.pt"
    manifest_path = out_dir / "manifest.pt"
    if args.resume and body_path.exists():
        print(f"[Skip] {recording}: existing ego body features found at {body_path}")
        return out_dir
    if not manifest_path.exists() or not (out_dir / "smplx_gt.pt").exists() or not (out_dir / "camera_head_traj.pt").exists() or not (out_dir / "calibration.pt").exists():
        raise FileNotFoundError(f"Missing base preprocessing outputs under {out_dir}")

    manifest = torch.load(manifest_path, map_location="cpu")
    smplx_gt = torch.load(out_dir / "smplx_gt.pt", map_location="cpu")
    traj = torch.load(out_dir / "camera_head_traj.pt", map_location="cpu")
    calib = torch.load(out_dir / "calibration.pt", map_location="cpu")
    pv_paths = [Path(p) for p in manifest["pv_img_paths"]]
    K_ego = manifest["K_ego"]
    T_holo_to_pv = traj["T_holo_pv"]
    T_kinect_to_holo = calib["T_kinect12_to_holo"]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_smpl else "cpu")
    smplx_lite = make_smplx("supermotion_v437coco17").to(device).eval()
    verts_all, joints_all = [], []
    L = smplx_gt["exo"]["kinect12"]["body_pose"].shape[0]
    for start in tqdm(range(0, L, args.smpl_batch_size), desc="Interactee SMPLX for ego PV bbox", leave=False):
        end = min(start + args.smpl_batch_size, L)
        with torch.no_grad():
            verts, joints = smplx_lite(**_smpl_lite_inputs(smplx_gt["exo"]["kinect12"], slice(start, end), device))
        verts_all.append(verts.cpu())
        joints_all.append(joints.cpu())
    exo_verts437 = torch.cat(verts_all)
    exo_joints17 = torch.cat(joints_all)
    # In ego/PV images, the visible person we crop is the interactee/exo,
    # not the camera wearer. Project exo SMPL-X GT into the ego camera.
    ego_body_full_bbx, ego_body_kp2d = _compute_ego_pv_bbx(
        exo_verts437, exo_joints17, T_kinect_to_holo, T_holo_to_pv, K_ego, args.smpl_batch_size, device
    )
    if "ego_image_size" in manifest:
        width, height = [float(x) for x in manifest["ego_image_size"]]
    else:
        full = manifest["bbx_ego_full"][0]
        width, height = float(full[0] * 2), float(full[1] * 2)
    ego_body_bbx, ego_body_bbox_valid = _visible_keypoint_bbx(ego_body_kp2d, width, height)
    ego_body_bbox_valid = ego_body_bbox_valid & _bbox_overlap_mask(ego_body_bbx, width, height)
    extractor = Extractor(tqdm_leave=False)
    ego_body_features = _extract_features_for_paths(extractor, pv_paths, ego_body_bbx, args.feature_chunk, "ego body HMR2 features")
    _save(body_path, {"features": ego_body_features, "bbx_xys": ego_body_bbx, "image_paths": [str(p) for p in pv_paths]})
    _save_ego_body_bbox_debug(out_dir, pv_paths, ego_body_bbx, ego_body_bbox_valid, manifest["frame_ids"])
    manifest["bbx_ego_body_gt"] = ego_body_bbx
    manifest["bbx_ego_body_full_gt"] = ego_body_full_bbx
    manifest["kp2d_ego_body_gt"] = ego_body_kp2d
    manifest.setdefault("mask", {})["features_ego_body"] = torch.ones(L, dtype=torch.bool)
    manifest.setdefault("mask", {})["ego_body_bbox"] = ego_body_bbox_valid
    manifest["ego_image_size"] = torch.tensor([width, height], dtype=torch.float32)
    manifest.setdefault("files", {})["features_ego_body"] = "features_ego_body_hmr2.pt"
    _save(manifest_path, manifest)
    print(f"Saved {body_path}")
    return out_dir


def _bbox_overlap_mask(bbx_xys: torch.Tensor, width: float, height: float, max_scale: float = 4.0) -> torch.Tensor:
    cx, cy, size = bbx_xys[:, 0], bbx_xys[:, 1], bbx_xys[:, 2]
    x1, y1 = cx - size / 2, cy - size / 2
    x2, y2 = cx + size / 2, cy + size / 2
    ix1 = torch.clamp(x1, 0, width)
    iy1 = torch.clamp(y1, 0, height)
    ix2 = torch.clamp(x2, 0, width)
    iy2 = torch.clamp(y2, 0, height)
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    image_area = float(width * height)
    max_side = max(float(width), float(height))
    return torch.isfinite(bbx_xys).all(dim=-1) & (size > 8.0) & (size < max_side * max_scale) & (inter > image_area * 0.0025)


def _visible_keypoint_bbx(
    kp2d: torch.Tensor,
    width: float,
    height: float,
    enlarge: float = 1.25,
    min_size: float = 32.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a crop box from projected visible keypoints clipped to the image.

    This is intended for ego/PV interactee crops where the full body often goes
    outside the first-person image. The resulting box describes the visible body
    evidence plus padding, not the full off-screen SMPL mesh extent.
    """
    boxes = []
    valid = []
    W = float(width)
    H = float(height)
    for frame_kp in kp2d:
        mask = frame_kp[:, 2] > 0.5
        pts = frame_kp[mask, :2]
        pts = pts[torch.isfinite(pts).all(dim=-1)]
        if pts.numel() == 0:
            boxes.append(torch.tensor([W / 2.0, H / 2.0, max(W, H)], dtype=kp2d.dtype))
            valid.append(False)
            continue
        pts = pts.clone()
        pts[:, 0].clamp_(0.0, W - 1.0)
        pts[:, 1].clamp_(0.0, H - 1.0)
        mn = pts.min(dim=0).values
        mx = pts.max(dim=0).values
        center = (mn + mx) * 0.5
        size = torch.clamp((mx - mn).max() * enlarge, min=min_size)
        boxes.append(torch.stack([center[0], center[1], size]).to(dtype=kp2d.dtype))
        valid.append(bool(size > min_size * 0.5))
    return torch.stack(boxes, dim=0), torch.tensor(valid, dtype=torch.bool)


def _make_full_bbx(pv_meta: dict, length: int) -> torch.Tensor:
    size = max(float(pv_meta["w"]), float(pv_meta["h"]))
    return torch.tensor([[pv_meta["cx"], pv_meta["cy"], size]], dtype=torch.float32).repeat(length, 1)


def _draw_bbx(img: np.ndarray, bbx: torch.Tensor, color: tuple[int, int, int], label: str) -> None:
    cx, cy, s = [float(x) for x in bbx]
    p1 = (int(cx - s / 2), int(cy - s / 2))
    p2 = (int(cx + s / 2), int(cy + s / 2))
    cv2.rectangle(img, p1, p2, color, 2)
    cv2.putText(img, label, (p1[0], max(20, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _save_ego_body_bbox_debug(out_dir: Path, pv_paths: list[Path], ego_body_bbx: torch.Tensor, ego_body_valid: torch.Tensor, frame_ids: torch.Tensor | np.ndarray) -> None:
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    if len(pv_paths) == 0:
        return
    picks = sorted(set([0, len(pv_paths) // 2, len(pv_paths) - 1]))
    panels = []
    for idx in picks:
        img = cv2.imread(str(pv_paths[idx]))
        if img is None:
            continue
        color = (0, 255, 0) if bool(ego_body_valid[idx]) else (0, 0, 255)
        _draw_bbx(img, ego_body_bbx[idx], color, "interactee GT bbox in ego PV")
        fid = int(frame_ids[idx]) if len(frame_ids) > idx else idx
        cv2.putText(img, f"frame {fid} valid={bool(ego_body_valid[idx])}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imwrite(str(debug_dir / f"ego_pv_interactee_bbox_frame_{fid:05d}.jpg"), img)
        panels.append(img)
    if panels:
        h = min(im.shape[0] for im in panels)
        resized = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in panels]
        cv2.imwrite(str(debug_dir / "ego_pv_interactee_bbox_strip.jpg"), cv2.hconcat(resized))


def _save_debug_visualization(out_dir: Path, kinect_paths: list[Path], exo_bbx: torch.Tensor, ego_bbx: torch.Tensor, smplx_gt: dict, traj: dict, frame_ids: np.ndarray) -> None:
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    mid = len(kinect_paths) // 2
    img = cv2.imread(str(kinect_paths[mid]))
    if img is not None:
        _draw_bbx(img, exo_bbx[mid], (0, 255, 0), "exo GT bbox")
        _draw_bbx(img, ego_bbx[mid], (255, 0, 0), "ego GT bbox")
        cv2.imwrite(str(debug_dir / f"bbox_frame_{int(frame_ids[mid]):05d}.jpg"), img)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    exo_t = smplx_gt["exo"]["holo"]["transl"].numpy()
    ego_t = smplx_gt["ego"]["holo"]["transl"].numpy()
    head_t = traj["T_holo_head"][:, :3, 3].numpy()
    ax.plot(exo_t[:, 0], exo_t[:, 1], exo_t[:, 2], label="exo root", color="green")
    ax.plot(ego_t[:, 0], ego_t[:, 1], ego_t[:, 2], label="ego root", color="blue")
    ax.plot(head_t[:, 0], head_t[:, 1], head_t[:, 2], label="tracked head", color="red")
    ax.scatter(exo_t[0, 0], exo_t[0, 1], exo_t[0, 2], color="green", marker="o")
    ax.scatter(ego_t[0, 0], ego_t[0, 1], ego_t[0, 2], color="blue", marker="o")
    ax.set_title("Holo world trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    all_pts = np.concatenate([exo_t, ego_t, head_t], axis=0)
    center = all_pts.mean(axis=0)
    radius = np.linalg.norm(all_pts - center, axis=1).max()
    for axis, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        axis(c - radius, c + radius)
    fig.tight_layout()
    fig.savefig(debug_dir / "holo_world_trajectories.png", dpi=160)
    plt.close(fig)


def preprocess_recording(args: argparse.Namespace) -> Path:
    root = args.root
    recording = args.recording
    split = args.split if args.split != "auto" else _get_split(root, recording)
    out_dir = args.output_root / recording
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (out_dir / "manifest.pt").exists() and (out_dir / "smplx_gt.pt").exists() and (out_dir / "camera_head_traj.pt").exists():
        feature_ready = args.skip_features or (
            (out_dir / "features_exo_hmr2.pt").exists()
            and (out_dir / "features_ego_full_hmr2.pt").exists()
            and (out_dir / "features_ego_body_hmr2.pt").exists()
        )
        if feature_ready:
            print(f"[Skip] {recording}: existing preprocessing found at {out_dir}")
            return out_dir

    pv_dir = _find_pv_dir(root, recording)
    pv_txt = _find_pv_txt(root, recording)
    kinect_dir = root / "kinect_color" / recording / "master"
    if not kinect_dir.exists():
        raise FileNotFoundError(kinect_dir)

    pv_images_all = _collect_images(pv_dir)
    kinect_images_all = _collect_images(kinect_dir)
    pv_by_frame = {}
    pv_ts_by_frame = {}
    for p in pv_images_all:
        ts, fid = _parse_pv_image(p)
        pv_by_frame[fid] = p
        pv_ts_by_frame[fid] = ts
    kinect_by_frame = {_parse_kinect_image(p): p for p in kinect_images_all}

    frame_ids = np.asarray(sorted(set(pv_by_frame) & set(kinect_by_frame)), dtype=np.int64)
    frame_range = _get_frame_range(root, recording)
    if frame_range is not None:
        lo, hi = frame_range
        frame_ids = frame_ids[(frame_ids >= lo) & (frame_ids <= hi)]
    if args.start_frame_id is not None:
        frame_ids = frame_ids[frame_ids >= args.start_frame_id]
    if args.end_frame_id is not None:
        frame_ids = frame_ids[frame_ids <= args.end_frame_id]
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]
    if len(frame_ids) == 0:
        raise RuntimeError(f"No synchronized frames for {recording}")

    pv_paths = [pv_by_frame[int(fid)] for fid in frame_ids]
    kinect_paths = [kinect_by_frame[int(fid)] for fid in frame_ids]
    query_ts = np.asarray([pv_ts_by_frame[int(fid)] for fid in frame_ids], dtype=np.int64)

    K_exo = _load_exo_intrinsics(root, args.exo_intrinsics_json)
    T_holo_to_kinect = _load_holo_to_kinect12(root, recording)
    T_kinect_to_holo = torch.linalg.inv(T_holo_to_kinect)
    T_kinect_to_scene = _load_kinect12_to_scene(root, recording)
    T_holo_to_scene = T_kinect_to_scene @ T_holo_to_kinect

    pv_meta, pv_frames = _read_pv_txt(pv_txt)
    T_holo_pv_np, K_ego_np, matched_pv_ts, pv_diff = _nearest_pv(query_ts, pv_meta, pv_frames)
    T_holo_pv = torch.tensor(T_holo_pv_np, dtype=torch.float32)
    T_pv_holo = torch.linalg.inv(T_holo_pv)
    K_ego = torch.tensor(K_ego_np, dtype=torch.float32)

    head_csv = _find_head_hand_eye_csv(root, recording)
    head_data = _read_head_hand_eye(head_csv)
    head_idx, matched_head_ts, head_diff = _nearest_indices(query_ts, head_data["timestamps"])
    T_holo_head_np = head_data["head_mats"][head_idx]
    gaze_origin = head_data["gaze_origin"][head_idx]
    gaze_direction = head_data["gaze_direction"][head_idx]
    gaze_distance = head_data["gaze_distance"][head_idx]
    gaze_valid = head_data["gaze_valid"][head_idx]
    T_head_cpf, cpf_stats = _estimate_T_head_cpf(T_holo_head_np, gaze_origin, gaze_valid, args.cpf_offset_source)
    T_head_cpf[:3, :3] = _head_cpf_rotation_matrix(args.head_cpf_rotation)
    T_holo_head = torch.tensor(T_holo_head_np, dtype=torch.float32)
    T_head_cpf_t = torch.tensor(T_head_cpf, dtype=torch.float32)
    T_holo_cpf = T_holo_head @ T_head_cpf_t.unsqueeze(0)

    left_hand = torch.tensor(head_data["left_hand"][head_idx], dtype=torch.float32)
    right_hand = torch.tensor(head_data["right_hand"][head_idx], dtype=torch.float32)
    camera_head_traj = {
        "T_holo_head": T_holo_head,
        "T_holo_cpf": T_holo_cpf,
        "T_holo_pv": T_holo_pv,
        "T_pv_holo": T_pv_holo,
        "T_head_cpf": T_head_cpf_t,
        "head_angvel": _relative_rot6d(T_holo_head),
        "pv_cam_angvel": compute_cam_angvel(T_pv_holo[:, :3, :3]),
        "gaze_origin_holo": torch.tensor(gaze_origin, dtype=torch.float32),
        "gaze_direction_holo": torch.tensor(gaze_direction, dtype=torch.float32),
        "gaze_distance": torch.tensor(gaze_distance, dtype=torch.float32),
        "gaze_valid": torch.tensor(gaze_valid, dtype=torch.bool),
        "left_hand_joints_holo": left_hand,
        "right_hand_joints_holo": right_hand,
        "left_hand_summary_holo": torch.tensor(_hand_summary(left_hand.numpy()), dtype=torch.float32),
        "right_hand_summary_holo": torch.tensor(_hand_summary(right_hand.numpy()), dtype=torch.float32),
        "left_hand_valid": torch.tensor(head_data["left_hand_valid"][head_idx], dtype=torch.bool),
        "right_hand_valid": torch.tensor(head_data["right_hand_valid"][head_idx], dtype=torch.bool),
        "head_valid": torch.tensor(head_data["head_valid"][head_idx], dtype=torch.bool),
        "matched_head_timestamps": torch.tensor(matched_head_ts, dtype=torch.long),
        "matched_pv_timestamps": torch.tensor(matched_pv_ts, dtype=torch.long),
        "head_timestamp_diff_ticks": torch.tensor(head_diff, dtype=torch.long),
        "pv_timestamp_diff_ticks": torch.tensor(pv_diff, dtype=torch.long),
        "cpf_stats": cpf_stats,
    }

    exo_kinect, exo_valid, exo_pkl_paths = _stack_smpl_params(root, split, recording, "exo", frame_ids)
    ego_kinect, ego_valid, ego_pkl_paths = _stack_smpl_params(root, split, recording, "ego", frame_ids)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_smpl else "cpu")
    exo_verts437, exo_joints17, exo_bbx = _compute_joints_verts_and_bbx(exo_kinect, K_exo, args.smpl_batch_size, device)
    ego_verts437, ego_joints17, ego_bbx = _compute_joints_verts_and_bbx(ego_kinect, K_exo, args.smpl_batch_size, device)
    exo_kp2d_gt = _project_points(exo_joints17, K_exo)
    ego_kp2d_gt = _project_points(ego_joints17, K_exo)
    # In ego/PV images, the visible person we crop is the interactee/exo,
    # not the camera wearer. Project exo SMPL-X GT into the ego camera.
    ego_body_full_bbx, ego_body_kp2d = _compute_ego_pv_bbx(
        exo_verts437, exo_joints17, T_kinect_to_holo, T_holo_pv, K_ego, args.smpl_batch_size, device
    )
    ego_body_bbx, ego_body_bbox_valid = _visible_keypoint_bbx(ego_body_kp2d, pv_meta["w"], pv_meta["h"])
    ego_body_bbox_valid = ego_body_bbox_valid & _bbox_overlap_mask(ego_body_bbx, pv_meta["w"], pv_meta["h"])
    valid = exo_valid & ego_valid & torch.isfinite(exo_bbx).all(dim=-1) & torch.isfinite(ego_bbx).all(dim=-1)
    valid = valid & (exo_bbx[:, 2] > 1.0) & (ego_bbx[:, 2] > 1.0)

    manifest = {
        "schema_version": "egoexo_v1",
        "recording": recording,
        "split": split,
        "length": int(len(frame_ids)),
        "frame_ids": torch.tensor(frame_ids, dtype=torch.long),
        "query_timestamps": torch.tensor(query_ts, dtype=torch.long),
        "pv_img_paths": [str(p) for p in pv_paths],
        "kinect_img_paths": [str(p) for p in kinect_paths],
        "exo_pkl_paths": exo_pkl_paths,
        "ego_pkl_paths": ego_pkl_paths,
        "K_exo": K_exo.unsqueeze(0).repeat(len(frame_ids), 1, 1),
        "K_ego": K_ego,
        "ego_image_size": torch.tensor([float(pv_meta["w"]), float(pv_meta["h"])], dtype=torch.float32),
        "bbx_exo_gt": exo_bbx,
        "bbx_ego_in_exo_gt": ego_bbx,
        "bbx_ego_full": _make_full_bbx(pv_meta, len(frame_ids)),
        "bbx_ego_body_gt": ego_body_bbx,
        "bbx_ego_body_full_gt": ego_body_full_bbx,
        "kp2d_exo_gt": exo_kp2d_gt,
        "kp2d_ego_in_exo_gt": ego_kp2d_gt,
        "kp2d_ego_body_gt": ego_body_kp2d,
        "mask": {
            "valid": valid,
            "exo_gt": exo_valid,
            "ego_gt": ego_valid,
            "head": camera_head_traj["head_valid"],
            "gaze": camera_head_traj["gaze_valid"],
            "left_hand": camera_head_traj["left_hand_valid"],
            "right_hand": camera_head_traj["right_hand_valid"],
            "features_exo": torch.zeros(len(frame_ids), dtype=torch.bool),
            "features_ego_full": torch.zeros(len(frame_ids), dtype=torch.bool),
            "features_ego_body": torch.zeros(len(frame_ids), dtype=torch.bool),
            "ego_body_bbox": ego_body_bbox_valid,
        },
        "files": {
            "smplx_gt": "smplx_gt.pt",
            "camera_head_traj": "camera_head_traj.pt",
            "features_exo": "features_exo_hmr2.pt",
            "features_ego_full": "features_ego_full_hmr2.pt",
            "features_ego_body": "features_ego_body_hmr2.pt",
        },
        "sources": {
            "pv_txt": str(pv_txt),
            "head_hand_eye_csv": str(head_csv),
            "exo_intrinsics_json": str(root / args.exo_intrinsics_json),
        },
        "notes": "Training bbox/keypoints are GT-derived. bbx_ego_body_gt is the interactee/exo projected into the ego PV camera. Detector/VitPose are intentionally not used here.",
    }
    smplx_gt = {
        "exo": {"kinect12": exo_kinect, "valid": exo_valid, "pkl_paths": exo_pkl_paths},
        "ego": {"kinect12": ego_kinect, "valid": ego_valid, "pkl_paths": ego_pkl_paths},
    }
    calib = {
        "T_holo_to_kinect12": T_holo_to_kinect,
        "T_kinect12_to_holo": T_kinect_to_holo,
        "T_kinect12_to_scene": T_kinect_to_scene,
        "T_holo_to_scene": T_holo_to_scene,
    }

    _save(out_dir / "manifest.pt", manifest)
    _save(out_dir / "smplx_gt.pt", smplx_gt)
    _save(out_dir / "camera_head_traj.pt", camera_head_traj)
    _save(out_dir / "calibration.pt", calib)

    if not args.skip_features:
        extractor = Extractor(tqdm_leave=False)
        exo_features = _extract_features_for_paths(extractor, kinect_paths, exo_bbx, args.feature_chunk, "exo HMR2 features")
        _save(out_dir / "features_exo_hmr2.pt", {"features": exo_features, "bbx_xys": exo_bbx, "image_paths": [str(p) for p in kinect_paths]})
        manifest["mask"]["features_exo"] = torch.ones(len(frame_ids), dtype=torch.bool)
        ego_full_features = _extract_features_for_paths(extractor, pv_paths, manifest["bbx_ego_full"], args.feature_chunk, "ego full HMR2 features")
        _save(out_dir / "features_ego_full_hmr2.pt", {"features": ego_full_features, "bbx_xys": manifest["bbx_ego_full"], "image_paths": [str(p) for p in pv_paths]})
        manifest["mask"]["features_ego_full"] = torch.ones(len(frame_ids), dtype=torch.bool)
        ego_body_features = _extract_features_for_paths(extractor, pv_paths, ego_body_bbx, args.feature_chunk, "ego body HMR2 features")
        _save(out_dir / "features_ego_body_hmr2.pt", {"features": ego_body_features, "bbx_xys": ego_body_bbx, "image_paths": [str(p) for p in pv_paths]})
        manifest["mask"]["features_ego_body"] = torch.ones(len(frame_ids), dtype=torch.bool)
        _save(out_dir / "manifest.pt", manifest)

    metadata = {
        "recording": recording,
        "split": split,
        "num_frames": int(len(frame_ids)),
        "first_frame_id": int(frame_ids[0]),
        "last_frame_id": int(frame_ids[-1]),
        "valid_frames": int(valid.sum()),
        "head_valid_frames": int(camera_head_traj["head_valid"].sum()),
        "left_hand_valid_frames": int(camera_head_traj["left_hand_valid"].sum()),
        "right_hand_valid_frames": int(camera_head_traj["right_hand_valid"].sum()),
        "max_head_time_diff_ticks": int(head_diff.max()),
        "max_pv_time_diff_ticks": int(pv_diff.max()),
        "cpf_stats": cpf_stats,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if not args.skip_debug:
        _save_ego_body_bbox_debug(out_dir, pv_paths, ego_body_bbx, ego_body_bbox_valid, torch.tensor(frame_ids, dtype=torch.long))
        # Full SMPL-X geometry is intentionally not saved. Generate it only for this
        # one debug view, then drop it to keep preprocessing outputs compact.
        debug_smplx_gt = {
            "exo": {
                **smplx_gt["exo"],
                "holo": {"transl": _transform_smpl(exo_kinect, T_kinect_to_holo)["transl"]},
                "joints17_kinect": exo_joints17,
                "verts437_kinect": exo_verts437,
            },
            "ego": {
                **smplx_gt["ego"],
                "holo": {"transl": _transform_smpl(ego_kinect, T_kinect_to_holo)["transl"]},
                "joints17_kinect": ego_joints17,
                "verts437_kinect": ego_verts437,
            },
        }
        _save_debug_visualization(out_dir, kinect_paths, exo_bbx, ego_bbx, debug_smplx_gt, camera_head_traj, frame_ids)
    print(json.dumps(metadata, indent=2))
    print(f"Saved {out_dir}")
    return out_dir



def _recordings_from_csv(root: Path, split_filter: str) -> list[str]:
    df = pd.read_csv(root / "data_splits.csv")
    splits = ["train", "val", "test"] if split_filter == "all" else [split_filter]
    recordings = []
    for split in splits:
        if split not in df.columns:
            raise ValueError(f"split {split} not found in data_splits.csv")
        recordings.extend([str(x) for x in df[split].dropna().tolist()])
    return recordings


def _recordings_from_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/public/home/wenxin/egobody"))
    p.add_argument("--output-root", type=Path, default=Path("/public/home/wenxin/egobody/output/egoexo_v1"))
    p.add_argument("--recording", default=None, help="Single recording to process")
    p.add_argument("--all", action="store_true", help="Process recordings from data_splits.csv")
    p.add_argument("--recordings-file", type=Path, default=None, help="Optional newline-separated recording list")
    p.add_argument("--split", choices=["auto", "train", "val", "test"], default="auto")
    p.add_argument("--split-filter", choices=["all", "train", "val", "test"], default="all", help="Used with --all")
    p.add_argument("--exo-intrinsics-json", default="kinect_cam_params/kinect_master/Color.json")
    p.add_argument("--head-cpf-rotation", choices=["smpl", "identity"], default="smpl")
    p.add_argument("--cpf-offset-source", choices=["auto", "gaze-world", "gaze-head-local", "zero"], default="auto", help="How to derive T_head_cpf translation from gaze/head data")
    p.add_argument("--start-frame-id", type=int, default=None)
    p.add_argument("--end-frame-id", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--skip-features", action="store_true")
    p.add_argument("--only-missing-ego-body-features", action="store_true", help="Append features_ego_body_hmr2.pt to existing preprocessing outputs")
    p.add_argument("--update-cpf-only", action="store_true", help="Only refresh camera_head_traj.pt CPF fields and metadata; do not recompute boxes/features")
    p.add_argument("--skip-debug", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip recordings whose expected outputs already exist")
    p.add_argument("--feature-chunk", type=int, default=256)
    p.add_argument("--smpl-batch-size", type=int, default=128)
    p.add_argument("--cpu-smpl", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.recordings_file is not None:
        recordings = _recordings_from_file(args.recordings_file)
    elif args.all:
        recordings = _recordings_from_csv(args.root, args.split_filter)
    elif args.recording is not None:
        recordings = [args.recording]
    else:
        raise SystemExit("Provide --recording, --recordings-file, or --all")

    failures = []
    for recording in tqdm(recordings, desc="EgoBody recordings"):
        args.recording = recording
        try:
            if args.update_cpf_only:
                _update_cpf_only(args)
            elif args.only_missing_ego_body_features:
                _append_ego_body_features(args)
            else:
                preprocess_recording(args)
        except Exception as exc:
            failures.append((recording, repr(exc)))
            print(f"[Error] {recording}: {exc}")
            if len(recordings) == 1:
                raise
    if failures:
        print("Failed recordings:")
        for recording, error in failures:
            print(f"  {recording}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
