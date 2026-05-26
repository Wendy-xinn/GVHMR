import argparse
from pathlib import Path
import json
import torch
import numpy as np
import cv2
import pandas as pd
import pickle

from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import get_video_lwh, get_writer
from hmr4d.utils.preproc import Tracker, VitPoseExtractor, Extractor
from hmr4d.utils.geo.hmr_cam import estimate_K, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.preproc.vitfeat_extractor import get_batch
from hmr4d.utils.smplx_utils import make_smplx


def parse_args():
    parser = argparse.ArgumentParser(description="EgoBody offline preprocessing (image sequences)")
    parser.add_argument("--root", type=str, required=True, help="EgoBody root directory")
    parser.add_argument("--output_root", type=str, required=True, help="Output directory")

    parser.add_argument("--input_type", type=str, default="images", choices=["video", "images"], help="Input type")
    # parser.add_argument("--view1_glob", type=str, default="**/egocentric_color/**/PV", help="Glob for view1 image dirs or videos")
    # parser.add_argument("--view3_glob", type=str, default="**/kinect_color/**/master", help="Glob for view3 image dirs or videos")
    parser.add_argument("--image_exts", type=str, default=".jpg,.jpeg,.png", help="Image extensions for image sequences")
    parser.add_argument("--fps", type=int, default=30, help="FPS for image sequences")

    parser.add_argument("--static_view3", action="store_true", help="Assume view3 camera is static")
    parser.add_argument("--f_mm_view1", type=int, default=None, help="Optional focal length (mm) for view1")
    parser.add_argument("--f_mm_view3", type=int, default=None, help="Optional focal length (mm) for view3")
    parser.add_argument("--exo_intrinsics_json", type=str, default="kinect_cam_params/kinect_master/Color.json", help="Path to exo (Kinect) intrinsics json")
    # parser.add_argument("--ego_pv_txt", type=str, default="", help="Path to ego pv.txt (contains per-frame fx/fy and pv2world)")

    parser.add_argument("--save_video_debug", action="store_true")
    return parser.parse_args()


def _save_tensor(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def _collect_images(image_dir: Path, image_exts):
    exts = {e.lower() for e in image_exts}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(images)


def parse_name(img_paths):
    # timestamp_frame_xxxxx.jpg
    ts_list = []
    frame_list = []

    for img_path in img_paths:
        fname = Path(img_path).name
        stem = Path(fname).stem
        ts, frame = stem.split("_frame_")
        ts_list.append(ts)
        frame_list.append(int(frame))

    return ts_list, frame_list


def _load_exo_intrinsics(json_path: str):
    if not json_path:
        return None
    data = json.loads(Path(json_path).read_text())
    K = torch.tensor(data["camera_mtx"], dtype=torch.float32)
    return K


def _load_ego_pv_txt(pv_path: str):
    if not pv_path:
        return None
    lines = [l.strip() for l in Path(pv_path).read_text().splitlines() if l.strip()]
    cx, cy, w, h = [float(x) for x in lines[0].split(",")]
    per_frame = {}
    for line in lines[1:]:
        vals = [float(x) for x in line.split(",")]
        timestamp = int(vals[0])
        fx, fy = vals[1], vals[2]
        mat = np.array(vals[3:]).reshape(4, 4)
        per_frame[str(timestamp)] = {"fx": fx, "fy": fy, "pv2world": mat}
    meta = {"cx": cx, "cy": cy, "w": w, "h": h}
    return meta, per_frame

def load_seq_info(csv_path, recording_name):
    df = pd.read_csv(csv_path)
    row = df[df["recording_name"] == recording_name].iloc[0]
    return int(row["start_frame"]), int(row["end_frame"])

def filter_by_frame_range(imgnames, start_f, end_f):
    keep = []
    for i, p in enumerate(imgnames):
        frame_id = int(Path(p).stem.split("_")[-1])  # timestamp_frame_xxxxx
        if start_f <= frame_id <= end_f:
            keep.append(i)
    return np.array(keep, dtype=np.int64)

def load_pv_npz(pv_dir):
    kp = np.load(pv_dir / "keypoints.npz", allow_pickle=True)
    vf = np.load(pv_dir / "valid_frame.npz", allow_pickle=True)
    return kp, vf

def build_pv_inputs(pv_dir, data_info_csv):
    # 1) load npz
    kp_npz, vf_npz = load_pv_npz(pv_dir)

    imgname = kp_npz["imgname"]          # list of image paths
    center = kp_npz["center"]            # (N,2)
    scale  = kp_npz["scale"]             # (N,)
    kpts25 = kp_npz["keypoints"]         # (N,25,3) (x,y,conf)
    valid  = vf_npz["valid"].astype(bool)  # (N,)

    # 2) filter by start/end frame
    recording_name = Path(pv_dir).parents[1].name  # egocentric_color/RECORDING/DATE/PV
    start_f, end_f = load_seq_info(data_info_csv, recording_name)
    keep = filter_by_frame_range(imgname, start_f, end_f)

    imgname = imgname[keep]
    center = center[keep]
    scale  = scale[keep]
    kpts25 = kpts25[keep]
    valid  = valid[keep]
    timestamps, frame_ids = parse_name(imgname)
    # 3) build bbx_xys from center/scale
    # 这里用一个简化转换：w = h = scale * 200（可根据数据文档调整）
    bbx_xys = np.zeros((len(center), 3), dtype=np.float32)
    bbx_xys[:, 0] = center[:, 0]
    bbx_xys[:, 1] = center[:, 1]
    bbx_xys[:, 2] = scale * 200.0  # TODO: 按官方定义调整

    # 4) mask
    mask = {
        "valid": valid,
        "vitpose": False,
        "bbx_xys": True,
        "f_imgseq": True,
        "spv_incam_only": False,
    }

    return imgname, bbx_xys, kpts25, mask, timestamps, frame_ids

def _make_temp_video_from_images(image_dir: Path, out_video: Path, fps: int, image_exts):
    img_paths = _collect_images(image_dir, image_exts)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {image_dir}")

    first = cv2.imread(str(img_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read image: {img_paths[0]}")
    height, width = first.shape[:2]

    writer = get_writer(out_video, fps=fps, crf=23)
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        writer.write_frame(img[:, :, ::-1])  # BGR -> RGB
    writer.close()
    return out_video


def _process_one_video(
    video_path: Path,
    out_dir: Path,
    f_mm=None,
    static_cam=False,
    source_type="video",
    source_path=None,
    frame_ids=None,
    K_fullimg_override=None,
    cam_angvel_override=None,
):
    length, width, height = get_video_lwh(video_path)

    # 1) Tracking
    tracker = Tracker()
    bbx_xyxy = tracker.get_one_track(str(video_path)).float()
    bbx_xys = torch.zeros((len(bbx_xyxy), 3), dtype=torch.float32)
    bbx_xys[:, :2] = (bbx_xyxy[:, :2] + bbx_xyxy[:, 2:]) / 2.0
    bbx_xys[:, 2] = (bbx_xyxy[:, 2] - bbx_xyxy[:, 0]).clamp(min=1)

    # 2) 2D keypoints (VitPose)
    vitpose_extractor = VitPoseExtractor()
    kp2d = vitpose_extractor.extract(str(video_path), bbx_xys)

    # 3) ViT features
    extractor = Extractor()
    f_imgseq = extractor.extract_video_features(str(video_path), bbx_xys)

    # 4) Camera intrinsics
    if K_fullimg_override is not None:
        K_fullimg = K_fullimg_override
    elif f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    # 5) Camera angular velocity
    if cam_angvel_override is not None:
        cam_angvel = cam_angvel_override
    elif static_cam:
        cam_angvel = torch.zeros((length, 6), dtype=torch.float32)
    else:
        cam_angvel = torch.zeros((length, 6), dtype=torch.float32)

    output = {
        "length": length,
        "bbx_xys": bbx_xys,
        "kp2d": kp2d,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "f_imgseq": f_imgseq,
    }
    if frame_ids is not None:
        output["frame_ids"] = frame_ids

    _save_tensor(out_dir / "preprocess.pt", output)
    meta = {
        "video": str(video_path),
        "source_type": source_type,
        "source_path": str(source_path) if source_path is not None else str(video_path),
        "length": int(length),
        "width": int(width),
        "height": int(height),
        "static_cam": static_cam,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def find_view1_dirs(root):
    view1 = []
    for seq in (root / "egocentric_color").iterdir():
        if not seq.is_dir():
            continue
        sub_folders = [f for f in seq.iterdir() if f.is_dir()]
        if not sub_folders:
            continue
        pv_dir = sub_folders[0] / "PV"
        if pv_dir.is_dir():
            view1.append(pv_dir)
    return view1

def find_view3_dirs(root):
    view3 = []
    for seq in (root / "Kinect_color").iterdir():
        if not seq.is_dir():
            continue
        master_dir = seq / "master"
        if master_dir.is_dir():
            view3.append(master_dir)
    return view3

def get_split_from_csv(data_splits_csv, recording_name):
    df = pd.read_csv(data_splits_csv)
    # CSV 里 train/val/test 列包含 recording_name
    for split in ["train", "val", "test"]:
        if split in df.columns and recording_name in df[split].dropna().tolist():
            return split
    return None

def load_smplx_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # data: dict with betas, global_orient, transl, body_pose (numpy)
    return data

def project_joints_to_2d(j3d, K):
    z = j3d[:, 2].copy()
    z[z < 1e-3] = 1e-3
    u = (j3d[:, 0] * K[0, 0]) / z + K[0, 2]
    v = (j3d[:, 1] * K[1, 1]) / z + K[1, 2]
    vis = (j3d[:, 2] > 0.1).astype(np.float32)
    return np.stack([u, v, vis], axis=-1)

def bbox_from_j2d(j2d):
    vis = j2d[:, 2] > 0.5
    if vis.sum() < 3:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    xs = j2d[vis, 0]
    ys = j2d[vis, 1]
    cx = (xs.min() + xs.max()) / 2
    cy = (ys.min() + ys.max()) / 2
    size = max(xs.max() - xs.min(), ys.max() - ys.min())
    return np.array([cx, cy, size], dtype=np.float32)

def load_kinect_to_world(calib_root, recording_name):
    # path: calibrations/RECORDING_NAME/cal_trans/kinect12_to_world/*.json
    calib_dir = Path(calib_root) / recording_name / "cal_trans" / "kinect12_to_world"
    json_path = sorted(list(calib_dir.glob("*.json")))[0]
    data = json.loads(json_path.read_text())
    T = np.array(data["trans"], dtype=np.float32)  # 4x4
    return T

def main():
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.output_root)
    data_info_csv = root / "data_info_release.csv"

    view1_inputs = find_view1_dirs(root / "egocentric_color") 
    view3_inputs = find_view3_dirs(root / "Kinect_color")

    Log.info(f"[EgoBody] view1 inputs: {len(view1_inputs)}")
    Log.info(f"[EgoBody] view3 inputs: {len(view3_inputs)}")

    image_exts = [e.strip() for e in args.image_exts.split(",") if e.strip()]

    exo_K = _load_exo_intrinsics(args.exo_intrinsics_json)    # 第三视角相机内参矩阵

    for input_path in view1_inputs:
        pv_txt_path = list(input_path.parent.glob("*_pv.txt"))[0]
        ego_meta, ego_per_frame = _load_ego_pv_txt(pv_txt_path)             # 第一视角相机内参是逐帧变化的
        img_paths = _collect_images(input_path, image_exts)

        out_dir = out_root /  "view1" / input_path.relative_to(root / "egocentric_color").parts[0]
        Log.info(f"[View1] {input_path} -> {out_dir}")
        imgname, bbx_xys, kpts25, mask, timestamps, frame_ids = build_pv_inputs(input_path.parent, data_info_csv)
        imgs = np.stack([cv2.imread(p)[..., ::-1] for p in img_paths], axis=0)  # (F,H,W,3) RGB
        bbx_xys = torch.tensor(bbx_xys, dtype=torch.float32)                    # (F,3)
        imgs_tensor, bbx_xys_ds = get_batch(imgs, bbx_xys, path_type="np")
        extractor = Extractor()
        f_imgseq = extractor.extract_video_features(imgs_tensor, bbx_xys_ds)

        K_fullimg = None
        cam_angvel = None
        if ego_meta is not None:
            K_list = []
            R_list = []
            default_key = list(ego_per_frame.keys())[0]
            for ts in timestamps:
                meta = ego_per_frame.get(ts, ego_per_frame[default_key])
                fx, fy = meta["fx"], meta["fy"]
                pv2world = meta["pv2world"]   # 4x4
                K = torch.tensor(
                    [[fx, 0.0, ego_meta["cx"]], [0.0, fy, ego_meta["cy"]], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                )
                K_list.append(K)
                R_list.append(torch.tensor(pv2world[:3, :3], dtype=torch.float32))
            K_fullimg = torch.stack(K_list, dim=0)
            R_w2c = torch.stack(R_list, dim=0).transpose(1, 2)
            cam_angvel = compute_cam_angvel(R_w2c)

        output = {
            "length": len(imgname),
            "bbx_xys": bbx_xys,
            "kp2d_25": torch.tensor(kpts25, dtype=torch.float32),
            "mask": mask,
            "f_imgseq": f_imgseq,
            "frame_ids": frame_ids,    # 保持 list/np
            "timestamps": timestamps,  # 保持 list/np
            "imgname": imgname,      # 保持 list/np
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
        }
        _save_tensor(out_dir / "preprocess.pt", output)
     
      
    smplx_model = make_smplx("supermotion").cuda()
    for input_path in view3_inputs:
        recording_name = input_path.relative_to(root / "kinect_color").parts[0]
        out_dir = out_root /  "view3" / recording_name
        Log.info(f"[View3] {input_path} -> {out_dir}")

        # 对齐第一视角和第三视角的帧
        view1_data = torch.load(out_root / "view1" / recording_name / "preprocess.pt")
        frame_ids_view1 = view1_data["frame_ids"].tolist()
        img_paths = _collect_images(input_path, image_exts)
        timestamps, frame_ids_view3 = parse_name(img_paths)
        keep = np.isin(frame_ids_view3, frame_ids_view1)
        frame_ids_view3 = frame_ids_view3[keep]

        split = get_split_from_csv(root / "data_splits.csv", recording_name)
        if split is None:
            Log.warning(f"Cannot find split for {recording_name}, skip.")
            continue
        # 读取calibrations获得R_w2c
        T_kinect_to_world = load_kinect_to_world(root / "calibrations", recording_name)
        kp2d_exo, kp2d_ego = [], []
        bbx_exo, bbx_ego = [], []
        smpl_params_exo_cam, smpl_params_ego_cam = [], []
        for fid in frame_ids_view3:
            # 读取3d的gt
            pkl_exo = root / f"smplx_interactee_{split}/{recording_name}/*/results/frame_{fid:05d}/000.pkl"
            pkl_ego = root / f"smplx_camera_wearer_{split}/{recording_name}/*/results/frame_{fid:05d}/000.pkl"
            if not pkl_exo.exists() or not pkl_ego.exists():
                continue

            exo = load_smplx_pkl(pkl_exo)
            ego = load_smplx_pkl(pkl_ego)
            smpl_params_exo_cam.append(exo)
            smpl_params_ego_cam.append(ego)
            exo_out = smplx_model(
                betas=torch.tensor(exo["betas"]).unsqueeze(0).cuda(),
                global_orient=torch.tensor(exo["global_orient"]).unsqueeze(0).cuda(),
                body_pose=torch.tensor(exo["body_pose"]).unsqueeze(0).cuda(),
                transl=torch.tensor(exo["transl"]).unsqueeze(0).cuda(),
            )
            ego_out = smplx_model(
                betas=torch.tensor(ego["betas"]).unsqueeze(0).cuda(),
                global_orient=torch.tensor(ego["global_orient"]).unsqueeze(0).cuda(),
                body_pose=torch.tensor(ego["body_pose"]).unsqueeze(0).cuda(),
                transl=torch.tensor(ego["transl"]).unsqueeze(0).cuda(),
            )
            投影到2d
            分别获取边界框
            
            计算图像特征

        

        cam_angvel = torch.zeros((len(frame_ids), 6), dtype=torch.float32) if args.static_view3 else None

       
        


if __name__ == "__main__":
    main()