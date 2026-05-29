import argparse
from pathlib import Path
import json
import torch
import numpy as np
import cv2
import pandas as pd
import pickle
from tqdm import tqdm
import gc

from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import get_video_lwh, get_writer
from hmr4d.utils.preproc import Tracker, VitPoseExtractor, Extractor
from hmr4d.utils.geo.hmr_cam import estimate_K, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.preproc.vitfeat_extractor import get_batch
from hmr4d.utils.smplx_utils import make_smplx


def parse_args():
    parser = argparse.ArgumentParser(description="EgoBody offline preprocessing (image sequences)")
    parser.add_argument("--root", type=str, default="/public/home/wenxin/egobody", help="EgoBody root directory")
    parser.add_argument("--output_root", type=str, default="/public/home/wenxin/egobody/output", help="Output directory")

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
    tmp_path = path.with_suffix(".tmp")
    torch.save(data, tmp_path)
    tmp_path.replace(path)


def _collect_images(image_dir: Path, image_exts):
    exts = {e.lower() for e in image_exts}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(images)


def parse_name(img_paths):
    # timestamp_frame_xxxxx.jpg
    ts_list = []
    frame_list = []

    for img_path in img_paths:
        stem = Path(img_path).stem
        ts, frame = stem.split("_frame_")
        ts_list.append(ts)
        frame_list.append(int(frame))

    return (
        np.array(ts_list, dtype=object),
        np.array(frame_list, dtype=np.int64),
    )


def _load_exo_intrinsics(json_path: str):
    if not json_path:
        return None
    data = json.loads(Path(json_path).read_text())
    K = np.array(data["camera_mtx"], dtype=np.float32)
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

    imgname = np.array(kp_npz["imgname"], dtype=object)        # list of image paths
    # print(imgname)
    center = kp_npz["center"]            # (N,2)
    scale  = kp_npz["scale"]             # (N,)
    kpts25 = kp_npz["keypoints"]         # (N,25,3) (x,y,conf)
    valid  = vf_npz["valid"].astype(bool)  # (N,)

    # 2) filter by start/end frame
    recording_name = Path(pv_dir).parents[0].name  # egocentric_color/RECORDING/DATE/PV
    start_f, end_f = load_seq_info(data_info_csv, recording_name)
    # print(start_f, end_f)
    keep = filter_by_frame_range(imgname, start_f, end_f)

    imgname = imgname[keep]
    # print(imgname[:20])
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
    # view1.append(root / "egocentric_color" / "recording_20210907_S02_S01_01" / "2021-09-07-155421" / "PV" )
    return view1

def find_view3_dirs(root):
    view3 = []
    for seq in (root / "kinect_color").iterdir():
        if not seq.is_dir():
            continue
        master_dir = seq / "master"
        if master_dir.is_dir():
            view3.append(master_dir)
    # view3.append(root / "kinect_color" / "recording_20210907_S02_S01_01" / "master")
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
    T = torch.tensor(data["trans"], dtype=torch.float32)  # 4x4
    return T

def xys_to_xyxy(bbx_xys):
    cx, cy, s = bbx_xys
    x1 = cx - s / 2
    y1 = cy - s / 2
    x2 = cx + s / 2
    y2 = cy + s / 2
    return [x1, y1, x2, y2]

def xyxy_to_xys(bbx_xyxy):
    x1, y1, x2, y2 = bbx_xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1)
    return np.array([cx, cy, s], dtype=np.float32)

def iou_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def match_yolo_with_gt(yolo_boxes, gt_exo_xyxy, gt_ego_xyxy):
    if len(yolo_boxes) == 0:
        return None, None
    if len(yolo_boxes) == 1:
        iou_exo = iou_xyxy(yolo_boxes[0], gt_exo_xyxy)
        iou_ego = iou_xyxy(yolo_boxes[0], gt_ego_xyxy)
        return (yolo_boxes[0], None) if iou_exo >= iou_ego else (None, yolo_boxes[0])

    best_exo = max(yolo_boxes, key=lambda b: iou_xyxy(b, gt_exo_xyxy))
    best_ego = max(yolo_boxes, key=lambda b: iou_xyxy(b, gt_ego_xyxy))
    if np.allclose(best_exo, best_ego):
        sorted_boxes = sorted(yolo_boxes, key=lambda b: iou_xyxy(b, gt_ego_xyxy), reverse=True)
        best_ego = sorted_boxes[1] if len(sorted_boxes) > 1 else None
    return best_exo, best_ego

def main():
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.output_root)
    data_info_csv = root / "data_info_release.csv"

    view1_inputs = find_view1_dirs(root) 
    view3_inputs = find_view3_dirs(root)

    Log.info(f"[EgoBody] view1 inputs: {len(view1_inputs)}")
    Log.info(f"[EgoBody] view3 inputs: {len(view3_inputs)}")

    image_exts = [e.strip() for e in args.image_exts.split(",") if e.strip()]
    exo_K = _load_exo_intrinsics(root / args.exo_intrinsics_json)    # 第三视角相机内参矩阵
    extractor = Extractor()

    for input_path in tqdm(view1_inputs, desc="[View1] sequences"):
        pv_txt_path = list(input_path.parent.glob("*_pv.txt"))[0]
        ego_meta, ego_per_frame = _load_ego_pv_txt(pv_txt_path)             # 第一视角相机内参是逐帧变化的

        out_dir = out_root /  "view1" / input_path.relative_to(root / "egocentric_color").parts[0]
        if out_dir.exists():
            Log.info(f"[Skip] {out_dir} already exists")
            continue
        Log.info(f"[View1] {input_path} -> {out_dir}")

        imgname, bbx_xys, kpts25, mask, timestamps, frame_ids = build_pv_inputs(input_path.parent, data_info_csv)
        img_paths = [Path(p) for p in imgname]
        # imgs = np.stack([cv2.imread(str(root / p))[..., ::-1] for p in img_paths], axis=0)  # (F,H,W,3) RGB    容易爆cpu内存
        chunk = 512  # 或 128
        f_list = []
        for start in range(0, len(img_paths), chunk):
            end = min(start + chunk, len(img_paths))
            imgs = np.stack([cv2.imread(str(root / p))[..., ::-1] for p in img_paths[start:end]], axis=0)
            imgs_t, bbx_ds = get_batch(imgs, torch.tensor(bbx_xys, dtype=torch.float32)[start:end], img_ds=1.0, path_type="np")
            f_list.append(extractor.extract_video_features(imgs_t, bbx_ds))
            del imgs, imgs_t, bbx_ds
            gc.collect()
        if not f_list:
            print("f_list 为空，跳过当前拼接:", input_path)
            continue
        f_imgseq = torch.cat(f_list, dim=0)     
        # bbx_xys = torch.tensor(bbx_xys, dtype=torch.float32)                    # (F,3)
        # imgs_tensor, bbx_xys_ds = get_batch(imgs, bbx_xys, img_ds=1.0, path_type="np")
        # f_imgseq = extractor.extract_video_features(imgs_tensor, bbx_xys_ds)

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
            "bbx_xys_ego": bbx_xys,
            # "kp2d_25": torch.tensor(kpts25, dtype=torch.float32),
            "mask": mask,
            "f_imgseq_ego": f_imgseq,
            "frame_ids": frame_ids,    # 保持 list/np
            # "timestamps": timestamps,  # 保持 list/np
            "imgname": imgname,      # 保持 list/np，相对路径
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "R_w2c": R_w2c,   # 这里的坐标系存疑：应该不是pv直接到场景世界坐标系，而是从pv相机坐标系先转换到hololens的世界坐标系，然后再通过holo to kinect12文件转换到kinect12的坐标系，最终转换到场景世界坐标系
        }
        _save_tensor(out_dir / "preprocess_view1.pt", output)
        Log.info(f"[View1_done] {input_path} -> {out_dir}")
        del bbx_xys, f_imgseq, K_fullimg, cam_angvel
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
      
    smplx_model = make_smplx("supermotion").cuda()
    tracker = Tracker()
    batch_size = 64
    smplx_keys = [
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
    
    for input_path in tqdm(view3_inputs, desc="[View3] sequences"):
        recording_name = input_path.relative_to(root / "kinect_color").parts[0]
        out_dir = out_root /  "view3" / recording_name
        if out_dir.exists():
            Log.info(f"[Skip] {out_dir} already exists")
            continue
        Log.info(f"[View3] {input_path} -> {out_dir}")
        view1_dir = out_root / "view1" / recording_name / "preprocess_view1.pt"
        if not view1_dir.exists():
            print("view1 为空，跳过当前拼接:", input_path)
            continue
        # 对齐第一视角和第三视角的帧
        view1_data = torch.load(view1_dir)
        frame_ids_view1 = view1_data["frame_ids"].tolist()
        img_paths = _collect_images(input_path, image_exts)
        # timestamps, frame_ids_view3 = parse_name(img_paths)
        frame_list = []
        for p in img_paths:
            stem = p.stem
            frame = stem.split("frame_")[-1]
            frame_list.append(int(frame))
        frame_ids_view3 = np.array(frame_list, dtype=np.int64)
        keep = np.isin(frame_ids_view3, frame_ids_view1)   # 计算得到的是一个布尔数组，表示哪些frame_ids_view3在frame_ids_view1中出现过
        frame_ids_view3 = frame_ids_view3[keep]
        # timestamps = timestamps[keep]
        img_paths = [p for p, k in zip(img_paths, keep) if k]  # 绝对路径

        split = get_split_from_csv(root / "data_splits.csv", recording_name)
        if split is None:
            Log.warning(f"Cannot find split for {recording_name}, skip.")
            continue
        # 读取calibrations获得R_c2w
        T_kinect_to_world = load_kinect_to_world(root / "calibrations", recording_name)  # 4x4 
        # kp2d_exo, kp2d_ego = [], []
        bbx_exo, bbx_ego = [], []
        valid_view3 = []
        
        # 分 batch
        num_frames = len(frame_ids_view3)
        for start in tqdm(range(0, num_frames, batch_size), desc=f"[View3:{recording_name}] frames", leave=False):
            end = min(start + batch_size, num_frames)
            batch_fids = frame_ids_view3[start:end]
            batch_img_paths = img_paths[start:end]
            # 读图
            batch_imgs = []
            img_ok = []
            for p in batch_img_paths:
                im = cv2.imread(str(p))
                if im is None:
                    batch_imgs.append(None)
                    img_ok.append(False)
                else:
                    batch_imgs.append(im)
                    img_ok.append(True)

            # 找 GT pkl
            exo_pkls = []
            ego_pkls = []
            pkl_ok = []
            for fid in batch_fids:
                pkl_exo_list = sorted(
                    (root / f"smplx_interactee_{split}" / recording_name).glob(f"*/results/frame_{fid:05d}/000.pkl")
                )
                pkl_ego_list = sorted(
                    (root / f"smplx_camera_wearer_{split}" / recording_name).glob(f"*/results/frame_{fid:05d}/000.pkl")
                )
                if not pkl_exo_list or not pkl_ego_list:
                    exo_pkls.append(None)
                    ego_pkls.append(None)
                    pkl_ok.append(False)
                else:
                    exo_pkls.append(pkl_exo_list[0])
                    ego_pkls.append(pkl_ego_list[0])
                    pkl_ok.append(True)

            # batch smplx 前向（只对 ok 的）
            ok_indices = [i for i, ok in enumerate(pkl_ok) if ok and img_ok[i]]
            j2d_exo_list = [None] * len(batch_fids)
            j2d_ego_list = [None] * len(batch_fids)

            if ok_indices:
                exo_params = [load_smplx_pkl(exo_pkls[i]) for i in ok_indices]
                ego_params = [load_smplx_pkl(ego_pkls[i]) for i in ok_indices]

                exo_inputs = {}
                with torch.no_grad():
                    for k in smplx_keys:
                        if k in exo_params[0]:
                            exo_inputs[k] = torch.tensor(
                                np.stack([p[k].squeeze(0) for p in exo_params]),
                                dtype=torch.float32
                            ).cuda()
                            # print(f"{k}: {exo_inputs[k].shape}")
                    exo_out = smplx_model(**exo_inputs)
                    ego_inputs = {}
                    for k in smplx_keys:
                        if k in ego_params[0]:
                            ego_inputs[k] = torch.tensor(
                                np.stack([p[k].squeeze(0) for p in ego_params]),
                                dtype=torch.float32
                            ).cuda()
                    ego_out = smplx_model(**ego_inputs)

                exo_j3d = exo_out.joints.detach().cpu().numpy()
                ego_j3d = ego_out.joints.detach().cpu().numpy()
                del exo_out, ego_out
                for k, idx in enumerate(ok_indices):
                    j2d_exo_list[idx] = project_joints_to_2d(exo_j3d[k], exo_K)
                    j2d_ego_list[idx] = project_joints_to_2d(ego_j3d[k], exo_K)

            # yolo检测边界框
            yolo_imgs = [im for im in batch_imgs if im is not None]
            yolo_results = []
            if yolo_imgs:
                yolo_results = tracker.yolo.predict(yolo_imgs, conf=0.5, classes=0, verbose=False)

            # 映射回每帧
            yi = 0
            for i in range(len(batch_fids)):
                if not img_ok[i] or not pkl_ok[i]:
                    bbx_exo.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                    bbx_ego.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                    valid_view3.append(False)
                    continue

                res = yolo_results[yi] if yolo_results else None
                yi += 1

                yolo_boxes = []
                if res is not None and res.boxes is not None and res.boxes.xyxy is not None:
                    yolo_boxes = res.boxes.xyxy.cpu().numpy().tolist()

                # 规则：YOLO 检测不到直接无效
                if len(yolo_boxes) < 2:
                    bbx_exo.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                    bbx_ego.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                    valid_view3.append(False)
                    continue

                # 用投影区分身份
                j2d_exo = j2d_exo_list[i]
                j2d_ego = j2d_ego_list[i]
                bbx_exo_gt = xys_to_xyxy(bbox_from_j2d(j2d_exo))
                bbx_ego_gt = xys_to_xyxy(bbox_from_j2d(j2d_ego))
                exo_xyxy, ego_xyxy = match_yolo_with_gt(yolo_boxes, bbx_exo_gt, bbx_ego_gt)

                bbx_exo.append(xyxy_to_xys(exo_xyxy) if exo_xyxy is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32))
                bbx_ego.append(xyxy_to_xys(ego_xyxy) if ego_xyxy is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32))
                valid_view3.append(True)
           
            
        # kp2d_exo = np.stack(kp2d_exo)
        # kp2d_ego = np.stack(kp2d_ego)
        bbx_exo = np.stack(bbx_exo)
        bbx_ego = np.stack(bbx_ego)
        valid_view3 = np.array(valid_view3, dtype=bool)
            
        # 计算图像特征
        # imgs = np.stack([cv2.imread(str(p))[..., ::-1] for p in img_paths], axis=0)
        # imgs_t, bbx_ds = get_batch(imgs, torch.tensor(bbx_exo, dtype=torch.float32), img_ds=1.0, path_type="np")
        # f_imgseq_exo =  extractor.extract_video_features(imgs_t, bbx_ds, img_ds=1.0)

        # imgs_t, bbx_ds = get_batch(imgs, torch.tensor(bbx_ego, dtype=torch.float32), img_ds=1.0, path_type="np")
        # f_imgseq_ego =  extractor.extract_video_features(imgs_t, bbx_ds, img_ds=1.0)
        chunk = 512  # 或 128
        f_list_exo = []
        f_list_ego = []
        for start in range(0, len(img_paths), chunk):
            end = min(start + chunk, len(img_paths))
            imgs = np.stack([cv2.imread(str(root / p))[..., ::-1] for p in img_paths[start:end]], axis=0)
            imgs_t, bbx_ds = get_batch(imgs, torch.tensor(bbx_exo, dtype=torch.float32)[start:end], img_ds=1.0, path_type="np")
            f_list_exo.append(extractor.extract_video_features(imgs_t, bbx_ds))
            del imgs_t, bbx_ds
            gc.collect()
            imgs_t, bbx_ds = get_batch(imgs, torch.tensor(bbx_ego, dtype=torch.float32)[start:end], img_ds=1.0, path_type="np")
            f_list_ego.append(extractor.extract_video_features(imgs_t, bbx_ds))
            del imgs, imgs_t, bbx_ds
            gc.collect()
        f_imgseq_exo = torch.cat(f_list_exo, dim=0) 
        f_imgseq_ego = torch.cat(f_list_ego, dim=0) 

        mask_view3 = {
            "valid": np.array(valid_view3, dtype=bool),
            "vitpose": False,
            "bbx_xys": True,
            "f_imgseq": True,
            "spv_incam_only": False,
        }
        # 保存
        output = {
            "frame_ids": frame_ids_view3,
            # "kp2d_exo": torch.tensor(kp2d_exo, dtype=torch.float32),
            # "kp2d_ego": torch.tensor(kp2d_ego, dtype=torch.float32),
            "mask": mask_view3,
            "bbx_xys_exo": torch.tensor(bbx_exo, dtype=torch.float32),
            "bbx_xys_ego": torch.tensor(bbx_ego, dtype=torch.float32),
            "f_imgseq_exo": f_imgseq_exo,
            "f_imgseq_ego": f_imgseq_ego,
            "K_fullimg": torch.from_numpy(exo_K).unsqueeze(0).repeat(len(frame_ids_view3), 1, 1),
            "R_w2c": T_kinect_to_world[:3, :3].T,
            # "smplx_params_exo_cam": smpl_params_exo_cam,
            # "smplx_params_ego_cam": smpl_params_ego_cam,
            "imgname": np.array(img_paths, dtype=object),  # 绝对路径
            "length": len(frame_ids_view3),
        }
        _save_tensor(out_dir / "preprocess_view3.pt", output)
        Log.info(f"[View3_done] {input_path} -> {out_dir}")
        del f_imgseq_exo, f_imgseq_ego
        del bbx_exo, bbx_ego
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        

       
        


if __name__ == "__main__":
    main()