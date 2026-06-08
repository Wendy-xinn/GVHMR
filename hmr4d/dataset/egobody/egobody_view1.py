import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import pickle
import json
import cv2

from hmr4d.configs import MainStore, builds
from hmr4d.utils.pylogger import Log
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from hmr4d.utils.geo.hmr_global import get_R_c2gv
from hmr4d.utils.preproc import VitPoseExtractor
from hmr4d.utils.preproc.vitfeat_extractor import get_batch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


def _load_kinect_to_world(calib_root, recording_name):
    calib_dir = Path(calib_root) / recording_name / "cal_trans" / "kinect12_to_world"
    json_path = sorted(list(calib_dir.glob("*.json")))[0]
    data = json.loads(json_path.read_text())
    T = torch.tensor(data["trans"], dtype=torch.float32)  # (4,4) c2w
    return T


def _load_holo_to_kinect(calib_root, recording_name):
    """加载 holo → kinect12 的变换矩阵"""
    calib_path = Path(calib_root) / recording_name / "cal_trans" / "holo_to_kinect12.json"
    if not calib_path.exists():
        Log.warning(f"[EgoBodyView1] holo_to_kinect12.json not found: {calib_path}")
        return None
    with open(calib_path, 'r') as f:
        holo_to_kinect = json.load(f)
    T_h2k = np.array(holo_to_kinect['trans'], dtype=np.float32)  # (4,4) holo → kinect
    T_k2h = np.linalg.inv(T_h2k)  # kinect → holo
    return torch.tensor(T_k2h, dtype=torch.float32)  # kinect → holo


def _load_smplx_pkl(pkl_path: Path) -> Dict[str, np.ndarray]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def _aa_to_mat(aa: torch.Tensor) -> torch.Tensor:
    return axis_angle_to_matrix(aa)


def _mat_to_aa(R: torch.Tensor) -> torch.Tensor:
    return matrix_to_axis_angle(R)


def _transform_c2w(smpl_params_c: Dict[str, torch.Tensor], T_c2w: torch.Tensor) -> Dict[str, torch.Tensor]:
    """将相机坐标下 SMPL 参数转换到世界坐标."""
    R_c2w = T_c2w[:3, :3]
    t_c2w = T_c2w[:3, 3]

    R_c = _aa_to_mat(smpl_params_c["global_orient"])  # (F,3,3)
    R_w = R_c2w @ R_c
    global_orient_w = _mat_to_aa(R_w)

    transl_c = smpl_params_c["transl"]  # (F,3)
    transl_w = (R_c2w @ transl_c.T).T + t_c2w  # (F,3)

    smpl_params_w = {
        "body_pose": smpl_params_c["body_pose"].clone(),
        "betas": smpl_params_c["betas"].clone(),
        "global_orient": global_orient_w,
        "transl": transl_w,
    }
    return smpl_params_w


def _load_pv_txt(recording_name: str, root: Path, imgname_slice: List[str]):
    """从 PV 文件读取每帧的 pv2world，返回 world2pv 变换矩阵和 K_fullimg.
    
    PV 文件路径: root/egocentric_color/recording_name/date/date_pv.txt
    图像文件名格式: timestamp_frame_xxxxx.jpg
    
    Args:
        recording_name: 录制名称
        root: EgoBody 根目录
        imgname_slice: 图像路径列表，用于从中提取 timestamps
    
    Returns:
        T_w2c: (F, 4, 4) world → PV camera
        K_fullimg: (F, 3, 3) 相机内参
    """
    # 找到 PV 目录: root/egocentric_color/recording_name/date/PV
    pv_dir = None
    rec_dir = root / "egocentric_color" / recording_name
    if rec_dir.exists():
        for date_dir in rec_dir.iterdir():
            if date_dir.is_dir():
                pv_txt = date_dir / f"{date_dir.name}_pv.txt"
                if pv_txt.exists():
                    pv_dir = date_dir
                    break
    
    if pv_dir is None:
        Log.warning(f"[EgoBodyView1] PV txt not found for {recording_name}")
        return None, None
    
    pv_txt_path = pv_dir / f"{pv_dir.name}_pv.txt"
    lines = [l.strip() for l in pv_txt_path.read_text().splitlines() if l.strip()]
    cx, cy, w, h = [float(x) for x in lines[0].split(",")]
    
    # 解析每帧数据
    per_frame = {}
    for line in lines[1:]:
        vals = [float(x) for x in line.split(",")]
        timestamp = int(vals[0])
        fx, fy = vals[1], vals[2]
        pv2world = np.array(vals[3:]).reshape(4, 4)
        per_frame[timestamp] = {"fx": fx, "fy": fy, "pv2world": pv2world}
    
    # 从 imgname 中提取 timestamps
    # 格式: timestamp_frame_xxxxx.jpg 或 path/timestamp_frame_xxxxx.jpg
    timestamps = []
    for p in imgname_slice:
        stem = Path(p).stem  # timestamp_frame_xxxxx
        ts_str = stem.split("_frame_")[0]
        timestamps.append(int(ts_str))
    
    # 根据 timestamps 构建 T_w2c 和 K
    T_w2c_list = []
    K_list = []
    
    # 使用第一个可用的 timestamp 作为 fallback
    default_key = list(per_frame.keys())[0] if per_frame else None
    
    for ts in timestamps:
        meta = per_frame.get(ts, per_frame.get(default_key, None))
        if meta is None:
            # fallback: 使用单位矩阵
            T_w2c_list.append(np.eye(4, dtype=np.float32))
            K_list.append(np.eye(3, dtype=np.float32))
            continue
        
        pv2world = meta["pv2world"]  # (4,4) PV → world
        world2pv = np.linalg.inv(pv2world)  # world → PV
        T_w2c_list.append(world2pv)
        
        fx, fy = meta["fx"], meta["fy"]
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        K_list.append(K)
    
    T_w2c = torch.tensor(np.stack(T_w2c_list), dtype=torch.float32)  # (F, 4, 4)
    K_fullimg = torch.tensor(np.stack(K_list), dtype=torch.float32)  # (F, 3, 3)
    
    return T_w2c, K_fullimg


class EgoBodyView1Dataset(ImgfeatMotionDatasetBase):
    """EgoBody View1 (第一人称 PV 相机) 数据集.
    
    坐标系说明:
        - SMPLX pkl 文件中的参数在 Kinect12 相机坐标系下
        - 通过 holo_to_kinect12.json 获取 kinect → holo world 的变换
        - 通过 pv.txt 中的 T_w2c (holo world → PV camera) 转换到 PV 相机坐标系
        - smpl_params_c: PV 相机坐标系下的 SMPL 参数 (用于 incam loss)
        - smpl_params_w: Holo 世界坐标系下的 SMPL 参数 (用于 global loss)
    """
    def __init__(
        self,
        root="/public/home/wenxin/egobody",
        output_root="/public/home/wenxin/egobody/output",
        split="train",
        role="ego",  # exo or ego
        motion_frames=120,
        lazy_load=True,
        use_kp2d="vitpose",  
    ):
        self.root = Path(root)
        self.output_root = Path(output_root)
        self.split = split
        self.role = role
        self.motion_frames = motion_frames
        self.lazy_load = lazy_load
        self.use_kp2d = use_kp2d

        self._preproc_paths = {}
        self._seq_lens = {}
        self._pkl_index = {}
        self._T_c2w_cache = {}
        self._T_kinect2holo_cache = {}

        super().__init__()

    def _load_dataset(self):
        split_csv = self.root / "data_splits.csv"
        df = pd.read_csv(split_csv)
        if self.split not in df.columns:
            raise ValueError(f"split {self.split} not in {split_csv}")

        recs = df[self.split].dropna().tolist()
        valid_recs = []
        for r in recs:
            p = self.output_root / "view1" / r / "preprocess_view1.pt"
            if p.exists():
                valid_recs.append(r)
                self._preproc_paths[r] = p
                data = torch.load(p, map_location="cpu")
                self._seq_lens[r] = int(data["length"])
            else:
                Log.warning(f"[EgoBodyView1] preprocess missing: {p}")
        self.recordings = valid_recs
        Log.info(f"[EgoBodyView1] split={self.split}, recordings={len(self.recordings)}")

    def _get_idx2meta(self):
        self.idx2meta = []
        for r in self.recordings:
            L = self._seq_lens[r]
            num_samples = max(L // self.motion_frames, 1)
            self.idx2meta.extend([r] * num_samples)

    def _build_pkl_index(self, recording_name: str, role: str):
        key = (recording_name, role)
        if key in self._pkl_index:
            return

        if role == "exo":
            base = self.root / f"smplx_interactee_{self.split}" / recording_name
        else:
            base = self.root / f"smplx_camera_wearer_{self.split}" / recording_name

        frame_map = {}
        for p in base.glob("*/results/frame_*/000.pkl"):
            # .../frame_00012/000.pkl
            frame_id = int(p.parent.name.replace("frame_", ""))
            frame_map[frame_id] = p
        self._pkl_index[key] = frame_map

    def _get_T_c2w(self, recording_name: str):
        """获取 kinect12 → world 的变换矩阵 (用于 view3 兼容)"""
        if recording_name in self._T_c2w_cache:
            return self._T_c2w_cache[recording_name]
        T = _load_kinect_to_world(self.root / "calibrations", recording_name)
        self._T_c2w_cache[recording_name] = T
        return T

    def _get_T_kinect2holo(self, recording_name: str):
        """获取 kinect12 → holo world 的变换矩阵"""
        if recording_name in self._T_kinect2holo_cache:
            return self._T_kinect2holo_cache[recording_name]
        T = _load_holo_to_kinect(self.root / "calibrations", recording_name)
        self._T_kinect2holo_cache[recording_name] = T
        return T

    def _load_smpl_params_for_role(self, recording: str, role: str, frame_ids: np.ndarray, mask_valid: np.ndarray):
        """加载指定 role 的 SMPLX 参数 (在 kinect12 相机坐标系下)."""
        self._build_pkl_index(recording, role)
        pkl_map = self._pkl_index[(recording, role)]

        body_pose = []
        betas = []
        global_orient = []
        transl = []

        for fid, v in zip(frame_ids, mask_valid):
            pkl_path = pkl_map.get(int(fid), None)
            if pkl_path is None or not v:
                body_pose.append(np.zeros((63,), dtype=np.float32))
                betas.append(np.zeros((10,), dtype=np.float32))
                global_orient.append(np.zeros((3,), dtype=np.float32))
                transl.append(np.zeros((3,), dtype=np.float32))
                continue

            p = _load_smplx_pkl(pkl_path)
            body_pose.append(np.asarray(p["body_pose"]).reshape(-1)[:63].astype(np.float32))
            betas.append(np.asarray(p["betas"]).reshape(-1)[:10].astype(np.float32))
            global_orient.append(np.asarray(p["global_orient"]).reshape(-1)[:3].astype(np.float32))
            transl.append(np.asarray(p["transl"]).reshape(-1)[:3].astype(np.float32))

        smpl_params_kinect = {
            "body_pose": torch.tensor(np.stack(body_pose), dtype=torch.float32),
            "betas": torch.tensor(np.stack(betas), dtype=torch.float32),
            "global_orient": torch.tensor(np.stack(global_orient), dtype=torch.float32),
            "transl": torch.tensor(np.stack(transl), dtype=torch.float32),
        }
        return smpl_params_kinect

    def _load_data(self, idx):
        recording = self.idx2meta[idx]
        data = torch.load(self._preproc_paths[recording], map_location="cpu")

        length = int(data["length"])
        target_length = self.motion_frames
        if target_length > length:
            start, end = 0, length
        else:
            start = np.random.randint(0, length - target_length + 1)
            end = start + target_length

        # View1 始终使用 ego 的图像特征和 bbox (PV 相机视角)
        bbx_xys = torch.as_tensor(data["bbx_xys_ego"][start:end]).float()
        f_imgseq = torch.as_tensor(data["f_imgseq_ego"][start:end]).float()
        frame_ids = np.array(data["frame_ids"])[start:end]
        
        # 加载 imgname（图像路径）
        if "imgname" in data:
            imgname_all = data["imgname"]
            if isinstance(imgname_all, np.ndarray):
                imgname = [str(p) for p in imgname_all[start:end]]
            else:
                imgname = [str(p) for p in list(imgname_all)[start:end]]
        else:
            imgname = []

        # ============================================================
        # 从 PV 文件读取 T_w2c (world → PV camera) 和 K_fullimg
        # ============================================================
        T_w2c, K_from_pv = _load_pv_txt(recording, self.root, imgname)
        
        # 如果 PV 文件读取失败，fallback 到 preprocess 中的 K_fullimg
        if K_from_pv is not None:
            K_fullimg = K_from_pv
        else:
            K_fullimg = data["K_fullimg"][start:end].float()
            T_w2c = None

        mask_valid = np.array(data["mask"]["valid"])[start:end].astype(bool)

        # ============================================================
        # 加载 SMPLX 参数 (role 指定的目标人物，如 ego=camera_wearer)
        # ============================================================
        role = self.role
        smpl_params_kinect = self._load_smpl_params_for_role(recording, role, frame_ids, mask_valid)

        # 坐标系转换链路: kinect12 → holo world → PV camera
        # Step 1: kinect12 → holo world
        T_kinect2holo = self._get_T_kinect2holo(recording)
        if T_kinect2holo is not None:
            smpl_params_w = _transform_c2w(smpl_params_kinect, T_kinect2holo)
        else:
            # fallback: 直接用 kinect 坐标作为 world
            smpl_params_w = {k: v.clone() for k, v in smpl_params_kinect.items()}
            T_kinect2holo = torch.eye(4, dtype=torch.float32)

        # Step 2: holo world → PV camera (使用从 PV 文件读取的 T_w2c)
        if T_w2c is not None:
            # 逐帧转换: world → PV camera
            R_w2c = T_w2c[:, :3, :3]  # (F, 3, 3)
            t_w2c = T_w2c[:, :3, 3]   # (F, 3)
            
            R_w = _aa_to_mat(smpl_params_w["global_orient"])  # (F, 3, 3)
            R_c = R_w2c @ R_w  # (F, 3, 3)
            global_orient_c = _mat_to_aa(R_c)
            
            transl_w = smpl_params_w["transl"]  # (F, 3)
            transl_c = (R_w2c @ transl_w.unsqueeze(-1)).squeeze(-1) + t_w2c  # (F, 3)
            
            smpl_params_c = {
                "body_pose": smpl_params_w["body_pose"].clone(),
                "betas": smpl_params_w["betas"].clone(),
                "global_orient": global_orient_c,
                "transl": transl_c,
            }
        else:
            # fallback: 没有 T_w2c 时，用 kinect 坐标作为 camera 坐标
            smpl_params_c = {k: v.clone() for k, v in smpl_params_kinect.items()}

        # ============================================================
        # 可选: 加载 interactee 的 SMPLX 参数 (用于后续 exo 头训练)
        # ============================================================
        interactee_role = "exo" 
        interactee_smpl_params_c = None
        interactee_smpl_params_w = None
        
        try:
            interactee_smpl_kinect = self._load_smpl_params_for_role(
                recording, interactee_role, frame_ids, mask_valid
            )
            if T_kinect2holo is not None:
                interactee_smpl_params_w = _transform_c2w(interactee_smpl_kinect, T_kinect2holo)
            
            if T_w2c is not None:
                R_w2c = T_w2c[:, :3, :3]
                t_w2c = T_w2c[:, :3, 3]
                
                R_w = _aa_to_mat(interactee_smpl_params_w["global_orient"])
                R_c = R_w2c @ R_w
                global_orient_c = _mat_to_aa(R_c)
                
                transl_w = interactee_smpl_params_w["transl"]
                transl_c = (R_w2c @ transl_w.unsqueeze(-1)).squeeze(-1) + t_w2c
                
                interactee_smpl_params_c = {
                    "body_pose": interactee_smpl_params_w["body_pose"].clone(),
                    "betas": interactee_smpl_params_w["betas"].clone(),
                    "global_orient": global_orient_c,
                    "transl": transl_c,
                }
        except Exception as e:
            Log.warning(f"[EgoBodyView1] Failed to load interactee ({interactee_role}) SMPLX: {e}")

        # ============================================================
        # 计算 R_c2gv (相机 → gravity-view 坐标系)
        # ============================================================
        # 使用 PV 相机的外参 (T_w2c 的旋转部分)
        if T_w2c is not None:
            R_w2pv = T_w2c[0, :3, :3]  # (3,3) 
        R_c2gv = get_R_c2gv(R_w2pv).unsqueeze(0).repeat(end - start, 1, 1)

        # 相机角速度 (PV 相机跟随 camera wearer 运动，从 preprocess 读取)
        if "cam_angvel" in data:
            cam_angvel = data["cam_angvel"][start:end].float()
        else:
            cam_angvel = torch.zeros((end - start, 6), dtype=torch.float32)

        # kp2d (val/test 使用 vitpose)
        if self.use_kp2d == "vitpose" and self.split in ("val", "test"):
            kp2d_file = self.root / "vitpose" / "view1" / recording / "vitpose_kp2d.pt"
            
            # 读取字典里存放的 key，例如 "kp2d_exo" 或 "kp2d_ego"
            kp2d_key = f"kp2d_{role}" 
            
            if kp2d_file.exists():
                kp2d_data = torch.load(kp2d_file, map_location="cpu")
                if kp2d_key in kp2d_data:
                    kp2d = kp2d_data[kp2d_key][start:end].float()
                else:
                    Log.warning(f"[{self.split}] Key {kp2d_key} missing in {kp2d_file}, filling with zeros.")
                    kp2d = torch.zeros((end - start, 17, 3), dtype=torch.float32)
            else:
                Log.warning(f"[{self.split}] Kp2d file missing: {kp2d_file}, filling with zeros.")
                kp2d = torch.zeros((end - start, 17, 3), dtype=torch.float32)
            kp2d[~torch.tensor(mask_valid)] = 0
            vitpose_flag = True
        else:
            kp2d = torch.zeros((end - start, 17, 3), dtype=torch.float32)
            vitpose_flag = False

        # ============================================================
        # 组装返回数据
        # ============================================================
        return_data = {
            "meta": {"data_name": "egobody_view1", "recording": recording, "view": "view1", "role": role},
            "length": end - start,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,
            "gravity_vec": torch.tensor([0.0, -1.0, 0.0]),
            "bbx_xys": bbx_xys,
            "K_fullimg": K_fullimg,
            "f_imgseq": f_imgseq,
            "kp2d": kp2d,
            "cam_angvel": cam_angvel,
            "imgname": imgname,
            # Interactee 数据 (可选，用于后续 exo 头训练)
            "interactee_smpl_params_c": interactee_smpl_params_c,
            "interactee_smpl_params_w": interactee_smpl_params_w,
            
            "mask": {
                "valid": torch.tensor(mask_valid, dtype=torch.bool),
                "vitpose": vitpose_flag,
                "bbx_xys": True,
                "f_imgseq": True,
                "spv_incam_only": False,
            },
        }
        return return_data

    def _process_data(self, data, idx):
        length = data["length"]
        max_len = self.motion_frames

        return_data = {
            **data,
            "length": length,
        }

        return_data["smpl_params_c"] = repeat_to_max_len_dict(return_data["smpl_params_c"], max_len)
        return_data["smpl_params_w"] = repeat_to_max_len_dict(return_data["smpl_params_w"], max_len)
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
        return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)
        return_data["mask"]["valid"] = get_valid_mask(max_len, length)
        
        # 处理 interactee 数据 (如果存在)
        if return_data.get("interactee_smpl_params_c") is not None:
            return_data["interactee_smpl_params_c"] = repeat_to_max_len_dict(
                return_data["interactee_smpl_params_c"], max_len
            )
        if return_data.get("interactee_smpl_params_w") is not None:
            return_data["interactee_smpl_params_w"] = repeat_to_max_len_dict(
                return_data["interactee_smpl_params_w"], max_len
            )

        return return_data


group_name = "train_datasets/egobody"
node_v1 = builds(EgoBodyView1Dataset, populate_full_signature=True)
MainStore.store(name="view1_v1", node=node_v1, group=group_name)