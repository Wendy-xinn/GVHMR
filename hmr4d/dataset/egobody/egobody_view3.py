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


class EgoBodyView3Dataset(ImgfeatMotionDatasetBase):
    def __init__(
        self,
        root="/public/home/wenxin/egobody",
        output_root="/public/home/wenxin/egobody/output",
        split="train",
        role="ego",  # exo or ego
        motion_frames=120,
        lazy_load=True,
        use_kp2d="none",  # none / project (后续可扩展)
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
        self._vitpose = None

        super().__init__()

    def _load_dataset(self):
        split_csv = self.root / "data_splits.csv"
        df = pd.read_csv(split_csv)
        if self.split not in df.columns:
            raise ValueError(f"split {self.split} not in {split_csv}")

        recs = df[self.split].dropna().tolist()
        valid_recs = []
        for r in recs:
            p = self.output_root / "view3" / r / "preprocess_view3.pt"
            if p.exists():
                valid_recs.append(r)
                self._preproc_paths[r] = p
                data = torch.load(p, map_location="cpu")
                self._seq_lens[r] = int(data["length"])
            else:
                Log.warning(f"[EgoBodyView3] preprocess missing: {p}")
        self.recordings = valid_recs
        Log.info(f"[EgoBodyView3] split={self.split}, recordings={len(self.recordings)}")

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
        if recording_name in self._T_c2w_cache:
            return self._T_c2w_cache[recording_name]
        T = _load_kinect_to_world(self.root / "calibrations", recording_name)
        self._T_c2w_cache[recording_name] = T
        return T

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

        role = self.role
        bbx_key = "bbx_xys_exo" if role == "exo" else "bbx_xys_ego"
        f_key = "f_imgseq_exo" if role == "exo" else "f_imgseq_ego"

        bbx_xys = data[bbx_key][start:end].float()
        f_imgseq = data[f_key][start:end].float()
        K_fullimg = data["K_fullimg"][start:end].float()
        frame_ids = np.array(data["frame_ids"])[start:end]

        mask_valid = np.array(data["mask"]["valid"])[start:end].astype(bool)

        # Load SMPLX params (camera coord)
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
                mask_valid[np.where(frame_ids == fid)[0][0]] = False
                continue

            p = _load_smplx_pkl(pkl_path)
            body_pose.append(np.asarray(p["body_pose"]).reshape(-1)[:63].astype(np.float32))
            betas.append(np.asarray(p["betas"]).reshape(-1)[:10].astype(np.float32))
            global_orient.append(np.asarray(p["global_orient"]).reshape(-1)[:3].astype(np.float32))
            transl.append(np.asarray(p["transl"]).reshape(-1)[:3].astype(np.float32))

        smpl_params_c = {
            "body_pose": torch.tensor(np.stack(body_pose), dtype=torch.float32),
            "betas": torch.tensor(np.stack(betas), dtype=torch.float32),
            "global_orient": torch.tensor(np.stack(global_orient), dtype=torch.float32),
            "transl": torch.tensor(np.stack(transl), dtype=torch.float32),
        }

        # Convert to world
        T_c2w = self._get_T_c2w(recording)
        smpl_params_w = _transform_c2w(smpl_params_c, T_c2w)

        # R_c2gv
        R_w2c = T_c2w[:3, :3].T  # (3,3)
        R_c2gv = get_R_c2gv(R_w2c).unsqueeze(0).repeat(end - start, 1, 1)

        # static camera
        cam_angvel = torch.zeros((end - start, 6), dtype=torch.float32)

        # kp2d (val/test 使用 vitpose)
        if self.use_kp2d == "vitpose" and self.split in ("val", "test"):
            if self._vitpose is None:
                self._vitpose = VitPoseExtractor(tqdm_leave=False)
            img_paths = np.array(data["imgname"], dtype=object)[start:end]
            imgs_np = np.stack([cv2.imread(str(p))[..., ::-1] for p in img_paths], axis=0)
            imgs_t, bbx_xys_ds = get_batch(imgs_np, bbx_xys.clone(), img_ds=1.0, path_type="np")
            kp2d = self._vitpose.extract(imgs_t, bbx_xys_ds, img_ds=1.0).float()
            kp2d[~torch.tensor(mask_valid)] = 0
            vitpose_flag = True
        else:
            kp2d = torch.zeros((end - start, 17, 3), dtype=torch.float32)
            vitpose_flag = False

        return {
            "meta": {"data_name": "egobody_view3", "recording": recording, "view": "view3", "role": role},
            "length": end - start,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,
            "gravity_vec": torch.tensor([0.0, 0.0, -1.0]),
            "bbx_xys": bbx_xys,
            "K_fullimg": K_fullimg,
            "f_imgseq": f_imgseq,
            "kp2d": kp2d,
            "cam_angvel": cam_angvel,
            "mask": {
                "valid": torch.tensor(mask_valid, dtype=torch.bool),
                "vitpose": vitpose_flag,
                "bbx_xys": True,
                "f_imgseq": True,
                "spv_incam_only": False,
            },
        }

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

        return return_data


group_name = "train_datasets/egobody"
node_v1 = builds(EgoBodyView3Dataset, populate_full_signature=True)
MainStore.store(name="view3_v1", node=node_v1, group=group_name)