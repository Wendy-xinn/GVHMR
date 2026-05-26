import torch
import numpy as np
from pathlib import Path
from hmr4d.configs import MainStore, builds

from hmr4d.utils.pylogger import Log
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from hmr4d.utils.geo.hmr_global import get_R_c2gv
from hmr4d.utils.geo_transform import compute_cam_angvel

class EgoBodyPairedDataset(torch.utils.data.Dataset):
    def __init__(self, root, motion_frames=120, split="train"):
        self.root = Path(root)
        self.motion_frames = motion_frames
        self.split = split
        self._load_index()

    def _load_index(self):
        # TODO: load pair list (each item aligns view1 + view3)
        # self.index = [(seq_id, start, end), ...]
        self.index = []
        Log.info(f"[EgoBodyPaired] Loaded {len(self.index)} pairs")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # TODO: load paired frames by index
        # v1 = self._load_view1(...)
        # v3 = self._load_view3(...)
        v1 = {}
        v3 = {}

        # ---- view1 (ego camera, exo only) ----
        view1 = {
            "length": v1["length"],
            "smpl_params_c": v1["smpl_params_c_exo"],  # exo in ego cam
            "smpl_params_w": v1["smpl_params_w_exo"],
            "R_c2gv": v1["R_c2gv"],
            "bbx_xys": v1["bbx_xys"],
            "K_fullimg": v1["K_fullimg"],
            "kp2d": v1["kp2d"],
            "cam_angvel": v1["cam_angvel"],
            "f_imgseq": v1["f_imgseq"],
            "mask": v1["mask"],
            "view_id": 0,
        }

        # ---- view3 (third camera, exo + ego) ----
        view3 = {
            "length": v3["length"],
            "smpl_params_c": v3["smpl_params_c_exo"],  # exo in third cam
            "smpl_params_w": v3["smpl_params_w_exo"],
            "R_c2gv": v3["R_c2gv"],
            "bbx_xys": v3["bbx_xys_exo"],
            "K_fullimg": v3["K_fullimg"],
            "kp2d": v3["kp2d_exo"],
            "cam_angvel": v3["cam_angvel"],
            "f_imgseq": v3["f_imgseq"],
            "mask": v3["mask_exo"],

            # ego extra
            "smpl_params_c_ego": v3["smpl_params_c_ego"],
            "smpl_params_w_ego": v3["smpl_params_w_ego"],
            "R_c2gv_ego": v3["R_c2gv_ego"],
            "bbx_xys_ego": v3["bbx_xys_ego"],
            "kp2d_ego": v3["kp2d_ego"],
            "cam_angvel_ego": v3["cam_angvel_ego"],
            "mask_ego": v3["mask_ego"],
            "view_id": 1,
        }

        return {"view1": view1, "view3": view3}


group_name = "train_datasets/egobody"
node_v1 = builds(EgoBodyPairedDataset, populate_full_signature=True)
MainStore.store(name="paired_v1", node=node_v1, group=group_name)