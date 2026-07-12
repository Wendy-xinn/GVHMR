import torch
import numpy as np
import pandas as pd
from pathlib import Path
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from hmr4d.utils.smpl_root_transform import transform_smpl_root

from hmr4d.configs import MainStore, builds
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.utils.geo.hmr_global import get_R_c2gv
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len
from hmr4d.utils.pylogger import Log


def _repeat_nested(data, max_len: int):
    if isinstance(data, dict):
        return {k: _repeat_nested(v, max_len) for k, v in data.items()}
    if torch.is_tensor(data) and data.ndim > 0:
        return repeat_to_max_len(data, max_len)
    return data


def _slice_nested(data, sl):
    if isinstance(data, dict):
        return {k: _slice_nested(v, sl) for k, v in data.items()}
    if torch.is_tensor(data) and data.ndim > 0:
        return data[sl].clone()
    if isinstance(data, list):
        return data[sl]
    return data


def _repeat_list_to_max_len(items, max_len: int):
    if len(items) == 0:
        return items
    if len(items) >= max_len:
        return items[:max_len]
    return items + [items[-1]] * (max_len - len(items))


def _transform_smpl(smpl_params_c, T_c2w):
    out = {k: v.clone() for k, v in smpl_params_c.items()}
    out['global_orient'], out['transl'] = transform_smpl_root(
        smpl_params_c['global_orient'], smpl_params_c['transl'], T_c2w, smpl_params_c.get('betas')
    )
    return out


def _world_to_yup_matrix(dtype=torch.float32):
    # EgoBody Kinect/PV-style world uses OpenCV-like y-down, z-forward axes.
    # Rotate 180 deg around x to get a proper right-handed y-up training world.
    T = torch.eye(4, dtype=dtype)
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    return T


def _kinect_gravity_yup_matrix(calib, dtype=torch.float32):
    """Kinect-origin world whose +y is aligned to scene gravity.

    EgoBody's kinect12_to_scene calibration uses a y-up scene frame for these
    recordings. We keep only its rotation so the Kinect/exo camera remains at
    the origin, while its real pitch/roll/yaw relative to gravity is preserved.
    """
    T = torch.eye(4, dtype=dtype)
    T[:3, :3] = calib['T_kinect12_to_scene'][:3, :3].to(dtype)
    return T


def _train_world_transform(calib, world_coord, dtype=torch.float32):
    if world_coord == 'kinect12':
        return _kinect_gravity_yup_matrix(calib, dtype=dtype)
    return _world_to_yup_matrix(dtype=dtype)


def _opencv_camera_to_yup_local_matrix(dtype=torch.float32):
    # Camera image/SMPL incam coordinates are OpenCV-like: +x right, +y down, +z forward.
    # Viser/GVHMR camera-local convention is +x right, +y up, -z forward.
    T = torch.eye(4, dtype=dtype)
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    return T


def _transform_smpl_world(smpl_params_w, T_raw_to_train):
    out = {k: v.clone() for k, v in smpl_params_w.items()}
    out['global_orient'], out['transl'] = transform_smpl_root(
        smpl_params_w['global_orient'], smpl_params_w['transl'], T_raw_to_train, smpl_params_w.get('betas')
    )
    return out


def _transform_T_world(T_world_obj, T_raw_to_train):
    return T_raw_to_train.to(T_world_obj.dtype) @ T_world_obj


class EgoBodyEgoExoV1Dataset(ImgfeatMotionDatasetBase):
    """Dataset for tools/egobody/preprocess_egoexo.py outputs."""

    def __init__(
        self,
        root='/public/home/wenxin/egobody',
        output_root='/public/home/wenxin/egobody/output/egoexo_v1',
        split='train',
        motion_frames=120,
        world_coord='holo',
        overfit_single_sample=False,
        require_features=True,
    ):
        self.root = Path(root)
        self.output_root = Path(output_root)
        self.split = split
        self.motion_frames = motion_frames
        self.world_coord = world_coord
        self.overfit_single_sample = overfit_single_sample
        self.require_features = require_features
        self._seq_lens = {}
        self._rec_dirs = {}
        super().__init__()

    def _load_dataset(self):
        df = pd.read_csv(self.root / 'data_splits.csv')
        if self.split not in df.columns:
            raise ValueError(f'split={self.split} not in data_splits.csv')
        recordings = [str(x) for x in df[self.split].dropna().tolist()]
        valid = []
        for rec in recordings:
            rec_dir = self.output_root / rec
            manifest_path = rec_dir / 'manifest.pt'
            if not manifest_path.exists():
                Log.warning(f'[EgoBodyEgoExoV1] missing manifest: {manifest_path}')
                continue
            if self.require_features and not (rec_dir / 'features_exo_hmr2.pt').exists():
                Log.warning(f'[EgoBodyEgoExoV1] missing exo features: {rec_dir}')
                continue
            if self.require_features and not (rec_dir / 'features_ego_full_hmr2.pt').exists():
                Log.warning(f'[EgoBodyEgoExoV1] missing ego full features: {rec_dir}')
                continue
            if self.require_features and not (rec_dir / 'features_ego_body_hmr2.pt').exists():
                Log.warning(f'[EgoBodyEgoExoV1] missing ego body features: {rec_dir}')
                continue
            manifest = torch.load(manifest_path, map_location='cpu')
            L = int(manifest['length'])
            if L <= 0:
                continue
            valid.append(rec)
            self._rec_dirs[rec] = rec_dir
            self._seq_lens[rec] = L
        self.recordings = valid
        Log.info(f'[EgoBodyEgoExoV1] split={self.split}, recordings={len(self.recordings)}')

    def _get_idx2meta(self):
        self.idx2meta = []
        for rec in self.recordings:
            L = self._seq_lens[rec]
            n = max(L // self.motion_frames, 1)
            self.idx2meta.extend([(rec, i) for i in range(n)])
            if self.overfit_single_sample:
                self.idx2meta = [(rec, 0)]
                return

    def _target_T_c2w(self, calib):
        if self.world_coord == 'holo':
            return calib['T_kinect12_to_holo']
        if self.world_coord == 'scene':
            return calib['T_kinect12_to_scene']
        if self.world_coord == 'kinect12':
            return torch.eye(4)
        raise ValueError(f'Unknown world_coord={self.world_coord}')

    def _target_T_holo_to_world(self, calib):
        if self.world_coord == 'holo':
            return torch.eye(4)
        if self.world_coord == 'kinect12':
            return calib['T_holo_to_kinect12']
        if self.world_coord == 'scene':
            return calib['T_kinect12_to_scene'] @ calib['T_holo_to_kinect12']
        raise ValueError(f'Unknown world_coord={self.world_coord}')

    def _load_data(self, idx):
        rec, sample_idx = self.idx2meta[idx]
        rec_dir = self._rec_dirs[rec]
        manifest = torch.load(rec_dir / 'manifest.pt', map_location='cpu')
        smplx_gt = torch.load(rec_dir / 'smplx_gt.pt', map_location='cpu')
        traj = torch.load(rec_dir / 'camera_head_traj.pt', map_location='cpu')
        calib = torch.load(rec_dir / 'calibration.pt', map_location='cpu')
        feat_exo = torch.load(rec_dir / 'features_exo_hmr2.pt', map_location='cpu') if (rec_dir / 'features_exo_hmr2.pt').exists() else None
        feat_ego = torch.load(rec_dir / 'features_ego_full_hmr2.pt', map_location='cpu') if (rec_dir / 'features_ego_full_hmr2.pt').exists() else None
        feat_ego_body = torch.load(rec_dir / 'features_ego_body_hmr2.pt', map_location='cpu') if (rec_dir / 'features_ego_body_hmr2.pt').exists() else None

        L = int(manifest['length'])
        if self.motion_frames >= L:
            start, end = 0, L
        elif self.split == 'train' and not self.overfit_single_sample:
            start = np.random.randint(0, L - self.motion_frames + 1)
            end = start + self.motion_frames
        else:
            start = min(sample_idx * self.motion_frames, max(L - self.motion_frames, 0))
            end = start + self.motion_frames
        sl = slice(start, end)

        T_c2w = self._target_T_c2w(calib)
        exo_c = _slice_nested(smplx_gt['exo']['kinect12'], sl)
        ego_c = _slice_nested(smplx_gt['ego']['kinect12'], sl)
        exo_w_raw = _transform_smpl(exo_c, T_c2w)
        ego_w_raw = _transform_smpl(ego_c, T_c2w)
        T_raw_to_train = _train_world_transform(calib, self.world_coord, dtype=T_c2w.dtype)
        exo_w = _transform_smpl_world(exo_w_raw, T_raw_to_train)
        ego_w = _transform_smpl_world(ego_w_raw, T_raw_to_train)

        f_exo = feat_exo['features'][sl].float() if feat_exo is not None else torch.zeros((end - start, 1024))
        f_ego = feat_ego['features'][sl].float() if feat_ego is not None else torch.zeros((end - start, 1024))
        f_ego_body_raw = feat_ego_body['features'][sl].float() if feat_ego_body is not None else f_ego.clone()
        ego_body_bbox_valid = manifest['mask'].get('ego_body_bbox', torch.ones(L, dtype=torch.bool))[sl].bool()
        f_ego_body = torch.where(ego_body_bbox_valid[:, None], f_ego_body_raw, f_ego)
        valid = manifest['mask']['valid'][sl].bool()
        exo_valid = manifest['mask'].get('exo_gt', valid)[sl].bool() & valid
        ego_valid = manifest['mask'].get('ego_gt', valid)[sl].bool() & valid

        R_train_w2c = T_raw_to_train[:3, :3].T
        R_c2gv = get_R_c2gv(R_train_w2c, axis_gravity_in_w=[0, -1, 0]).unsqueeze(0).repeat(end - start, 1, 1)

        T_holo_to_world_raw = self._target_T_holo_to_world(calib).float()
        T_holo_to_world = T_raw_to_train.float() @ T_holo_to_world_raw
        T_holo_head = traj['T_holo_head'][sl].float()
        T_holo_cpf = traj['T_holo_cpf'][sl].float()
        T_holo_pv = traj['T_holo_pv'][sl].float()
        T_world_head_raw = T_holo_to_world_raw @ T_holo_head
        T_world_cpf_raw = T_holo_to_world_raw @ T_holo_cpf
        T_world_pv_raw = T_holo_to_world_raw @ T_holo_pv
        ego_cond = {
            'T_holo_head': T_holo_head,
            'T_holo_cpf': T_holo_cpf,
            'T_holo_pv': T_holo_pv,
            'T_world_head': _transform_T_world(T_world_head_raw, T_raw_to_train),
            'T_world_cpf': _transform_T_world(T_world_cpf_raw, T_raw_to_train),
            'T_world_pv': _transform_T_world(T_world_pv_raw, T_raw_to_train),
            'T_world_head_raw': T_world_head_raw,
            'T_world_cpf_raw': T_world_cpf_raw,
            'T_world_pv_raw': T_world_pv_raw,
            'T_raw_to_train_world': T_raw_to_train.float().unsqueeze(0).repeat(end - start, 1, 1),
            'head_angvel': traj['head_angvel'][sl].float(),
            'head_valid': traj['head_valid'][sl].bool(),
            'gaze_valid': traj['gaze_valid'][sl].bool(),
            'left_hand_summary_holo': traj['left_hand_summary_holo'][sl].float(),
            'right_hand_summary_holo': traj['right_hand_summary_holo'][sl].float(),
            'left_hand_valid': traj['left_hand_valid'][sl].bool(),
            'right_hand_valid': traj['right_hand_valid'][sl].bool(),
        }

        return {
            'meta': {
                'data_name': 'egobody_egoexo_v1',
                'recording': rec,
                'start': int(start),
                'end': int(end),
                'world_coord': 'gvhmr_yup',
                'raw_world_coord': self.world_coord,
            },
            'length': end - start,
            'frame_ids': manifest['frame_ids'][sl].long(),
            'K_fullimg': manifest['K_exo'][sl].float(),
            # Generic camera trajectory for the current exo input stream. It maps
            # y-up camera-local coordinates (+x right, +y up, -z forward) to the
            # gravity-aligned world. For EgoBody Kinect it is static; dynamic exo
            # datasets can fill this per frame.
            'T_world_cam': (T_raw_to_train @ _opencv_camera_to_yup_local_matrix(dtype=T_raw_to_train.dtype)).float().unsqueeze(0).repeat(end - start, 1, 1),
            'T_world_exo_cam': (T_raw_to_train @ _opencv_camera_to_yup_local_matrix(dtype=T_raw_to_train.dtype)).float().unsqueeze(0).repeat(end - start, 1, 1),
            'K_ego': manifest.get('K_ego', manifest['K_exo'])[sl].float(),
            'R_c2gv': R_c2gv.float(),
            'cam_angvel': torch.zeros((end - start, 6), dtype=torch.float32),
            'imgname': manifest.get('pv_img_paths', manifest.get('kinect_img_paths', []))[sl],
            'ego_imgname': manifest.get('pv_img_paths', [])[sl],
            'exo_imgname': manifest.get('kinect_img_paths', [])[sl],
            'exo': {
                'smpl_params_c': exo_c,
                'smpl_params_w': exo_w,
                'smpl_params_w_raw': exo_w_raw,
                'bbx_xys': manifest['bbx_exo_gt'][sl].float(),
                'kp2d': manifest['kp2d_exo_gt'][sl].float(),
                'f_imgseq': f_exo,
                'valid': exo_valid,
            },
            'ego': {
                'smpl_params_c': ego_c,
                'smpl_params_w': ego_w,
                'smpl_params_w_raw': ego_w_raw,
                'bbx_xys': manifest['bbx_ego_in_exo_gt'][sl].float(),
                'kp2d': manifest['kp2d_ego_in_exo_gt'][sl].float(),
                'f_imgseq': f_ego,
                'f_body_imgseq': f_ego_body,
                'bbx_body_xys': manifest.get('bbx_ego_body_gt', manifest['bbx_ego_full'])[sl].float(),
                'kp2d_body': manifest.get('kp2d_ego_body_gt', manifest['kp2d_ego_in_exo_gt'])[sl].float(),
                'valid': ego_valid,
            },
            'ego_cond': ego_cond,
            'mask': {
                'valid': valid,
                'exo_valid': exo_valid,
                'ego_valid': ego_valid,
                'head': ego_cond['head_valid'],
                'spv_incam_only': torch.zeros((end - start,), dtype=torch.bool),
                'vitpose': torch.zeros((end - start,), dtype=torch.bool),
                'bbx_xys': torch.ones((end - start,), dtype=torch.bool),
                'f_imgseq': torch.full((end - start,), feat_exo is not None, dtype=torch.bool),
                'ego_body_f_imgseq': torch.full((end - start,), feat_ego_body is not None, dtype=torch.bool) & ego_body_bbox_valid,
            },
        }

    def _process_data(self, data, idx):
        length = data['length']
        max_len = self.motion_frames
        out = {**data, 'length': length}
        for key in ('frame_ids', 'K_fullimg', 'K_ego', 'R_c2gv', 'cam_angvel', 'T_world_cam', 'T_world_exo_cam'):
            out[key] = repeat_to_max_len(out[key], max_len)
        for key in ('imgname', 'ego_imgname', 'exo_imgname'):
            if key in out:
                out[key] = _repeat_list_to_max_len(out[key], max_len)
        for key in ('exo', 'ego', 'ego_cond'):
            out[key] = _repeat_nested(out[key], max_len)
        out['mask'] = _repeat_nested(out['mask'], max_len)
        out['mask']['valid'] = get_valid_mask(max_len, length)
        out['mask']['exo_valid'] = out['mask']['exo_valid'] & out['mask']['valid']
        out['mask']['ego_valid'] = out['mask']['ego_valid'] & out['mask']['valid']
        out['mask']['head'] = out['mask']['head'] & out['mask']['valid']
        return out


group_name = 'train_datasets/egobody'
node = builds(EgoBodyEgoExoV1Dataset, populate_full_signature=True)
MainStore.store(name='egoexo_v1_preprocessed', node=node, group=group_name)
