import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from hmr4d import PROJ_ROOT


_ROOT_BUFFERS = None


def _get_root_buffers(device, dtype, num_betas):
    global _ROOT_BUFFERS
    if _ROOT_BUFFERS is None:
        path = PROJ_ROOT / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
        data = np.load(path, allow_pickle=True)
        shapedirs = torch.from_numpy(np.asarray(data["shapedirs"][:, :, :10])).float()
        v_template = torch.from_numpy(np.asarray(data["v_template"])).float()
        J_regressor = torch.from_numpy(np.asarray(data["J_regressor"][:22])).float()
        J_template = J_regressor @ v_template
        J_shapedirs = torch.einsum("jv,vcd->jcd", J_regressor, shapedirs)
        _ROOT_BUFFERS = (J_template[0], J_shapedirs[0])
    J0_template, J0_shapedirs = _ROOT_BUFFERS
    return J0_template.to(device=device, dtype=dtype), J0_shapedirs[:, :num_betas].to(device=device, dtype=dtype)


def get_smplx_root_offset(betas):
    if betas is None:
        return 0.0
    num_betas = min(int(betas.shape[-1]), 10)
    betas = betas[..., :num_betas]
    J0_template, J0_shapedirs = _get_root_buffers(betas.device, betas.dtype, num_betas)
    return J0_template + torch.einsum("...d,cd->...c", betas, J0_shapedirs)


def transform_smpl_root(global_orient, transl, T_old_to_new, betas):
    """Transform SMPL-X root parameters by a rigid coordinate transform.

    `transl` in SMPL-X is coupled with the shaped root-joint offset; rotating the
    coordinate frame therefore needs the same pivot compensation used by GVHMR's
    camera/root utilities: t' = R @ (t + J0(beta)) + T - J0(beta).
    """
    R = T_old_to_new[..., :3, :3].to(device=transl.device, dtype=transl.dtype)
    t = T_old_to_new[..., :3, 3].to(device=transl.device, dtype=transl.dtype)
    R_root = axis_angle_to_matrix(global_orient)
    R_new = R @ R_root
    offset = get_smplx_root_offset(betas).to(device=transl.device, dtype=transl.dtype)
    transl_new = torch.einsum("...ij,...j->...i", R, transl + offset) + t - offset
    return matrix_to_axis_angle(R_new), transl_new


def transform_smpl_root_to_local(global_orient_w, transl_w, T_w_local, betas):
    T_local_w = torch.zeros_like(T_w_local)
    R = T_w_local[..., :3, :3]
    t = T_w_local[..., :3, 3]
    T_local_w[..., :3, :3] = R.mT
    T_local_w[..., :3, 3] = -(R.mT @ t[..., None]).squeeze(-1)
    T_local_w[..., 3, 3] = 1
    return transform_smpl_root(global_orient_w, transl_w, T_local_w, betas)
