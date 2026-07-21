import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from hmr4d.utils.net_utils import gaussian_smooth

from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
from hmr4d.model.gvhmr.utils.postprocess import (
    pp_static_joint,
    pp_static_joint_cam,
    pp_static_joint_ego,
    pp_static_joint_cam_ego,
    process_ik,
)
from hmr4d.model.gvhmr.utils import stats_compose
from hmr4d.utils.smpl_root_transform import transform_smpl_root, transform_smpl_root_to_local

SMPL_BODY_KEYS = ("body_pose", "betas", "global_orient", "transl")


def _body_smpl_params(params):
    return {k: v for k, v in params.items() if k in SMPL_BODY_KEYS}


def _compose_teacher_residual(base_params, residual, pose_scale=0.1, root_rot_scale=0.1, root_trans_scale=0.25):
    """Compose small SO(3) corrections onto an exo SMPL base motion."""
    B, L = residual.shape[:2]
    base_body_R = axis_angle_to_matrix(base_params["body_pose"].reshape(B, L, 21, 3))
    delta_body_R = axis_angle_to_matrix((residual[..., :63] * pose_scale).reshape(B, L, 21, 3))
    body_pose = matrix_to_axis_angle(delta_body_R @ base_body_R).flatten(-2)
    base_root_R = axis_angle_to_matrix(base_params["global_orient"])
    delta_root_R = axis_angle_to_matrix(residual[..., 63:66] * root_rot_scale)
    global_orient = matrix_to_axis_angle(delta_root_R @ base_root_R)
    transl = base_params["transl"] + residual[..., 66:69] * root_trans_scale
    return {
        "body_pose": body_pose,
        "betas": base_params["betas"],
        "global_orient": global_orient,
        "transl": transl,
    }


def safe_masked_mean(loss, mask):
    """Masked mean with the original mask denominator, while dropping NaN/Inf values."""
    mask = mask.to(device=loss.device, dtype=loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    loss = loss * mask
    num_valid = mask.sum()
    return loss.sum() / torch.clamp(num_valid, min=1)

def _loss_weight(weights, key, fallback_key=None, default=0.0):
    if key in weights:
        return float(weights.get(key, default))
    if fallback_key is not None and fallback_key in weights:
        return float(weights.get(fallback_key, default))
    return float(default)


def _frame_visibility_weight_from_kp2d(kp2d, target_ndim, eps=0.05):
    if kp2d is None or kp2d.shape[-1] < 3:
        return None
    vis = kp2d[..., 2].clamp(0.0, 1.0).mean(dim=-1)
    while vis.ndim < target_ndim:
        vis = vis.unsqueeze(-1)
    return vis.clamp_min(eps)

from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from hmr4d.utils.geo.hmr_cam import compute_bbox_info_bedlam, compute_transl_full_cam, get_a_pred_cam, normalize_kp2d, project_to_bi01
from hmr4d.utils.geo.hmr_global import (
    get_local_transl_vel,
    rollout_local_transl_vel,
    get_static_joint_mask,
    get_tgtcoord_rootparam,
)
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.smplx_utils import make_smplx


def _get_branch_mode(args):
    mode = args.get("branch_mode", "both")
    if mode not in ("exo", "ego", "both", "auto"):
        raise ValueError(f"branch_mode must be one of exo/ego/both/auto, got {mode}")
    return mode


def _effective_branch_mode(inputs, branch_mode):
    if branch_mode == "auto":
        return "ego" if "exo" in inputs and "ego" in inputs else "exo"
    if branch_mode in ("ego", "both") and not ("exo" in inputs and "ego" in inputs):
        return "exo"
    return branch_mode


def _opencv_cam_to_yup_matrix(device, dtype):
    C = torch.eye(4, device=device, dtype=dtype)
    C[:3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype))
    return C


def _R_c2gv_from_T_world_pv(T_world_pv):
    C = _opencv_cam_to_yup_matrix(T_world_pv.device, T_world_pv.dtype)[:3, :3]
    return T_world_pv[..., :3, :3] @ C


@autocast(enabled=False)
def _smpl_params_world_to_incam(smpl_params_w, T_world_pv):
    out_dtype = smpl_params_w["transl"].dtype
    T_world_pv_f = T_world_pv.float()
    C = _opencv_cam_to_yup_matrix(T_world_pv_f.device, T_world_pv_f.dtype)
    while C.ndim < T_world_pv_f.ndim:
        C = C.unsqueeze(0)
    T_world_opencv_cam = T_world_pv_f @ C
    T_opencv_cam_world = torch.linalg.inv(T_world_opencv_cam)
    betas = smpl_params_w.get("betas")
    global_orient_c, transl_c = transform_smpl_root(
        smpl_params_w["global_orient"].float(),
        smpl_params_w["transl"].float(),
        T_opencv_cam_world,
        betas.float() if betas is not None else None,
    )
    return {
        "body_pose": smpl_params_w["body_pose"],
        "betas": smpl_params_w["betas"],
        "global_orient": global_orient_c.to(dtype=out_dtype),
        "transl": transl_c.to(dtype=out_dtype),
    }


def _split_paired_inputs(inputs, branch_mode="both", input_role=None):
    """Map paired ego/exo batches to the flat legacy fields for one routed branch.

    branch_mode controls which observation/image stream feeds the shared
    transformer. Loss gating is handled later in Pipeline.forward.
    """
    if "exo" not in inputs or "ego" not in inputs:
        return inputs, None, None

    exo = inputs["exo"]
    ego = inputs["ego"]
    flat = dict(inputs)

    route_role = input_role or ("ego" if branch_mode == "ego" else "exo")
    flat["_input_role"] = route_role

    if route_role == "ego":
        flat.update(
            {
                "smpl_params_c": ego.get("smpl_params_c"),
                "smpl_params_w": ego.get("smpl_params_w"),
                "bbx_xys": ego.get("bbx_body_xys", ego.get("bbx_xys")),
                "f_imgseq": ego.get("f_body_imgseq", ego.get("f_imgseq")),
                "kp2d": ego.get("kp2d_body", ego.get("kp2d")),
                "K_fullimg": inputs.get("K_ego", inputs.get("K_fullimg")),
            }
        )
        ego_cond_for_motion = inputs.get("ego_cond", {})
        if isinstance(ego_cond_for_motion, dict):
            if ego_cond_for_motion.get("pv_cam_angvel", None) is not None:
                flat["cam_angvel"] = ego_cond_for_motion["pv_cam_angvel"]
            elif ego_cond_for_motion.get("head_angvel", None) is not None:
                flat["cam_angvel"] = ego_cond_for_motion["head_angvel"]
            if ego_cond_for_motion.get("pv_cam_trans_vel", None) is not None:
                flat["cam_trans_vel"] = ego_cond_for_motion["pv_cam_trans_vel"]
            if ego_cond_for_motion.get("pv_gravity_dir_cam", None) is not None:
                flat["gravity_dir_cam"] = ego_cond_for_motion["pv_gravity_dir_cam"]
            if ego_cond_for_motion.get("T_world_pv", None) is not None:
                flat["R_c2gv"] = _R_c2gv_from_T_world_pv(
                    ego_cond_for_motion["T_world_pv"].to(device=flat["bbx_xys"].device, dtype=flat["bbx_xys"].dtype)
                )
    else:
        flat.update(
            {
                "smpl_params_c": exo.get("smpl_params_c"),
                "smpl_params_w": exo.get("smpl_params_w"),
                "bbx_xys": exo.get("bbx_xys"),
                "f_imgseq": exo.get("f_imgseq"),
                "kp2d": exo.get("kp2d"),
                "K_fullimg": inputs.get("K_fullimg"),
            }
        )

    if "interactee_smpl_params_c" not in flat and flat.get("smpl_params_c") is not None:
        flat["interactee_smpl_params_c"] = flat["smpl_params_c"]
    if "interactee_smpl_params_w" not in flat and flat.get("smpl_params_w") is not None:
        flat["interactee_smpl_params_w"] = flat["smpl_params_w"]
    return flat, ego, inputs.get("ego_cond")




def _target_inputs_for_role(inputs, role):
    """Return a flat target view for one physical role.

    For cross-view teacher with ego as target view, the partner/person source is
    observed in exo but predicted in ego camera-motion coordinates, so its world
    GT is first transformed into the ego target world and then into the ego PV
    camera for the incam fields.
    """
    if role not in inputs:
        return inputs
    role_inputs = inputs[role]
    out = dict(inputs)
    mask_dict = dict(inputs.get("mask", {}))
    role_valid = mask_dict.get(f"{role}_valid", mask_dict.get("valid", None))
    if role_valid is not None:
        mask_dict["valid"] = role_valid
        out["mask"] = mask_dict

    target_ego_view = role == "exo" and inputs.get("_input_role", None) == "ego" and isinstance(inputs.get("ego_cond"), dict)
    if target_ego_view and role_inputs.get("smpl_params_w") is not None:
        ego_cond = inputs["ego_cond"]
        smpl_w = _smpl_params_to_ego_world(role_inputs["smpl_params_w"], ego_cond)
        T_world_pv = ego_cond.get("T_world_pv", None)
        if T_world_pv is not None:
            T_world_pv = T_world_pv.to(device=smpl_w["transl"].device, dtype=smpl_w["transl"].dtype)
            out["smpl_params_c"] = _smpl_params_world_to_incam(smpl_w, T_world_pv)
            out["R_c2gv"] = _R_c2gv_from_T_world_pv(T_world_pv)
        else:
            out["smpl_params_c"] = role_inputs.get("smpl_params_c")
        out["smpl_params_w"] = smpl_w
        out["interactee_smpl_params_c"] = out["smpl_params_c"]
        out["interactee_smpl_params_w"] = out["smpl_params_w"]
        ego_inputs = inputs.get("ego", {})
        out["bbx_xys"] = ego_inputs.get("bbx_body_xys", ego_inputs.get("bbx_xys", out.get("bbx_xys")))
        out["kp2d"] = ego_inputs.get("kp2d_body", ego_inputs.get("kp2d", out.get("kp2d")))
        out["f_imgseq"] = ego_inputs.get("f_body_imgseq", ego_inputs.get("f_imgseq", out.get("f_imgseq")))
        if inputs.get("K_ego", None) is not None:
            out["K_fullimg"] = inputs["K_ego"]
        return out

    if role_inputs.get("smpl_params_c") is not None:
        out["smpl_params_c"] = role_inputs["smpl_params_c"]
        out["interactee_smpl_params_c"] = role_inputs["smpl_params_c"]
    if role_inputs.get("smpl_params_w") is not None:
        out["smpl_params_w"] = role_inputs["smpl_params_w"]
        out["interactee_smpl_params_w"] = role_inputs["smpl_params_w"]
    if role_inputs.get("bbx_xys") is not None:
        out["bbx_xys"] = role_inputs["bbx_xys"]
    if role_inputs.get("kp2d") is not None:
        out["kp2d"] = role_inputs["kp2d"]
    if role_inputs.get("f_imgseq") is not None:
        out["f_imgseq"] = role_inputs["f_imgseq"]
    if role == "exo" and inputs.get("K_exo", None) is not None:
        out["K_fullimg"] = inputs["K_exo"]
    elif role == "ego" and inputs.get("K_ego", None) is not None:
        out["K_fullimg"] = inputs["K_ego"]
    return out


def _target_inputs_with_gt_cache(inputs, role, endecoder):
    out = _target_inputs_for_role(inputs, role)
    if out.get("interactee_smpl_params_c", None) is None:
        return out
    with torch.no_grad():
        gt_verts437, gt_j3d = endecoder.smplx_model(**_body_smpl_params(out["interactee_smpl_params_c"]))
        root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
    out["gt_j3d"] = gt_j3d
    out["gt_cr_coco17"] = gt_j3d - root_
    out["gt_c_verts437"] = gt_verts437
    out["gt_cr_verts437"] = gt_verts437 - root_
    return out


def _mask_for_role(inputs, role):
    mask = inputs["mask"].get(f"{role}_valid", inputs["mask"]["valid"])
    return mask & inputs["mask"].get("valid", mask)


def _targets_for_ego_world(inputs):
    ego_targets = _target_inputs_for_role(inputs, "ego")
    ego_cond = inputs.get("ego_cond", {})
    T_abs_to_ego_world = ego_cond.get("T_abs_to_ego_world", None) if isinstance(ego_cond, dict) else None
    if T_abs_to_ego_world is None:
        return ego_targets
    out = dict(ego_targets)
    smpl_w = ego_targets["smpl_params_w"]
    global_orient, transl = transform_smpl_root(
        smpl_w["global_orient"], smpl_w["transl"], T_abs_to_ego_world, smpl_w.get("betas")
    )
    out["smpl_params_w"] = {**smpl_w, "global_orient": global_orient, "transl": transl}
    out["interactee_smpl_params_w"] = out["smpl_params_w"]
    return out


def _smpl_params_to_ego_world(params_w, ego_cond):
    if params_w is None or not isinstance(ego_cond, dict) or "T_abs_to_ego_world" not in ego_cond:
        return params_w
    global_orient, transl = transform_smpl_root(
        params_w["global_orient"], params_w["transl"], ego_cond["T_abs_to_ego_world"], params_w.get("betas")
    )
    return {**params_w, "global_orient": global_orient, "transl": transl}



def _invert_T(T):
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R.mT
    T_inv[..., :3, 3] = -(R.mT @ t[..., None]).squeeze(-1)
    T_inv[..., 3, 3] = 1
    return T_inv


def _transform_root_to_local(global_orient_w, transl_w, T_w_local, betas=None):
    return transform_smpl_root_to_local(global_orient_w, transl_w, T_w_local, betas)


def _transform_root_to_world(global_orient_local, transl_local, T_w_local, betas=None):
    return transform_smpl_root(global_orient_local, transl_local, T_w_local, betas)


def _transform_root_from_opencv_cam_to_world(global_orient_c, transl_c, T_w_pv, betas=None):
    """Lift OpenCV-camera SMPL root parameters to dataset/canonical world using PV pose."""
    C = torch.eye(4, device=transl_c.device, dtype=transl_c.dtype)
    C[:3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=transl_c.device, dtype=transl_c.dtype))
    while C.ndim < T_w_pv.ndim:
        C = C.unsqueeze(0)
    T_w_c = T_w_pv @ C
    return transform_smpl_root(global_orient_c, transl_c, T_w_c, betas)


def _smpl_params_incam_to_world(smpl_params_c, T_w_pv):
    global_orient_w, transl_w = _transform_root_from_opencv_cam_to_world(
        smpl_params_c["global_orient"], smpl_params_c["transl"], T_w_pv, smpl_params_c.get("betas")
    )
    return {
        "body_pose": smpl_params_c["body_pose"],
        "betas": smpl_params_c["betas"],
        "global_orient": global_orient_w,
        "transl": transl_w,
    }


def _identity_T_like(transl):
    T = torch.eye(4, device=transl.device, dtype=transl.dtype)
    return T.expand(*transl.shape[:-1], 4, 4).clone()



def _identity_cam_angvel(B, L, device, dtype):
    eye = torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 3, 3).repeat(B, L, 1, 1)
    return matrix_to_rotation_6d(eye)


def _gravity_aligned_cam0_transform(T_abs_pv0):
    """Return raw/train-world -> cam0-rooted, y-up transform using cam0 yaw only."""
    device, dtype = T_abs_pv0.device, T_abs_pv0.dtype
    R_c2abs = T_abs_pv0[..., :3, :3]
    t_abs = T_abs_pv0[..., :3, 3]
    forward_abs = -R_c2abs[..., :, 2]
    forward_xz = forward_abs.clone()
    forward_xz[..., 1] = 0.0
    norm = forward_xz.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(forward_xz)
    fallback[..., 2] = -1.0
    forward_xz = torch.where(norm > 1e-6, forward_xz / norm.clamp(min=1e-6), fallback)
    up = torch.zeros_like(forward_xz)
    up[..., 1] = 1.0
    # Camera convention here is +x right, +y up, -z forward. For a right-handed
    # yaw frame, right = forward x up and z_back = -forward.
    right_xz = torch.cross(forward_xz, up, dim=-1)
    right_xz = right_xz / right_xz.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # Columns are camera-yaw frame axes expressed in the absolute y-up world.
    R_cam0_yaw_to_abs = torch.stack([right_xz, up, -forward_xz], dim=-1)
    R_abs_to_cam0_yaw = R_cam0_yaw_to_abs.mT
    T_abs_to_cam0_yaw = torch.zeros_like(T_abs_pv0)
    T_abs_to_cam0_yaw[..., :3, :3] = R_abs_to_cam0_yaw
    T_abs_to_cam0_yaw[..., :3, 3] = -(R_abs_to_cam0_yaw @ t_abs[..., None]).squeeze(-1)
    T_abs_to_cam0_yaw[..., 3, 3] = 1.0
    return T_abs_to_cam0_yaw, R_abs_to_cam0_yaw


def _rollout_camera_from_angvel(inputs, base_ego_cond=None):
    bbx = inputs["bbx_xys"]
    B, L = bbx.shape[:2]
    device, dtype = bbx.device, bbx.dtype
    cam_angvel = inputs.get("cam_angvel", None)
    if cam_angvel is None:
        cam_angvel = _identity_cam_angvel(B, L, device, dtype)
    R_rel = rotation_6d_to_matrix(cam_angvel.to(device=device, dtype=dtype))
    is_I = matrix_to_axis_angle(R_rel).norm(dim=-1) < 1e-5
    if is_I.any():
        eye3 = torch.eye(3, device=device, dtype=dtype)
        R_rel = R_rel.clone()
        R_rel[is_I] = eye3

    T_abs_pv0 = None
    if isinstance(base_ego_cond, dict):
        T_abs_pv0 = base_ego_cond.get("T_world_pv", None)
    if T_abs_pv0 is not None:
        T_abs_pv0 = T_abs_pv0.to(device=device, dtype=dtype)
        if T_abs_pv0.ndim == 3:
            T_abs_pv0 = T_abs_pv0.unsqueeze(0)
        if T_abs_pv0.shape[0] == 1 and B > 1:
            T_abs_pv0 = T_abs_pv0.expand(B, -1, -1, -1)
        T_abs_pv0 = T_abs_pv0[:, :1]
    else:
        T_abs_pv0 = torch.eye(4, device=device, dtype=dtype).reshape(1, 1, 4, 4).repeat(B, 1, 1, 1)

    T_abs_to_ego0, R_abs_to_ego0 = _gravity_aligned_cam0_transform(T_abs_pv0)
    R_c2abs0 = T_abs_pv0[:, 0, :3, :3]
    R_c2w = [R_abs_to_ego0[:, 0] @ R_c2abs0]
    for i in range(1, L):
        # cam_angvel follows R_rel @ R_w2c[t] = R_w2c[t+1].
        # Therefore R_c2w[t+1] = R_c2w[t] @ R_rel[t].T.
        R_c2w.append(R_c2w[-1] @ R_rel[:, i].mT)
    R_c2w = torch.stack(R_c2w, dim=1)

    cam_trans_vel = inputs.get("cam_trans_vel", None)
    if cam_trans_vel is None:
        cam_trans_vel = torch.zeros(B, L, 3, device=device, dtype=dtype)
    else:
        cam_trans_vel = cam_trans_vel.to(device=device, dtype=dtype)
        if cam_trans_vel.shape[0] == 1 and B > 1:
            cam_trans_vel = cam_trans_vel.expand(B, -1, -1)
        cam_trans_vel = cam_trans_vel[:, :L]
    # cam_trans_vel is camera-local delta t[t+1]-t[t]. Roll it out into the
    # cam0-rooted gravity/yaw-aligned world so T_world_cpf actually moves with
    # the input camera trajectory instead of staying at the origin.
    cam_vel_w = torch.einsum("blij,blj->bli", R_c2w, cam_trans_vel)
    cam_delta = torch.cat([torch.zeros_like(cam_vel_w[:, :1]), cam_vel_w[:, :-1]], dim=1)
    cam_trans_w = torch.cumsum(cam_delta, dim=1)

    T_world_cam = torch.eye(4, device=device, dtype=dtype).reshape(1, 1, 4, 4).repeat(B, L, 1, 1)
    T_world_cam[:, :, :3, :3] = R_c2w
    T_world_cam[:, :, :3, 3] = cam_trans_w
    return T_world_cam, T_abs_to_ego0.expand(B, L, 4, 4).clone()


def _make_cam_angvel_cpf_ego_cond(inputs, base_ego_cond=None, cfg=None):
    """Build ego CPF trajectory from camera angular velocity plus fixed cam->CPF offsets."""
    cfg = cfg or {}
    bbx = inputs["bbx_xys"]
    B, L = bbx.shape[:2]
    device, dtype = bbx.device, bbx.dtype
    T_world_cam, T_abs_to_ego_world = _rollout_camera_from_angvel(inputs, base_ego_cond)

    if "head_camera_offset" in cfg:
        offset = torch.as_tensor(cfg["head_camera_offset"], device=device, dtype=dtype).reshape(3)
    else:
        offset = torch.tensor([0.0, float(cfg.get("height", 0.5)), float(cfg.get("back_offset", 0.6))], device=device, dtype=dtype)

    T_cam_head = torch.eye(4, device=device, dtype=dtype)
    T_cam_head[:3, 3] = offset
    T_cam_head = T_cam_head.reshape(1, 1, 4, 4).repeat(B, L, 1, 1)

    T_head_cpf = torch.eye(4, device=device, dtype=dtype)
    T_head_cpf[:3, :3] = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype))
    if "head_cpf_offset" in cfg:
        head_cpf_offset = torch.as_tensor(cfg["head_cpf_offset"], device=device, dtype=dtype)
        T_head_cpf[:3, 3] = head_cpf_offset.reshape(3)
    T_head_cpf = T_head_cpf.reshape(1, 1, 4, 4).repeat(B, L, 1, 1)

    T_world_head = T_world_cam @ T_cam_head
    T_world_cpf = T_world_head @ T_head_cpf
    cond = dict(base_ego_cond) if base_ego_cond is not None else {}
    cond["T_world_pv"] = T_world_cam
    cond["T_abs_to_ego_world"] = T_abs_to_ego_world
    cond["T_world_head"] = T_world_head
    cond["T_world_cpf"] = T_world_cpf
    cond["head_valid"] = torch.ones((B, L), device=device, dtype=torch.bool)
    cond["head_angvel"] = inputs["cam_angvel"].to(device=device, dtype=dtype)
    cond["cam_angvel_cpf"] = torch.ones((B, L), device=device, dtype=torch.bool)
    return cond

def _make_fixed_observer_ego_cond(inputs, base_ego_cond=None, cfg=None):
    """Create a virtual head/CPF trajectory for exo-camera input.

    Ego CPF decoding expects a body-centric head/CPF trajectory, not a generic
    camera pose. For fixed or moving exo cameras we synthesize a simple observer
    standing behind the camera, facing the camera forward direction.
    """
    cfg = cfg or {}
    bbx = inputs["bbx_xys"]
    B, L = bbx.shape[:2]
    device, dtype = bbx.device, bbx.dtype

    # Keep the virtual ego trajectory in the same camera-motion frame as the
    # active input. A missing cam_angvel means a fixed camera, represented as
    # identity relative rotations rather than an absolute exo camera pose.
    rollout_inputs = inputs
    if inputs.get("cam_angvel", None) is None:
        rollout_inputs = dict(inputs)
        rollout_inputs["cam_angvel"] = _identity_cam_angvel(B, L, device, dtype)
    T_world_cam, T_abs_to_ego_world = _rollout_camera_from_angvel(rollout_inputs, None)

    if "head_camera_offset" in cfg:
        offset = torch.as_tensor(cfg["head_camera_offset"], device=device, dtype=dtype)
        if offset.numel() != 3:
            raise ValueError("virtual_ego_from_exo.head_camera_offset must have 3 values: [x, y, z]")
        offset = offset.reshape(3)
    else:
        head_height = float(cfg.get("height", 1.6))
        back_offset = float(cfg.get("back_offset", 0.6))
        height_mode = cfg.get("height_mode", "floor_relative")
        if height_mode == "floor_relative":
            # Legacy mode: infer camera->head/CPF y offset from two height priors.
            camera_height = float(cfg.get("camera_height", 1.1))
            virtual_y = head_height - camera_height
        elif height_mode == "camera_relative":
            virtual_y = head_height
        else:
            raise ValueError(f"Unsupported virtual_ego_from_exo.height_mode: {height_mode}")
        offset = torch.tensor([0.0, virtual_y, back_offset], device=device, dtype=dtype)

    # Real EgoBody PV/head rotations are nearly identical; CPF then applies
    # T_head_cpf.R = diag([-1, 1, -1]). Do not apply that flip twice here,
    # otherwise the synthesized CPF frame faces the opposite convention.
    T_cam_head = torch.eye(4, device=device, dtype=dtype)
    T_cam_head[:3, 3] = offset
    T_cam_head = T_cam_head.reshape(1, 1, 4, 4).repeat(B, L, 1, 1)

    T_head_cpf = torch.eye(4, device=device, dtype=dtype)
    T_head_cpf[:3, :3] = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype))
    if "head_cpf_offset" in cfg:
        head_cpf_offset = torch.as_tensor(cfg["head_cpf_offset"], device=device, dtype=dtype)
        if head_cpf_offset.numel() != 3:
            raise ValueError("virtual_ego_from_exo.head_cpf_offset must have 3 values: [x, y, z]")
        T_head_cpf[:3, 3] = head_cpf_offset.reshape(3)
    T_head_cpf = T_head_cpf.reshape(1, 1, 4, 4).repeat(B, L, 1, 1)

    T_world_head = T_world_cam @ T_cam_head
    T_world_cpf = T_world_head @ T_head_cpf
    cond = dict(base_ego_cond) if base_ego_cond is not None else {}
    cond["T_world_cpf"] = T_world_cpf
    cond["T_world_head"] = T_world_head
    cond["T_world_pv"] = T_world_cam
    cond["T_abs_to_ego_world"] = T_abs_to_ego_world
    cond["head_valid"] = torch.ones((B, L), device=device, dtype=torch.bool)
    cond["head_angvel"] = rollout_inputs["cam_angvel"].to(device=device, dtype=dtype)
    cond["virtual_ego_from_exo"] = torch.ones((B, L), device=device, dtype=torch.bool)
    return cond


def _smpl_params_incam_to_input_world(smpl_params_c, inputs):
    """Lift OpenCV-camera SMPL root params into the active input camera's world frame."""
    T_world_cam = None
    if inputs.get("_input_role", None) == "ego":
        ego_cond = inputs.get("ego_cond", None)
        if isinstance(ego_cond, dict):
            T_world_cam = ego_cond.get("T_world_pv", None)
    if T_world_cam is None:
        T_world_cam = inputs.get("T_world_cam", inputs.get("T_world_exo_cam", None))
    if T_world_cam is None:
        T_world_cam = _identity_T_like(smpl_params_c["transl"])
    return _smpl_params_incam_to_world(smpl_params_c, T_world_cam)


def _smpl_params_incam_to_kinect_yup(smpl_params_c):
    """Legacy fallback for old camera-axis worlds without T_world_cam."""
    return _smpl_params_incam_to_world(smpl_params_c, _identity_T_like(smpl_params_c["transl"]))


def _decode_ego_cpf_x(pred_x_ego):
    B, L = pred_x_ego.shape[:2]
    body_pose_r6d = pred_x_ego[..., :126]
    betas = pred_x_ego[..., 126:136]
    root_orient_cpf_r6d = pred_x_ego[..., 136:142]
    root_trans_cpf = pred_x_ego[..., 142:145]
    body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6))).flatten(-2)
    root_orient_cpf = matrix_to_axis_angle(rotation_6d_to_matrix(root_orient_cpf_r6d))
    out = {
        "body_pose": body_pose,
        "betas": betas,
        "root_orient_cpf": root_orient_cpf,
        "root_trans_cpf": root_trans_cpf,
    }
    if pred_x_ego.size(-1) >= 148:
        out["root_residual_vel_cpf"] = pred_x_ego[..., 145:148]
        # Backward-compatible alias used by older loss/visualization helpers.
        out["local_transl_vel"] = out["root_residual_vel_cpf"]
    return out


def _make_T_from_orient_trans(orient_aa, transl):
    T = torch.zeros((*transl.shape[:-1], 4, 4), device=transl.device, dtype=transl.dtype)
    T[..., :3, :3] = axis_angle_to_matrix(orient_aa)
    T[..., :3, 3] = transl
    T[..., 3, 3] = 1.0
    return T


def _head_id(args):
    return int(args.get("ego_head_joint_id", 15))


def _root_to_head_T(body_pose, betas, endecoder, head_joint_id):
    B, L = body_pose.shape[:2]
    zeros_orient = body_pose.new_zeros(B, L, 3)
    zeros_transl = body_pose.new_zeros(B, L, 3)
    _, _, fk_mat = endecoder.fk_v2(
        body_pose=body_pose,
        betas=betas,
        global_orient=zeros_orient,
        transl=zeros_transl,
        get_intermediate=True,
    )
    return fk_mat[:, :, head_joint_id]


def _world_head_T_from_smpl(smpl_w, endecoder, head_joint_id):
    _, _, fk_mat = endecoder.fk_v2(
        body_pose=smpl_w["body_pose"],
        betas=smpl_w["betas"],
        global_orient=smpl_w["global_orient"],
        transl=smpl_w["transl"],
        get_intermediate=True,
    )
    return fk_mat[:, :, head_joint_id]


def _root_params_from_head_T(T_world_head, body_pose, betas, endecoder, head_joint_id):
    T_root_head = _root_to_head_T(body_pose, betas, endecoder, head_joint_id).to(
        device=T_world_head.device, dtype=T_world_head.dtype
    )
    T_world_root = T_world_head @ _invert_T(T_root_head)
    return matrix_to_axis_angle(T_world_root[..., :3, :3]), T_world_root[..., :3, 3]



def _encode_ego_cpf_targets(inputs, endecoder, args, include_local_transl_vel=False):
    ego_targets = _targets_for_ego_world(inputs)
    ego_cond = inputs.get("ego_cond", {})
    if "T_world_cpf" not in ego_cond:
        raise KeyError("ego_cond['T_world_cpf'] is required for CPF-local ego supervision")
    smpl_c = ego_targets["smpl_params_c"]
    smpl_w = ego_targets["smpl_params_w"]
    B, L = smpl_c["body_pose"].shape[:2]
    body_pose = smpl_c["body_pose"].reshape(B, L, 21, 3)
    body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(-2)
    root_orient_cpf, root_trans_cpf = _transform_root_to_local(
        smpl_w["global_orient"], smpl_w["transl"], ego_cond["T_world_cpf"], smpl_w.get("betas")
    )
    root_orient_cpf_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(root_orient_cpf))

    # Direct root-in-CPF position is the per-frame camera-space anchor, analogous
    # to exo pred_cam/transl_c. The velocity head is an auxiliary dynamic
    # representation coupled to the direct trajectory.
    chunks = [body_pose_r6d, smpl_c["betas"], root_orient_cpf_r6d, root_trans_cpf]
    if include_local_transl_vel:
        chunks.append(get_local_transl_vel(root_trans_cpf, root_orient_cpf))
    return torch.cat(chunks, dim=-1)


def _ego_cpf_to_smpl_params_world(decode_dict_ego, ego_cond, endecoder, args, rollout_residual=False):
    if "T_world_cpf" not in ego_cond:
        return None
    root_orient_cpf = decode_dict_ego["root_orient_cpf"]
    root_trans_cpf = decode_dict_ego["root_trans_cpf"]
    if rollout_residual and "root_residual_vel_cpf" in decode_dict_ego:
        root_trans_cpf = rollout_local_transl_vel(
            decode_dict_ego["root_residual_vel_cpf"],
            root_orient_cpf,
            root_trans_cpf[:, :1],
        )
    global_orient_w, transl_w = _transform_root_to_world(
        root_orient_cpf,
        root_trans_cpf,
        ego_cond["T_world_cpf"],
        decode_dict_ego.get("betas"),
    )
    out = {
        "body_pose": decode_dict_ego["body_pose"],
        "betas": decode_dict_ego["betas"],
        "global_orient": global_orient_w,
        "transl": transl_w,
    }
    head_joint_id = _head_id(args)
    T_world_root = _make_T_from_orient_trans(global_orient_w, transl_w)
    T_root_head = _root_to_head_T(decode_dict_ego["body_pose"], decode_dict_ego["betas"], endecoder, head_joint_id).to(
        device=T_world_root.device, dtype=T_world_root.dtype
    )
    T_world_head = T_world_root @ T_root_head
    out["head_global_orient"] = matrix_to_axis_angle(T_world_head[..., :3, :3])
    out["head_transl"] = T_world_head[..., :3, 3]
    return out


def _ego_cpf_target_params(inputs, endecoder, args):
    # FK body losses operate on SMPL root params. The network output is now
    # root-in-CPF, so this target mirrors pred_smpl_params_cpf_ego directly.
    ego_targets = _targets_for_ego_world(inputs)
    ego_cond = inputs.get("ego_cond", {})
    root_orient_cpf, root_trans_cpf = _transform_root_to_local(
        ego_targets["smpl_params_w"]["global_orient"],
        ego_targets["smpl_params_w"]["transl"],
        ego_cond["T_world_cpf"],
        ego_targets["smpl_params_w"].get("betas"),
    )
    return {
        "body_pose": ego_targets["smpl_params_c"]["body_pose"],
        "betas": ego_targets["smpl_params_c"]["betas"],
        "global_orient": root_orient_cpf,
        "transl": root_trans_cpf,
    }

def _axis_angle_y(angle):
    aa = torch.zeros((*angle.shape, 3), device=angle.device, dtype=angle.dtype)
    aa[..., 1] = angle
    return aa


def _make_ego_head_condition(ego_cond):
    """CPF/head condition inspired by EgoAllo's non-redundant trajectory encoding.

    Output dimension is 25:
        relative rotation 6D, relative translation in previous CPF/head frame 3D,
        height 1D, yaw-canonicalized rotation 6D, linear velocity 3D,
        relative angular velocity 6D.
    """
    if ego_cond is None:
        return None
    T = None
    for key in ("T_world_cpf", "T_holo_cpf", "T_world_head", "T_holo_head"):
        if key in ego_cond:
            T = ego_cond[key]
            break
    if T is None:
        return None
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    B, L = t.shape[:2]

    rel_R = torch.eye(3, device=T.device, dtype=T.dtype).expand(B, L, 3, 3).clone()
    rel_t = torch.zeros((B, L, 3), device=T.device, dtype=T.dtype)
    if L > 1:
        rel_R[:, 1:] = R[:, :-1].mT @ R[:, 1:]
        rel_t[:, 1:] = torch.einsum("blij,blj->bli", R[:, :-1].mT, t[:, 1:] - t[:, :-1])
    rel_r6d = matrix_to_rotation_6d(rel_R)

    # Holo/EgoBody and GVHMR training coordinates are y-up in our preprocessing.
    height = t[..., [1]]
    vel = torch.zeros_like(t)
    if L > 1:
        vel[:, 1:] = t[:, 1:] - t[:, :-1]

    forward = R[..., :, 2]
    yaw = torch.atan2(forward[..., 0], forward[..., 2])
    R_canon = axis_angle_to_matrix(_axis_angle_y(-yaw)) @ R
    canon_r6d = matrix_to_rotation_6d(R_canon)

    if "head_angvel" in ego_cond:
        angvel = ego_cond["head_angvel"]
    else:
        angvel = rel_r6d
    return torch.cat([rel_r6d, rel_t, height, canon_r6d, vel, angvel], dim=-1)


def _make_ego_hand_condition(ego_cond):
    if ego_cond is None:
        return None
    left = None
    right = None
    for key in ("left_hand_summary", "left_hand_summary_holo"):
        if key in ego_cond:
            left = ego_cond[key]
            break
    for key in ("right_hand_summary", "right_hand_summary_holo"):
        if key in ego_cond:
            right = ego_cond[key]
            break
    if left is None or right is None:
        return None
    # First version uses wrist/palm/palm-normal summaries from each hand.
    return torch.cat([left[..., :9], right[..., :9]], dim=-1)


def _make_interaction_condition(inputs, ego_cond=None):
    """Oracle ego-exo relative condition for testing interaction cues."""
    exo = inputs.get("exo", None)
    if not isinstance(exo, dict) or exo.get("smpl_params_w", None) is None:
        return None
    exo_w = exo["smpl_params_w"]
    if ego_cond is not None and isinstance(ego_cond, dict) and "T_abs_to_ego_world" in ego_cond:
        exo_w = _smpl_params_to_ego_world(exo_w, ego_cond)
    if exo_w is None:
        return None
    transl = exo_w["transl"]
    if transl.ndim != 3:
        return None
    B, L = transl.shape[:2]
    cam_pos = transl.new_zeros(B, L, 3)
    if ego_cond is not None and isinstance(ego_cond, dict) and "T_world_pv" in ego_cond:
        T_cam = ego_cond["T_world_pv"].to(device=transl.device, dtype=transl.dtype)
        if T_cam.ndim == 3:
            T_cam = T_cam.unsqueeze(0)
        if T_cam.shape[0] == 1 and B > 1:
            T_cam = T_cam.expand(B, -1, -1, -1)
        cam_pos = T_cam[:, :L, :3, 3]
    rel = transl - cam_pos
    if L > 1:
        rel_vel = torch.cat([rel[:, 1:] - rel[:, :-1], rel[:, -1:] - rel[:, -2:-1]], dim=1)
    else:
        rel_vel = rel.zero_()
    R = axis_angle_to_matrix(exo_w["global_orient"])
    forward = F.normalize(R[..., :, 2], dim=-1)
    dist = rel.norm(dim=-1, keepdim=True)
    cond = torch.cat([rel / 3.0, rel_vel / 0.2, forward, dist / 5.0], dim=-1)
    cond = torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)
    mask = inputs.get("mask", {}).get("exo_valid", inputs.get("mask", {}).get("valid", None))
    if mask is not None:
        cond = cond * mask[:, :L, None].to(device=cond.device, dtype=cond.dtype)
    return cond


class Pipeline(nn.Module):
    def __init__(self, args, args_denoiser3d, **kwargs):
        super().__init__()
        self.args = args
        self.weights = args.weights  # loss weights

        # Networks
        self.denoiser3d = instantiate(args_denoiser3d, _recursive_=False)
        # Log.info(self.denoiser3d)

        # Normalizer
        self.endecoder: EnDecoder = instantiate(args.endecoder_opt, _recursive_=False)
        if self.args.normalize_cam_angvel:
            cam_angvel_stats = stats_compose.cam_angvel["manual"]
            self.register_buffer("cam_angvel_mean", torch.tensor(cam_angvel_stats["mean"]), persistent=False)
            self.register_buffer("cam_angvel_std", torch.tensor(cam_angvel_stats["std"]), persistent=False)

    # ========== Training ========== #

    def forward(self, inputs, train=False, postproc=False, static_cam=False):
        outputs = dict()
        branch_mode = _get_branch_mode(self.args)
        branch_mode = _effective_branch_mode(inputs, branch_mode)
        active_exo = branch_mode in ("exo", "both")
        active_ego = branch_mode in ("ego", "both")
        input_role = inputs.get("_input_role", self.args.get("input_role", None))
        if input_role == "auto":
            input_role = None
        supervise_role = inputs.get("_supervise_role", self.args.get("supervise_role", None))
        if supervise_role is None or supervise_role == "auto":
            supervise_role = branch_mode
        inputs, ego_inputs, ego_cond = _split_paired_inputs(inputs, branch_mode, input_role=input_role)
        input_role = inputs.get("_input_role", input_role or ("ego" if branch_mode == "ego" else "exo"))
        virtual_ego_cfg = self.args.get("virtual_ego_from_exo", {})
        ego_motion_mode = self.args.get("ego_motion_mode", "absolute_cpf")
        if active_ego and input_role == "exo" and virtual_ego_cfg.get("enabled", False):
            mode = virtual_ego_cfg.get("mode", "fixed_observer")
            if mode != "fixed_observer":
                raise ValueError(f"Unsupported virtual_ego_from_exo.mode: {mode}")
            ego_cond = _make_fixed_observer_ego_cond(inputs, ego_cond, virtual_ego_cfg)
        elif active_ego and input_role == "ego" and ego_motion_mode == "cam_angvel_cpf":
            ego_cond = _make_cam_angvel_cpf_ego_cond(inputs, ego_cond, virtual_ego_cfg)
        if ego_cond is not None:
            inputs["ego_cond"] = ego_cond
        length = inputs["length"]  # (B,) effective length of each sample

        # *. Conditions
        cliff_cam = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        f_cam_angvel = inputs["cam_angvel"]
        if self.args.normalize_cam_angvel:
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std
        B_cond, L_cond = f_cam_angvel.shape[:2]
        f_cam_trans_vel = inputs.get("cam_trans_vel", None)
        if f_cam_trans_vel is None:
            f_cam_trans_vel = f_cam_angvel.new_zeros(B_cond, L_cond, 3)
        else:
            f_cam_trans_vel = f_cam_trans_vel.to(device=f_cam_angvel.device, dtype=f_cam_angvel.dtype)
        f_gravity_dir = inputs.get("gravity_dir_cam", None)
        if f_gravity_dir is None:
            f_gravity_dir = f_cam_angvel.new_zeros(B_cond, L_cond, 3)
            f_gravity_dir[..., 1] = 1.0
        else:
            f_gravity_dir = f_gravity_dir.to(device=f_cam_angvel.device, dtype=f_cam_angvel.dtype)
        f_condition = {
            "obs": inputs["obs"],  # (B, L, J, 3)
            "f_cliffcam": cliff_cam,  # (B, L, 3)
            "f_cam_angvel": f_cam_angvel,  # (B, L, C=6)
            "f_cam_trans_vel": f_cam_trans_vel,  # (B, L, C=3)
            "f_gravity_dir": f_gravity_dir,  # (B, L, C=3)
            "f_imgseq": inputs["f_imgseq"],  # (B, L, C=1024)
            "f_ego_head": None if ego_motion_mode == "cam_angvel_cpf" else (_make_ego_head_condition(ego_cond) if active_ego else None),
            "f_ego_hand": None if ego_motion_mode == "cam_angvel_cpf" else (_make_ego_hand_condition(ego_cond) if active_ego else None),
            "f_interaction": _make_interaction_condition(inputs, ego_cond) if self.args.get("use_interaction_condition", False) else None,
            "ego_partner_visibility": ego_inputs.get("partner_visibility") if isinstance(ego_inputs, dict) else None,
        }
        if branch_mode == "both" and ego_inputs is not None and self.args.get("add_other_role_image_condition", False):
            f_ego_imgseq = ego_inputs.get("f_body_imgseq", ego_inputs.get("f_imgseq"))
            if f_ego_imgseq is not None:
                f_condition["f_ego_imgseq"] = f_ego_imgseq

        if (
            branch_mode == "both"
            and input_role == "ego"
            and ego_inputs is not None
            and self.args.get("use_cross_view_teacher", False)
        ):
            exo_token_dropout = float(self.args.get("cross_view_exo_token_dropout", 0.0))
            exo_person_dropout = float(self.args.get("cross_view_exo_person_dropout", 0.0))
            force_no_cross_source = bool(self.args.get("cross_view_force_no_source", False))
            cross_stats = {}
            cross_extrinsic = None
            T_abs_exo_cam = inputs.get("T_world_exo_cam", inputs.get("T_world_cam", None))
            if isinstance(ego_cond, dict):
                T_abs_to_target = ego_cond.get("T_abs_to_ego_world", None)
                T_target_pv = ego_cond.get("T_world_pv", None)
                if T_abs_exo_cam is not None and T_abs_to_target is not None and T_target_pv is not None:
                    T_abs_exo_cam = T_abs_exo_cam.to(device=f_cam_angvel.device, dtype=f_cam_angvel.dtype)
                    T_abs_to_target = T_abs_to_target.to(device=f_cam_angvel.device, dtype=f_cam_angvel.dtype)
                    T_target_pv = T_target_pv.to(device=f_cam_angvel.device, dtype=f_cam_angvel.dtype)
                    T_target_from_exo = _invert_T(T_target_pv) @ (T_abs_to_target @ T_abs_exo_cam)
                    rel_rot6d = matrix_to_rotation_6d(T_target_from_exo[..., :3, :3])
                    trans_scale = float(self.args.get("cross_view_extrinsic_trans_scale", 3.0))
                    rel_trans = T_target_from_exo[..., :3, 3] / max(trans_scale, 1e-6)
                    cross_extrinsic = torch.cat([rel_rot6d, rel_trans], dim=-1)

            def _as_condition_tensor(x, like):
                if x is None:
                    return None
                x = x.to(device=like.device, dtype=like.dtype)
                if x.shape[0] == 1 and like.shape[0] > 1:
                    x = x.expand(like.shape[0], -1, *x.shape[2:])
                if x.shape[1] != like.shape[1]:
                    x = x[:, : like.shape[1]]
                return x

            def _prep_cross_person(
                prefix,
                bbx,
                kp2d,
                img,
                K,
                valid_mask,
                feature_mask,
                person_name,
            ):
                if bbx is None or kp2d is None or img is None:
                    return False
                valid_mask = _as_condition_tensor(valid_mask, f_cam_angvel)
                valid_mask = valid_mask.bool() if valid_mask is not None else None
                feature_mask = _as_condition_tensor(feature_mask, f_cam_angvel)
                if feature_mask is not None:
                    feature_mask = feature_mask.bool()
                    if not feature_mask.any():
                        return False
                    valid_mask = feature_mask if valid_mask is None else (valid_mask & feature_mask)

                cross_obs = normalize_kp2d(kp2d, bbx)
                if valid_mask is not None:
                    cross_obs[~valid_mask] = 0
                cross_cliff = compute_bbox_info_bedlam(bbx, K)
                cross_angvel = _identity_cam_angvel(B_cond, L_cond, f_cam_angvel.device, f_cam_angvel.dtype)
                if self.args.normalize_cam_angvel:
                    cross_angvel = (cross_angvel - self.cam_angvel_mean) / self.cam_angvel_std
                cross_trans_vel = f_cam_angvel.new_zeros(B_cond, L_cond, 3)
                cross_gravity = f_cam_angvel.new_zeros(B_cond, L_cond, 3)
                cross_gravity[..., 1] = 1.0
                img = img.to(device=f_cam_angvel.device, dtype=f_cam_angvel.dtype)
                extrinsic = None if cross_extrinsic is None else cross_extrinsic.to(
                    device=f_cam_angvel.device, dtype=f_cam_angvel.dtype
                )

                source_valid = torch.ones(cross_obs.shape[:2], device=cross_obs.device, dtype=torch.bool)
                if valid_mask is not None:
                    source_valid &= valid_mask.to(device=cross_obs.device)
                if force_no_cross_source:
                    source_valid.zero_()
                if self.training:
                    keep = torch.ones_like(source_valid)
                    if exo_token_dropout > 0:
                        keep &= torch.rand_like(keep, dtype=torch.float32) >= exo_token_dropout
                    if exo_person_dropout > 0:
                        person_keep = torch.rand((keep.shape[0], 1), device=keep.device) >= exo_person_dropout
                        keep &= person_keep
                        cross_stats[f"{person_name}_person_keep_frac"] = person_keep.float().mean().detach()
                    cross_stats[f"{person_name}_token_keep_frac"] = keep.float().mean().detach()
                    source_valid &= keep

                    cross_obs = cross_obs.masked_fill(~keep[..., None, None], 0.0)
                    cross_cliff = cross_cliff.masked_fill(~keep[..., None], 0.0)
                    cross_angvel = cross_angvel.masked_fill(~keep[..., None], 0.0)
                    cross_trans_vel = cross_trans_vel.masked_fill(~keep[..., None], 0.0)
                    cross_gravity = cross_gravity.masked_fill(~keep[..., None], 0.0)
                    img = img.masked_fill(~keep[..., None], 0.0)
                    if extrinsic is not None:
                        extrinsic = extrinsic.masked_fill(~keep[..., None], 0.0)

                cross_stats[f"{person_name}_source_valid_frac"] = source_valid.float().mean().detach()
                f_condition.update({
                    f"{prefix}_obs": cross_obs,
                    f"{prefix}_f_cliffcam": cross_cliff,
                    f"{prefix}_f_cam_angvel": cross_angvel,
                    f"{prefix}_f_cam_trans_vel": cross_trans_vel,
                    f"{prefix}_f_gravity_dir": cross_gravity,
                    f"{prefix}_f_imgseq": img,
                    f"{prefix}_f_extrinsic": extrinsic,
                    f"{prefix}_key_padding_mask": ~source_valid,
                })
                return True

            cross_inputs = inputs.get("exo", None)
            added_people = []
            if isinstance(cross_inputs, dict):
                cross_K = inputs.get("K_exo", inputs.get("K_fullimg"))
                exo_valid = inputs["mask"].get("exo_valid", inputs["mask"]["valid"])
                ego_valid = inputs["mask"].get("ego_valid", inputs["mask"]["valid"])
                if _prep_cross_person(
                    "cross_interactee",
                    cross_inputs.get("bbx_xys"),
                    cross_inputs.get("kp2d"),
                    cross_inputs.get("f_imgseq"),
                    cross_K,
                    exo_valid,
                    inputs["mask"].get("f_imgseq", None),
                    "interactee",
                ):
                    added_people.append("interactee")
                if _prep_cross_person(
                    "cross_wearer",
                    cross_inputs.get("wearer_bbx_xys"),
                    cross_inputs.get("wearer_kp2d"),
                    cross_inputs.get("f_wearer_imgseq"),
                    cross_K,
                    ego_valid,
                    inputs["mask"].get("exo_wearer_f_imgseq", None),
                    "wearer",
                ):
                    added_people.append("wearer")
            if added_people:
                for stat_name, stat_value in cross_stats.items():
                    outputs[f"cross_view_{stat_name}"] = stat_value

        ego_condition_ablation = self.args.get("cross_view_ego_condition_ablation", "full")
        if ego_condition_ablation not in ("full", "no-visual", "no-motion", "none"):
            raise ValueError(f"Unsupported cross_view_ego_condition_ablation: {ego_condition_ablation}")
        if ego_condition_ablation in ("no-visual", "none"):
            for condition_name in ("obs", "f_imgseq", "f_ego_imgseq", "f_interaction"):
                value = f_condition.get(condition_name)
                if torch.is_tensor(value):
                    f_condition[condition_name] = torch.zeros_like(value)
            f_condition["ego_partner_visibility"] = None
        if ego_condition_ablation in ("no-motion", "none"):
            for condition_name in ("f_cam_angvel", "f_cam_trans_vel", "f_ego_head", "f_ego_hand"):
                value = f_condition.get(condition_name)
                if torch.is_tensor(value):
                    f_condition[condition_name] = torch.zeros_like(value)

        if train and self.training:
            f_condition = randomly_set_null_condition(f_condition, 0.1)

        # Forward & output
        bad_parameters = [
            name for name, parameter in self.denoiser3d.named_parameters()
            if parameter.requires_grad and not torch.isfinite(parameter).all()
        ]
        if bad_parameters:
            names = ", ".join(bad_parameters[:12])
            raise FloatingPointError(f"Non-finite trainable denoiser parameters: {names}")
        for condition_name, value in f_condition.items():
            if not torch.is_tensor(value) or value.dtype == torch.bool:
                continue
            finite = torch.isfinite(value)
            if not finite.all():
                bad_per_sample = (~finite).reshape(finite.shape[0], -1).sum(dim=1)
                bad_samples = torch.nonzero(bad_per_sample, as_tuple=False).flatten().tolist()
                raise FloatingPointError(
                    f"Non-finite denoiser condition {condition_name}: "
                    f"count={int((~finite).sum())}, batch_indices={bad_samples}"
                )
        model_output = self.denoiser3d(length=length, **f_condition)  # pred_x, pred_cam, static_conf_logits
        for gate_name in ("cross_partner_gate", "cross_wearer_gate"):
            gate_value = model_output.get(gate_name, None)
            if gate_value is not None:
                outputs[f"cross_view_{gate_name.removeprefix('cross_')}"] = gate_value.detach().mean()
        for output_name in ("pred_x", "pred_x_ego"):
            value = model_output.get(output_name, None)
            if value is not None and not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite {output_name} from denoiser3d")
        frozen_ego_feature = None
        if ego_inputs is not None:
            frozen_ego_feature = ego_inputs.get("f_body_imgseq", ego_inputs.get("f_imgseq"))
        if active_ego and self.args.get("enable_frozen_ego_image_exo", False) and frozen_ego_feature is not None:
            with torch.no_grad():
                frozen_condition = dict(f_condition)
                frozen_condition["f_imgseq"] = frozen_ego_feature
                frozen_condition["f_ego_imgseq"] = None
                frozen_condition["f_ego_head"] = None
                frozen_condition["f_ego_hand"] = None
                frozen_output = self.denoiser3d(length=length, **frozen_condition)
                frozen_decode = self.endecoder.decode(frozen_output["pred_x"].detach())
                frozen_global = get_smpl_params_w_Rt_v2(
                    global_orient_gv=frozen_decode["global_orient_gv"],
                    local_transl_vel=frozen_decode["local_transl_vel"],
                    global_orient_c=frozen_decode["global_orient"],
                    cam_angvel=inputs["cam_angvel"],
                )
                outputs["frozen_ego_image_exo_decode_dict"] = frozen_decode
                outputs["frozen_ego_image_exo_incam"] = {
                    "body_pose": frozen_decode["body_pose"],
                    "betas": frozen_decode["betas"],
                    "global_orient": frozen_decode["global_orient"],
                    "transl": compute_transl_full_cam(frozen_output["pred_cam"].detach(), inputs["bbx_xys"], inputs["K_fullimg"]),
                }
                outputs["frozen_ego_image_exo_kinect_from_incam"] = _smpl_params_incam_to_input_world(
                    outputs["frozen_ego_image_exo_incam"], inputs
                )
                if ego_cond is not None and "T_world_pv" in ego_cond:
                    outputs["frozen_ego_image_exo_world_from_pv"] = _smpl_params_incam_to_world(
                        outputs["frozen_ego_image_exo_incam"], ego_cond["T_world_pv"]
                    )
                outputs["frozen_ego_image_exo_global"] = {
                    "body_pose": frozen_decode["body_pose"],
                    "betas": frozen_decode["betas"],
                    **frozen_global,
                }
        decode_dict = self.endecoder.decode(model_output["pred_x"]) if active_exo else None
        outputs.update({"model_output": model_output})
        if decode_dict is not None:
            outputs["decode_dict"] = decode_dict
        decode_dict_ego = None
        if active_ego and "pred_x_ego" in model_output:
            if model_output["pred_x_ego"].size(-1) == 145 or self.args.get("ego_head_type", "cpf") == "cpf":
                decode_dict_ego = _decode_ego_cpf_x(model_output["pred_x_ego"])
            else:
                decode_dict_ego = self.endecoder.decode(model_output["pred_x_ego"])
            outputs.update({"decode_dict_ego": decode_dict_ego})

        # Post-processing
        if active_exo:
            outputs["pred_smpl_params_incam"] = {
                "body_pose": decode_dict["body_pose"],  # (B, L, 63)
                "betas": decode_dict["betas"],  # (B, L, 10)
                "global_orient": decode_dict["global_orient"],  # (B, L, 3)
                "transl": compute_transl_full_cam(model_output["pred_cam"], inputs["bbx_xys"], inputs["K_fullimg"]),
            }
            outputs["pred_smpl_params_kinect_from_incam"] = _smpl_params_incam_to_input_world(
                outputs["pred_smpl_params_incam"], inputs
            )
        if active_ego and input_role == "ego" and ego_motion_mode == "cam_angvel_cpf":
            outputs["ego_gt_smpl_params_w_aligned"] = _targets_for_ego_world(inputs)["smpl_params_w"]
            if "exo" in inputs and isinstance(inputs["exo"], dict):
                outputs["exo_gt_smpl_params_w_aligned"] = _smpl_params_to_ego_world(inputs["exo"].get("smpl_params_w"), ego_cond)

        if decode_dict_ego is not None:
            if "root_orient_cpf" in decode_dict_ego:
                outputs["pred_smpl_params_cpf_ego"] = {
                    "body_pose": decode_dict_ego["body_pose"],
                    "betas": decode_dict_ego["betas"],
                    "global_orient": decode_dict_ego["root_orient_cpf"],
                    "transl": decode_dict_ego["root_trans_cpf"],
                }
                if ego_cond is not None:
                    pred_world_ego_direct = _ego_cpf_to_smpl_params_world(decode_dict_ego, ego_cond, self.endecoder, self.args)
                    if pred_world_ego_direct is not None:
                        outputs["pred_smpl_params_global_ego_direct"] = pred_world_ego_direct
                        outputs["pred_smpl_params_global_ego_coarse"] = pred_world_ego_direct
                        outputs["pred_smpl_params_global_ego"] = pred_world_ego_direct
                        if "root_residual_vel_cpf" in decode_dict_ego:
                            outputs["pred_smpl_params_global_ego_vel"] = _ego_cpf_to_smpl_params_world(
                                decode_dict_ego, ego_cond, self.endecoder, self.args, rollout_residual=True
                            )
            else:
                outputs["pred_smpl_params_incam_ego"] = {
                    "body_pose": decode_dict_ego["body_pose"],
                    "betas": decode_dict_ego["betas"],
                    "global_orient": decode_dict_ego["global_orient"],
                    "transl": compute_transl_full_cam(model_output["pred_cam_ego"], inputs["bbx_xys"], inputs["K_fullimg"]),
                }

        source_base = model_output.get("cross_source_base", {})
        use_teacher_source_output = not bool(self.args.get("cross_view_force_no_source", False))
        if (
            self.args.get("cross_teacher_mvp", False)
            and use_teacher_source_output
            and source_base
            and isinstance(ego_cond, dict)
        ):
            source_specs = {
                "partner": (
                    inputs.get("exo", {}).get("bbx_xys"),
                    "teacher_partner_motion",
                ),
                "wearer": (
                    inputs.get("exo", {}).get("wearer_bbx_xys"),
                    "teacher_wearer_motion",
                ),
            }
            T_world_exo_cam = inputs.get("T_world_exo_cam", inputs.get("T_world_cam", None))
            K_exo = inputs.get("K_exo", inputs.get("K_fullimg"))
            for source_name, (source_bbx, output_name) in source_specs.items():
                base_output = source_base.get(source_name, None)
                if base_output is None or source_bbx is None or T_world_exo_cam is None:
                    continue
                base_decode = self.endecoder.decode(base_output["pred_x"])
                base_c = {
                    "body_pose": base_decode["body_pose"],
                    "betas": base_decode["betas"],
                    "global_orient": base_decode["global_orient"],
                    "transl": compute_transl_full_cam(base_output["pred_cam"], source_bbx, K_exo),
                }
                base_abs_world = _smpl_params_incam_to_world(base_c, T_world_exo_cam)
                base_target_world = _smpl_params_to_ego_world(base_abs_world, ego_cond)
                teacher_motion = _compose_teacher_residual(
                    base_target_world,
                    base_output["residual"],
                    pose_scale=float(self.args.get("teacher_pose_residual_scale", 0.1)),
                    root_rot_scale=float(self.args.get("teacher_root_rot_residual_scale", 0.1)),
                    root_trans_scale=float(self.args.get("teacher_root_trans_residual_scale", 0.25)),
                )
                outputs[f"exo_{source_name}_base_motion"] = base_target_world
                outputs[f"teacher_{source_name}_residual"] = base_output["residual"]
                outputs[output_name] = teacher_motion
                outputs[f"teacher_{source_name}_static_conf_logits"] = base_output.get("static_conf_logits")

            teacher_wearer = outputs.get("teacher_wearer_motion", None)
            if teacher_wearer is not None:
                # Canonical wearer output now starts from complete exo pose/root and receives ego-conditioned corrections.
                outputs["pred_smpl_params_global_ego"] = teacher_wearer
                outputs["pred_smpl_params_global_ego_direct"] = teacher_wearer
                outputs["pred_smpl_params_global_ego_coarse"] = teacher_wearer
                if "pred_smpl_params_cpf_ego" in outputs:
                    outputs["pred_smpl_params_cpf_ego"] = {
                        **outputs["pred_smpl_params_cpf_ego"],
                        "body_pose": teacher_wearer["body_pose"],
                        "betas": teacher_wearer["betas"],
                    }

            teacher_partner = outputs.get("teacher_partner_motion", None)
            if teacher_partner is not None and ego_cond.get("T_world_pv", None) is not None:
                teacher_partner_c = _smpl_params_world_to_incam(teacher_partner, ego_cond["T_world_pv"])
                outputs["teacher_partner_motion_incam"] = teacher_partner_c
                outputs["pred_smpl_params_incam"] = teacher_partner_c
                outputs["pred_smpl_params_kinect_from_incam"] = teacher_partner
                outputs["pred_smpl_params_global"] = teacher_partner

        if not train:
            if active_exo and "teacher_partner_motion" not in outputs:
                pred_smpl_params_global = get_smpl_params_w_Rt_v2(  # This function has for-loop
                    global_orient_gv=decode_dict["global_orient_gv"],
                    local_transl_vel=decode_dict["local_transl_vel"],
                    global_orient_c=decode_dict["global_orient"],
                    cam_angvel=inputs["cam_angvel"],
                )
                outputs["pred_smpl_params_global"] = {
                    "body_pose": decode_dict["body_pose"],
                    "betas": decode_dict["betas"],
                    **pred_smpl_params_global,
                }
                outputs["static_conf_logits"] = model_output["static_conf_logits"]

            if decode_dict_ego is not None and "root_orient_cpf" not in decode_dict_ego:
                pred_smpl_params_global_ego = get_smpl_params_w_Rt_v2(
                    global_orient_gv=decode_dict_ego["global_orient_gv"],
                    local_transl_vel=decode_dict_ego["local_transl_vel"],
                    global_orient_c=decode_dict_ego["global_orient"],
                    cam_angvel=inputs["cam_angvel"],
                )
                outputs["pred_smpl_params_global_ego"] = {
                    "body_pose": decode_dict_ego["body_pose"],
                    "betas": decode_dict_ego["betas"],
                    **pred_smpl_params_global_ego,
                }
                outputs["static_conf_logits_ego"] = model_output["static_conf_logits_ego"]
            elif decode_dict_ego is not None:
                outputs["static_conf_logits_ego"] = model_output.get("static_conf_logits_ego", None)

            if postproc:  # apply post-processing
                # Exo post-processing
                if active_exo:
                    if static_cam:  # extra post-processing to utilize static camera prior
                        outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_cam(outputs, self.endecoder)
                    else:
                        outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(outputs, self.endecoder)
                
                # Ego post-processing (如果存在)
                if decode_dict_ego is not None:
                    if static_cam:
                        outputs["pred_smpl_params_global_ego"]["transl"] = pp_static_joint_cam_ego(outputs, self.endecoder)
                    else:
                        outputs["pred_smpl_params_global_ego"]["transl"] = pp_static_joint_ego(outputs, self.endecoder)
                if active_exo:
                    body_pose = process_ik(outputs, self.endecoder)
                    decode_dict["body_pose"] = body_pose
                    outputs["pred_smpl_params_global"]["body_pose"] = body_pose
                    outputs["pred_smpl_params_incam"]["body_pose"] = body_pose

            return outputs

        # ========== Compute Loss ========== #
        total_loss = 0
        mask = inputs["mask"]["valid"]  # (B, L)
        residual_reg_weight = _loss_weight(self.weights, "teacher_residual_reg")
        if residual_reg_weight > 0:
            for role in ("wearer", "partner"):
                residual = outputs.get(f"teacher_{role}_residual", None)
                if residual is None:
                    continue
                role_mask = _mask_for_role(inputs, "ego" if role == "wearer" else "exo")
                residual_reg = safe_masked_mean(residual.pow(2), role_mask[..., None])
                total_loss += residual_reg * residual_reg_weight
                outputs[f"teacher_residual_reg_loss_{role}"] = residual_reg

        has_ego = active_ego and "pred_x_ego" in model_output
        supervise_ego = supervise_role in ("ego", "both")
        supervise_exo = supervise_role in ("exo", "both")
        
        # 1. Simple loss: MSE
        # Ego head loss (如果存在)
        if has_ego and supervise_ego:
            pred_x_ego = model_output["pred_x_ego"]
            if pred_x_ego.size(-1) == 145 or self.args.get("ego_head_type", "cpf") == "cpf":
                target_x = _encode_ego_cpf_targets(inputs, self.endecoder, self.args, include_local_transl_vel=pred_x_ego.size(-1) >= 148)
            else:
                ego_targets = _target_inputs_for_role(inputs, "ego")
                target_x = self.endecoder.encode(ego_targets)  # (B, L, C)
            simple_loss_ego = F.mse_loss(pred_x_ego, target_x, reduction="none")
            ego_mask = _mask_for_role(inputs, "ego")
            simple_loss_ego = safe_masked_mean(simple_loss_ego, ego_mask[:, :, None])
            total_loss += simple_loss_ego * _loss_weight(self.weights, "wearer_simple", default=1.0)
            outputs["simple_loss_ego"] = simple_loss_ego
        
        # Exo/partner head loss. In cross-view teacher with ego target view,
        # pred_x represents the partner decoded in the ego target coordinate
        # system, not the flat wearer fields routed through inputs.
        exo_loss_inputs = None
        if active_exo and supervise_exo and (not has_ego or not self.args.get("freeze_exo_head", False)):
            pred_x = model_output["pred_x"]
            exo_loss_inputs = _target_inputs_with_gt_cache(inputs, "exo", self.endecoder)
            target_x = self.endecoder.encode(exo_loss_inputs)
            simple_loss = F.mse_loss(pred_x, target_x, reduction="none")
            exo_mask = exo_loss_inputs["mask"]["valid"]
            mask_simple = exo_mask[:, :, None].expand(-1, -1, pred_x.size(2)).clone()
            spv_mask = exo_loss_inputs["mask"].get("spv_incam_only", torch.zeros_like(exo_mask, dtype=torch.bool))
            if spv_mask.ndim == 1:
                spv_mask = spv_mask[:, None].expand_as(exo_mask)
            mask_simple[..., 142:] = mask_simple[..., 142:] & (~spv_mask[..., None])
            simple_loss = safe_masked_mean(simple_loss, mask_simple)
            total_loss += simple_loss * _loss_weight(self.weights, "partner_simple", default=1.0)
            outputs["simple_loss"] = simple_loss

        # 2. Extra loss
        # Ego extra loss (如果存在)
        if has_ego and supervise_ego:
            ego_extra_loss, ego_extra_loss_dict = compute_extra_incam_loss_ego(inputs, outputs, self)
            total_loss += ego_extra_loss
            outputs.update(ego_extra_loss_dict)
            
            # Ego global loss
            ego_global_loss, ego_global_loss_dict = compute_extra_global_loss_ego(inputs, outputs, self)
            total_loss += ego_global_loss
            outputs.update(ego_global_loss_dict)
        
        # Exo/partner extra loss (如果存在且未冻结)
        if active_exo and supervise_exo and (not has_ego or not self.args.get("freeze_exo_head", False)):
            if exo_loss_inputs is None:
                exo_loss_inputs = _target_inputs_with_gt_cache(inputs, "exo", self.endecoder)
            extra_funcs = [
                compute_extra_incam_loss,
                compute_extra_global_loss,
            ]
            for extra_func in extra_funcs:
                extra_loss, extra_loss_dict = extra_func(exo_loss_inputs, outputs, self)
                total_loss += extra_loss
                outputs.update(extra_loss_dict)

        outputs["loss"] = total_loss
        return outputs


def randomly_set_null_condition(f_condition, uncond_prob=0.1):
    """Conditions are in shape (B, L, *)"""
    keys = list(f_condition.keys())
    for k in keys:
        if f_condition[k] is None or k.startswith("cross_") or k == "ego_partner_visibility":
            continue
        f_condition[k] = f_condition[k].clone()
        mask = torch.rand(f_condition[k].shape[:2], device=f_condition[k].device) < uncond_prob
        f_condition[k][mask] = 0.0
    return f_condition


def compute_extra_incam_loss_ego(inputs, outputs, ppl):
    """Ego CPF-local body loss. Reprojection is intentionally not used."""
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    mask = _mask_for_role(inputs, "ego")

    if "pred_smpl_params_cpf_ego" in outputs:
        pred_smpl_params = outputs["pred_smpl_params_cpf_ego"]
        gt_smpl_params = _ego_cpf_target_params(inputs, endecoder, ppl.args)
    else:
        pred_smpl_params = outputs["pred_smpl_params_incam_ego"]
        ego_targets = _target_inputs_for_role(inputs, "ego")
        gt_smpl_params = ego_targets["smpl_params_c"]

    pred_j3d = endecoder.fk_v2(**_body_smpl_params(pred_smpl_params))
    pred_cr_j3d = pred_j3d - pred_j3d[:, :, :1]
    gt_j3d = endecoder.fk_v2(**_body_smpl_params(gt_smpl_params))
    gt_cr_j3d = gt_j3d - gt_j3d[:, :, :1]

    cr_j3d_weight = _loss_weight(weights, "wearer_cr_j3d", "cr_j3d")
    if cr_j3d_weight > 0.0:
        cr_j3d_loss = F.mse_loss(pred_cr_j3d, gt_cr_j3d, reduction="none")
        cr_j3d_loss = safe_masked_mean(cr_j3d_loss, mask[..., None, None])
        extra_loss += cr_j3d_loss * cr_j3d_weight
        extra_loss_dict["cr_j3d_loss_ego"] = cr_j3d_loss

    j3d_vel_weight = _loss_weight(weights, "wearer_j3d_vel")
    if j3d_vel_weight > 0 and pred_cr_j3d.size(1) > 1:
        vel_mask = mask[:, 1:] & mask[:, :-1]
        pred_vel = pred_cr_j3d[:, 1:] - pred_cr_j3d[:, :-1]
        gt_vel = gt_cr_j3d[:, 1:] - gt_cr_j3d[:, :-1]
        j3d_vel_loss = F.smooth_l1_loss(pred_vel, gt_vel, reduction="none", beta=0.02)
        j3d_vel_loss = safe_masked_mean(j3d_vel_loss, vel_mask[..., None, None])
        extra_loss += j3d_vel_loss * j3d_vel_weight
        extra_loss_dict["j3d_vel_loss_ego"] = j3d_vel_loss

    upper_j3d_weight = _loss_weight(weights, "wearer_upper_j3d", "ego_upper_j3d")
    if upper_j3d_weight > 0:
        # SMPL22: neck/collars/head/shoulders/elbows/wrists.
        upper_ids = torch.as_tensor(
            args.get("ego_upper_joint_ids", [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
            device=pred_cr_j3d.device,
            dtype=torch.long,
        )
        pred_upper = pred_cr_j3d.index_select(-2, upper_ids)
        gt_upper = gt_cr_j3d.index_select(-2, upper_ids)
        upper_j3d_loss = F.mse_loss(pred_upper, gt_upper, reduction="none")
        upper_j3d_loss = safe_masked_mean(upper_j3d_loss, mask[..., None, None])
        extra_loss += upper_j3d_loss * upper_j3d_weight
        extra_loss_dict["upper_j3d_loss_ego"] = upper_j3d_loss

    upper_limb_weight = _loss_weight(weights, "wearer_upper_limb", "ego_upper_limb")
    if upper_limb_weight > 0:
        limb_edges = torch.as_tensor(
            args.get(
                "ego_upper_limb_edges",
                [[12, 13], [13, 16], [16, 18], [18, 20], [12, 14], [14, 17], [17, 19], [19, 21], [16, 17]],
            ),
            device=pred_cr_j3d.device,
            dtype=torch.long,
        )
        pred_limb = pred_cr_j3d.index_select(-2, limb_edges[:, 1]) - pred_cr_j3d.index_select(-2, limb_edges[:, 0])
        gt_limb = gt_cr_j3d.index_select(-2, limb_edges[:, 1]) - gt_cr_j3d.index_select(-2, limb_edges[:, 0])
        upper_limb_loss = F.smooth_l1_loss(pred_limb, gt_limb, reduction="none", beta=0.03)
        upper_limb_loss = safe_masked_mean(upper_limb_loss, mask[..., None, None])
        extra_loss += upper_limb_loss * upper_limb_weight
        extra_loss_dict["upper_limb_loss_ego"] = upper_limb_loss

    lower_j3d_weight = _loss_weight(weights, "wearer_lower_j3d", "ego_lower_j3d")
    lower_ids = torch.as_tensor(
        args.get("ego_lower_joint_ids", [1, 2, 4, 5, 7, 8, 10, 11]),
        device=pred_cr_j3d.device,
        dtype=torch.long,
    )
    if lower_j3d_weight > 0:
        pred_lower = pred_cr_j3d.index_select(-2, lower_ids)
        gt_lower = gt_cr_j3d.index_select(-2, lower_ids)
        lower_j3d_loss = F.mse_loss(pred_lower, gt_lower, reduction="none")
        lower_j3d_loss = safe_masked_mean(lower_j3d_loss, mask[..., None, None])
        extra_loss += lower_j3d_loss * lower_j3d_weight
        extra_loss_dict["lower_j3d_loss_ego"] = lower_j3d_loss

    lower_limb_weight = _loss_weight(weights, "wearer_lower_limb", "ego_lower_limb")
    if lower_limb_weight > 0:
        lower_edges = torch.as_tensor(
            args.get("ego_lower_limb_edges", [[1, 4], [4, 7], [7, 10], [2, 5], [5, 8], [8, 11], [1, 2]]),
            device=pred_cr_j3d.device,
            dtype=torch.long,
        )
        pred_lower_limb = pred_cr_j3d.index_select(-2, lower_edges[:, 1]) - pred_cr_j3d.index_select(
            -2, lower_edges[:, 0]
        )
        gt_lower_limb = gt_cr_j3d.index_select(-2, lower_edges[:, 1]) - gt_cr_j3d.index_select(
            -2, lower_edges[:, 0]
        )
        lower_limb_loss = F.smooth_l1_loss(pred_lower_limb, gt_lower_limb, reduction="none", beta=0.03)
        lower_limb_loss = safe_masked_mean(lower_limb_loss, mask[..., None, None])
        extra_loss += lower_limb_loss * lower_limb_weight
        extra_loss_dict["lower_limb_loss_ego"] = lower_limb_loss

    foot_rel_vel_weight = _loss_weight(weights, "wearer_foot_rel_vel", "ego_foot_rel_vel")
    if foot_rel_vel_weight > 0 and pred_cr_j3d.size(1) > 1:
        frame_dt = max(float(args.get("frame_dt", 0.033)), 1e-6)
        foot_ids = torch.as_tensor(
            args.get("floor_joint_ids", [7, 10, 8, 11]), device=pred_cr_j3d.device, dtype=torch.long
        )
        pred_rel_foot = pred_cr_j3d.index_select(-2, foot_ids)
        gt_rel_foot = gt_cr_j3d.index_select(-2, foot_ids)
        pred_rel_foot_vel = (pred_rel_foot[:, 1:] - pred_rel_foot[:, :-1]) / frame_dt
        gt_rel_foot_vel = (gt_rel_foot[:, 1:] - gt_rel_foot[:, :-1]) / frame_dt
        foot_vel_mask = mask[:, 1:] & mask[:, :-1]
        foot_rel_vel_loss = F.smooth_l1_loss(
            pred_rel_foot_vel, gt_rel_foot_vel, reduction="none", beta=0.05
        )
        foot_rel_vel_loss = safe_masked_mean(foot_rel_vel_loss, foot_vel_mask[..., None, None])
        extra_loss += foot_rel_vel_loss * foot_rel_vel_weight
        extra_loss_dict["foot_rel_vel_loss_ego"] = foot_rel_vel_loss

    cr_verts_weight = _loss_weight(weights, "wearer_cr_verts", "cr_verts")
    if cr_verts_weight > 0:
        pred_verts437, pred_j17 = endecoder.smplx_model(**_body_smpl_params(pred_smpl_params))
        pred_root = pred_j17[:, :, [11, 12], :].mean(-2, keepdim=True)
        pred_cr_verts437 = pred_verts437 - pred_root

        gt_verts437, gt_j17 = endecoder.smplx_model(**_body_smpl_params(gt_smpl_params))
        gt_root = gt_j17[:, :, [11, 12], :].mean(-2, keepdim=True)
        gt_cr_verts437 = gt_verts437 - gt_root

        cr_vert_loss = F.mse_loss(pred_cr_verts437, gt_cr_verts437, reduction="none")
        cr_vert_loss = safe_masked_mean(cr_vert_loss, mask[:, :, None, None])
        extra_loss += cr_vert_loss * cr_verts_weight
        extra_loss_dict["cr_verts_loss_ego"] = cr_vert_loss

    return extra_loss, extra_loss_dict


def compute_extra_global_loss_ego(inputs, outputs, ppl):
    """Ego world loss after lifting CPF-local prediction by T_world_cpf."""
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    ego_targets = _targets_for_ego_world(inputs)
    mask = _mask_for_role(inputs, "ego").clone()
    spv_mask = inputs["mask"].get("spv_incam_only", torch.zeros_like(mask, dtype=torch.bool))
    if spv_mask.ndim == 1:
        spv_mask = spv_mask[:, None].expand_as(mask)
    mask = mask & (~spv_mask.bool())

    model_output = outputs["model_output"]
    static_conf_logits_ego = model_output.get("static_conf_logits_ego", None)
    pred_world = outputs.get("pred_smpl_params_global_ego", None)
    gt_transl = ego_targets["smpl_params_w"]["transl"]

    gt_w_j3d_for_head = None

    if pred_world is not None and _loss_weight(weights, "wearer_transl_w", "transl_w") > 0:
        trans_w_loss = F.l1_loss(pred_world["transl"], gt_transl, reduction="none")
        trans_w_loss = safe_masked_mean(trans_w_loss, mask[..., None])
        extra_loss += trans_w_loss * _loss_weight(weights, "wearer_transl_w", "transl_w")
        extra_loss_dict["transl_w_loss_ego"] = trans_w_loss

    first_transl_weight = _loss_weight(weights, "wearer_first_transl_w", "ego_first_transl_w")
    if pred_world is not None and first_transl_weight > 0:
        first_loss = F.l1_loss(pred_world["transl"][:, :1], gt_transl[:, :1], reduction="none")
        first_loss = safe_masked_mean(first_loss, mask[:, :1, None])
        extra_loss += first_loss * first_transl_weight
        extra_loss_dict["first_transl_w_loss_ego"] = first_loss

    local_vel_weight = _loss_weight(weights, "wearer_root_residual_vel_cpf", "ego_local_transl_vel")
    if local_vel_weight > 0 and "decode_dict_ego" in outputs and "root_residual_vel_cpf" in outputs["decode_dict_ego"]:
        ego_cond = inputs.get("ego_cond", {})
        gt_root_orient_cpf, gt_root_trans_cpf = _transform_root_to_local(
            ego_targets["smpl_params_w"]["global_orient"],
            ego_targets["smpl_params_w"]["transl"],
            ego_cond["T_world_cpf"],
            ego_targets["smpl_params_w"].get("betas"),
        )
        gt_residual_vel_cpf = get_local_transl_vel(gt_root_trans_cpf, gt_root_orient_cpf)
        residual_vel_loss = F.smooth_l1_loss(
            outputs["decode_dict_ego"]["root_residual_vel_cpf"],
            gt_residual_vel_cpf,
            reduction="none",
            beta=0.02,
        )
        residual_vel_loss = safe_masked_mean(residual_vel_loss, mask[..., None])
        extra_loss += residual_vel_loss * local_vel_weight
        extra_loss_dict["root_residual_vel_cpf_loss_ego"] = residual_vel_loss

        direct_vel_weight = _loss_weight(weights, "wearer_direct_vel_consistency", "ego_direct_vel_consistency")
        if direct_vel_weight > 0 and "root_trans_cpf" in outputs["decode_dict_ego"]:
            pred_direct_vel_cpf = get_local_transl_vel(
                outputs["decode_dict_ego"]["root_trans_cpf"],
                outputs["decode_dict_ego"]["root_orient_cpf"],
            )
            direct_vel_loss = F.smooth_l1_loss(
                pred_direct_vel_cpf,
                outputs["decode_dict_ego"]["root_residual_vel_cpf"],
                reduction="none",
                beta=0.02,
            )
            direct_vel_loss = safe_masked_mean(direct_vel_loss, mask[..., None])
            extra_loss += direct_vel_loss * direct_vel_weight
            extra_loss_dict["direct_vel_consistency_loss_ego"] = direct_vel_loss

    root_accel_weight = _loss_weight(weights, "wearer_root_accel", "ego_root_accel")
    if pred_world is not None and root_accel_weight > 0 and gt_transl.size(1) > 2:
        accel_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        pred_accel = pred_world["transl"][:, 2:] - 2.0 * pred_world["transl"][:, 1:-1] + pred_world["transl"][:, :-2]
        gt_accel = gt_transl[:, 2:] - 2.0 * gt_transl[:, 1:-1] + gt_transl[:, :-2]
        root_accel_loss = F.smooth_l1_loss(pred_accel, gt_accel, reduction="none", beta=0.02)
        root_accel_loss = safe_masked_mean(root_accel_loss, accel_mask[..., None])
        extra_loss += root_accel_loss * root_accel_weight
        extra_loss_dict["root_accel_loss_ego"] = root_accel_loss

    pose_accel_weight = _loss_weight(weights, "wearer_pose_accel", "ego_pose_accel")
    if pred_world is not None and pose_accel_weight > 0 and pred_world["body_pose"].size(1) > 2:
        accel_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        pred_pose_accel = pred_world["body_pose"][:, 2:] - 2.0 * pred_world["body_pose"][:, 1:-1] + pred_world["body_pose"][:, :-2]
        gt_pose = ego_targets["smpl_params_w"]["body_pose"]
        gt_pose_accel = gt_pose[:, 2:] - 2.0 * gt_pose[:, 1:-1] + gt_pose[:, :-2]
        pose_accel_loss = F.smooth_l1_loss(pred_pose_accel, gt_pose_accel, reduction="none", beta=0.02)
        pose_accel_loss = safe_masked_mean(pose_accel_loss, accel_mask[..., None])
        extra_loss += pose_accel_loss * pose_accel_weight
        extra_loss_dict["pose_accel_loss_ego"] = pose_accel_loss

    floor_weight = _loss_weight(weights, "wearer_floor", "ego_floor")
    floor_pen_weight = _loss_weight(weights, "wearer_floor_penetration", "ego_floor_penetration", floor_weight)
    pred_floor_world = pred_world
    if pred_floor_world is not None and (floor_weight > 0 or floor_pen_weight > 0):
        pred_floor_j3d = endecoder.fk_v2(**_body_smpl_params(pred_floor_world))
        gt_floor_j3d = endecoder.fk_v2(**_body_smpl_params(ego_targets["smpl_params_w"]))
        foot_ids = torch.as_tensor(args.get("floor_joint_ids", [7, 10, 8, 11]), device=pred_floor_j3d.device, dtype=torch.long)
        pred_foot_y = pred_floor_j3d.index_select(-2, foot_ids)[..., 1]
        gt_foot_y = gt_floor_j3d.index_select(-2, foot_ids)[..., 1]
        gt_floor_y = gt_foot_y.min(dim=-1).values.detach()
        pred_floor_y = pred_foot_y.min(dim=-1).values
        if floor_weight > 0:
            floor_loss = F.smooth_l1_loss(pred_floor_y, gt_floor_y, reduction="none", beta=0.05)
            floor_loss = safe_masked_mean(floor_loss, mask)
            extra_loss += floor_loss * floor_weight
            extra_loss_dict["floor_loss_ego"] = floor_loss
        if floor_pen_weight > 0:
            penetration_margin = float(args.get("floor_penetration_margin", 0.03))
            floor_pen_loss = F.relu(gt_floor_y - pred_floor_y - penetration_margin)
            floor_pen_loss = safe_masked_mean(floor_pen_loss, mask)
            extra_loss += floor_pen_loss * floor_pen_weight
            extra_loss_dict["floor_penetration_loss_ego"] = floor_pen_loss

    ego_head_weight = _loss_weight(weights, "wearer_head_trans", "ego_head_trans")
    if pred_world is not None and ego_head_weight > 0 and "ego_cond" in inputs:
        ego_cond = inputs["ego_cond"]
        if "T_world_head" in ego_cond:
            pred_w_j3d = endecoder.fk_v2(**_body_smpl_params(pred_world))
            head_joint_id = int(args.get("ego_head_joint_id", 15))
            pred_head = pred_w_j3d[:, :, head_joint_id]
            head_loss_target = args.get("ego_head_loss_target", "smpl_head_gt")
            if head_loss_target == "sensor_head":
                gt_head = ego_cond["T_world_head"][..., :3, 3]
            elif head_loss_target == "smpl_head_gt":
                if gt_w_j3d_for_head is None:
                    gt_w_j3d_for_head = endecoder.fk_v2(**_body_smpl_params(ego_targets["smpl_params_w"]))
                gt_head = gt_w_j3d_for_head[:, :, head_joint_id]
            else:
                raise ValueError(f"Unknown ego_head_loss_target={head_loss_target}")
            head_mask = mask
            if "head_valid" in ego_cond:
                head_mask = head_mask & ego_cond["head_valid"]
            head_loss = F.l1_loss(pred_head, gt_head, reduction="none")
            head_loss = safe_masked_mean(head_loss, head_mask[..., None])
            extra_loss += head_loss * ego_head_weight
            extra_loss_dict["ego_head_trans_loss"] = head_loss

    static_weight = _loss_weight(weights, "wearer_static_conf_bce", "static_conf_bce")
    foot_sliding_weight = _loss_weight(weights, "wearer_foot_sliding", "ego_foot_sliding")
    if (static_weight > 0 and static_conf_logits_ego is not None) or (foot_sliding_weight > 0 and pred_world is not None):
        vel_thr = args.static_conf.vel_thr
        assert vel_thr > 0
        frame_dt = max(float(args.get("frame_dt", 0.033)), 1e-6)
        foot_ids = torch.as_tensor(
            args.get("floor_joint_ids", [7, 10, 8, 11]), device=gt_transl.device, dtype=torch.long
        )
        wrist_ids = torch.as_tensor([20, 21], device=gt_transl.device, dtype=torch.long)
        gt_w_j3d = endecoder.fk_v2(**_body_smpl_params(ego_targets["smpl_params_w"]))
        static_all = get_static_joint_mask(gt_w_j3d, vel_thr=vel_thr, repeat_last=True)
        gt_foot_y = gt_w_j3d.index_select(-2, foot_ids)[..., 1]
        valid_foot_y = gt_foot_y.masked_fill(~mask[..., None], torch.inf)
        floor_y = valid_foot_y.amin(dim=(1, 2), keepdim=True)
        fallback_floor_y = gt_foot_y.amin(dim=(1, 2), keepdim=True)
        floor_y = torch.where(torch.isfinite(floor_y), floor_y, fallback_floor_y).detach()
        contact_height_thr = float(args.get("foot_contact_height_thr", 0.08))
        foot_contact = static_all.index_select(-1, foot_ids) & (gt_foot_y <= floor_y + contact_height_thr)
        wrist_static = static_all.index_select(-1, wrist_ids)
        static_gt = torch.cat([foot_contact, wrist_static], dim=-1).float()

        if static_weight > 0 and static_conf_logits_ego is not None:
            static_conf_loss = F.binary_cross_entropy_with_logits(static_conf_logits_ego, static_gt, reduction="none")
            static_mask = mask[..., None].expand_as(static_conf_loss)
            static_conf_loss = (static_conf_loss * static_mask).sum() / static_mask.sum().clamp(min=1)
            extra_loss += static_conf_loss * static_weight
            extra_loss_dict["static_conf_loss_ego"] = static_conf_loss
            extra_loss_dict["foot_contact_ratio_ego"] = foot_contact.float().mean().detach()

        if foot_sliding_weight > 0 and pred_world is not None and gt_w_j3d.size(1) > 1:
            pred_w_j3d = endecoder.fk_v2(**_body_smpl_params(pred_world))
            pred_foot = pred_w_j3d.index_select(-2, foot_ids)
            foot_vel_mps = (pred_foot[:, 1:] - pred_foot[:, :-1]) / frame_dt
            contact = foot_contact.float()
            contact_pair = contact[:, 1:] * contact[:, :-1]
            foot_mask = (mask[:, 1:] & mask[:, :-1]).float()[..., None]
            foot_sliding_loss = foot_vel_mps.norm(dim=-1) * contact_pair * foot_mask
            denom = (contact_pair * foot_mask).sum().clamp(min=1.0)
            foot_sliding_loss = foot_sliding_loss.sum() / denom
            extra_loss += foot_sliding_loss * foot_sliding_weight
            extra_loss_dict["foot_sliding_loss_ego"] = foot_sliding_loss

    return extra_loss, extra_loss_dict


def compute_extra_incam_loss(inputs, outputs, ppl):
    model_output = outputs["model_output"]
    decode_dict = outputs["decode_dict"]
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"]  # effective length mask
    mask_reproj = ~inputs["mask"]["spv_incam_only"]  # do not supervise reproj for 3DPW

    # Exo head only (no ego fallback)
    pred_smpl_params = outputs["pred_smpl_params_incam"]
    pred_cam = model_output["pred_cam"]

    # Incam FK
    # prediction
    pred_c_j3d = endecoder.fk_v2(**_body_smpl_params(pred_smpl_params))
    pred_cr_j3d = pred_c_j3d - pred_c_j3d[:, :, :1]  # (B, L, J, 3)
    # gt
    gt_c_j3d = endecoder.fk_v2(**_body_smpl_params(inputs["interactee_smpl_params_c"]))  # (B, L, J, 3)
    gt_cr_j3d = gt_c_j3d - gt_c_j3d[:, :, :1]  # (B, L, J, 3)

    # Root aligned C-MPJPE Loss
    cr_j3d_weight = _loss_weight(weights, "partner_cr_j3d", "cr_j3d")
    if cr_j3d_weight > 0.0:
        cr_j3d_loss = F.mse_loss(pred_cr_j3d, gt_cr_j3d, reduction="none")
        cr_j3d_loss = safe_masked_mean(cr_j3d_loss, mask[..., None, None])
        extra_loss += cr_j3d_loss * cr_j3d_weight
        extra_loss_dict["cr_j3d_loss"] = cr_j3d_loss

    # Reprojection (to align with image)
    transl_c_weight = _loss_weight(weights, "partner_transl_c", "transl_c")
    if transl_c_weight > 0.0:
        # pred_transl = decode_dict["transl"]  # (B, L, 3)
        # gt_transl = inputs["smpl_params_c"]["transl"]
        # transl_c_loss = F.l1_loss(pred_transl, gt_transl, reduction="none")
        # transl_c_loss = (transl_c_loss * mask[..., None]).mean()

        # Instead of supervising transl, we convert gt to pred_cam (prevent divide 0)
        gt_transl = inputs["interactee_smpl_params_c"]["transl"]  # (B, L, 3)
        gt_pred_cam = get_a_pred_cam(gt_transl, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        gt_pred_cam[gt_pred_cam.isinf()] = -1  # this will be handled by valid_mask
        # (compute_transl_full_cam(gt_pred_cam, inputs["bbx_xys"], inputs["K_fullimg"]) - gt_transl).abs().max()

        # Skip gts that are not good during random construction
        gt_j3d_z_min = inputs["gt_j3d"][..., 2].min(dim=-1)[0]
        valid_mask = (
            (gt_j3d_z_min > 0.3)
            * (gt_pred_cam[..., 0] > 0.3)
            * (gt_pred_cam[..., 0] < 5.0)
            * (gt_pred_cam[..., 1] > -3.0)
            * (gt_pred_cam[..., 1] < 3.0)
            * (gt_pred_cam[..., 2] > -3.0)
            * (gt_pred_cam[..., 2] < 3.0)
            * (inputs["bbx_xys"][..., 2] > 0)
        )[..., None]
        transl_c_loss = F.mse_loss(pred_cam, gt_pred_cam, reduction="none")
        transl_c_loss = safe_masked_mean(transl_c_loss, mask[..., None] * valid_mask)

        extra_loss_dict["transl_c_loss"] = transl_c_loss
        extra_loss += transl_c_loss * transl_c_weight

    j2d_weight = _loss_weight(weights, "partner_j2d", "j2d")
    if j2d_weight > 0.0:
        # prevent divide 0 or small value to overflow(fp16)
        reproj_z_thr = 0.3
        pred_c_j3d_z0_mask = pred_c_j3d[..., 2].abs() <= reproj_z_thr
        pred_c_j3d[pred_c_j3d_z0_mask] = reproj_z_thr
        gt_c_j3d_z0_mask = gt_c_j3d[..., 2].abs() <= reproj_z_thr
        gt_c_j3d[gt_c_j3d_z0_mask] = reproj_z_thr

        pred_j2d_01 = project_to_bi01(pred_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"])
        gt_j2d_01 = project_to_bi01(gt_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, J, 2)

        valid_mask = (
            (gt_c_j3d[..., 2] > reproj_z_thr)
            * (pred_c_j3d[..., 2] > reproj_z_thr)  # Be safe
            * (gt_j2d_01[..., 0] > 0.0)
            * (gt_j2d_01[..., 0] < 1.0)
            * (gt_j2d_01[..., 1] > 0.0)
            * (gt_j2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False  # Do not supervise on 3dpw
        j2d_loss = F.mse_loss(pred_j2d_01, gt_j2d_01, reduction="none")
        vis_weight = _frame_visibility_weight_from_kp2d(inputs.get("kp2d", None), j2d_loss.ndim)
        if vis_weight is not None:
            valid_mask = valid_mask * vis_weight
        j2d_loss = safe_masked_mean(j2d_loss, mask[..., None, None] * valid_mask)

        extra_loss += j2d_loss * j2d_weight
        extra_loss_dict["j2d_loss"] = j2d_loss

    cr_verts_weight = _loss_weight(weights, "partner_cr_verts", "cr_verts")
    if cr_verts_weight > 0:
        # SMPL forward
        pred_c_verts437, pred_c_j17 = endecoder.smplx_model(**_body_smpl_params(pred_smpl_params))
        root_ = pred_c_j17[:, :, [11, 12], :].mean(-2, keepdim=True)
        pred_cr_verts437 = pred_c_verts437 - root_

        gt_cr_verts437 = inputs["gt_cr_verts437"]  # (B, L, 437, 3)
        cr_vert_loss = F.mse_loss(pred_cr_verts437, gt_cr_verts437, reduction="none")
        cr_vert_loss = safe_masked_mean(cr_vert_loss, mask[:, :, None, None])
        extra_loss += cr_vert_loss * cr_verts_weight
        extra_loss_dict["cr_vert_loss"] = cr_vert_loss

    verts2d_weight = _loss_weight(weights, "partner_verts2d", "verts2d")
    if verts2d_weight > 0:
        gt_c_verts437 = inputs["gt_c_verts437"]  # (B, L, 437, 3)

        # prevent divide 0 or small value to overflow(fp16)
        reproj_z_thr = 0.3
        pred_c_verts437_z0_mask = pred_c_verts437[..., 2].abs() <= reproj_z_thr
        pred_c_verts437[pred_c_verts437_z0_mask] = reproj_z_thr
        gt_c_verts437_z0_mask = gt_c_verts437[..., 2].abs() <= reproj_z_thr
        gt_c_verts437[gt_c_verts437_z0_mask] = reproj_z_thr

        pred_verts2d_01 = project_to_bi01(pred_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"])
        gt_verts2d_01 = project_to_bi01(gt_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 437, 2)

        valid_mask = (
            (gt_c_verts437[..., 2] > reproj_z_thr)
            * (pred_c_verts437[..., 2] > reproj_z_thr)  # Be safe
            * (gt_verts2d_01[..., 0] > 0.0)
            * (gt_verts2d_01[..., 0] < 1.0)
            * (gt_verts2d_01[..., 1] > 0.0)
            * (gt_verts2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False  # Do not supervise on 3dpw
        verts2d_loss = F.mse_loss(pred_verts2d_01, gt_verts2d_01, reduction="none")
        verts2d_loss = safe_masked_mean(verts2d_loss, mask[..., None, None] * valid_mask)

        extra_loss += verts2d_loss * verts2d_weight
        extra_loss_dict["verts2d_loss"] = verts2d_loss

    return extra_loss, extra_loss_dict


def compute_extra_global_loss(inputs, outputs, ppl):
    decode_dict = outputs["decode_dict"]
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"].clone()  # (B, L)
    mask[inputs["mask"]["spv_incam_only"]] = False

    # Exo head only (no ego fallback)
    model_output = outputs["model_output"]
    static_conf_logits = model_output["static_conf_logits"]

    transl_w_weight = _loss_weight(weights, "partner_transl_w", "transl_w")
    if transl_w_weight > 0:
        gt_transl_w = inputs["interactee_smpl_params_w"]["transl"]
        loss_mode = args.get("exo_world_loss_mode", "rollout_gt_init")
        if loss_mode == "incam_kinect":
            if "pred_smpl_params_kinect_from_incam" not in outputs:
                outputs["pred_smpl_params_kinect_from_incam"] = _smpl_params_incam_to_input_world(
                    outputs["pred_smpl_params_incam"], inputs
                )
            pred_transl_w = outputs["pred_smpl_params_kinect_from_incam"]["transl"]
        elif loss_mode == "rollout_gt_init":
            gt_global_orient_w = inputs["interactee_smpl_params_w"]["global_orient"]
            local_transl_vel = decode_dict["local_transl_vel"]
            pred_transl_w = rollout_local_transl_vel(local_transl_vel, gt_global_orient_w, gt_transl_w[:, [0]])
        else:
            raise ValueError(f"Unknown exo_world_loss_mode={loss_mode}")

        trans_w_loss = F.l1_loss(pred_transl_w, gt_transl_w, reduction="none")
        trans_w_loss = safe_masked_mean(trans_w_loss, mask[..., None])
        extra_loss += trans_w_loss * transl_w_weight
        extra_loss_dict["transl_w_loss"] = trans_w_loss

    # Static-Conf loss
    static_conf_weight = _loss_weight(weights, "partner_static_conf_bce", "static_conf_bce")
    if static_conf_weight > 0:
        # Compute gt by thresholding velocity
        vel_thr = args.static_conf.vel_thr
        assert vel_thr > 0
        joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        gt_w_j3d = endecoder.fk_v2(**_body_smpl_params(inputs["interactee_smpl_params_w"]))  # (B, L, J=22, 3)
        static_gt = get_static_joint_mask(gt_w_j3d, vel_thr=vel_thr, repeat_last=True)  # (B, L, J)
        static_gt = static_gt[:, :, joint_ids].float()  # (B, L, J')
        pred_static_conf_logits = static_conf_logits

        static_conf_loss = F.binary_cross_entropy_with_logits(pred_static_conf_logits, static_gt, reduction="none")
        static_conf_loss = safe_masked_mean(static_conf_loss, mask[..., None])
        extra_loss += static_conf_loss * static_conf_weight
        extra_loss_dict["static_conf_loss"] = static_conf_loss

    return extra_loss, extra_loss_dict


@autocast(enabled=False)
def get_smpl_params_w_Rt_v2(
    global_orient_gv,
    local_transl_vel,
    global_orient_c,
    cam_angvel,
):
    """Get global R,t in GV0(ay)
    Args:
        cam_angvel: (B, L, 6), defined as R @ R_{w2c}^{t} = R_{w2c}^{t+1}
    """

    # Get R_ct_to_c0 from cam_angvel
    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
        R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
        return R

    B = cam_angvel.shape[0]
    R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)
    R_t_to_tp1 = as_identity(R_t_to_tp1)

    # Get R_c2gv
    R_gv = axis_angle_to_matrix(global_orient_gv)  # (B, L, 3, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (B, L, 3, 3)

    # Camera view direction in GV coordinate: Rc2gv @ [0,0,1]
    R_c2gv = R_gv @ R_c.mT
    view_axis_gv = R_c2gv[:, :, :, 2]  # (B, L, 3)  Rc2gv is estimated, so the x-axis is not accurate, i.e. != 0

    # Rotate axis use camera relative rotation
    R_cnext2gv = R_c2gv @ R_t_to_tp1.mT
    view_axis_gv_next = R_cnext2gv[..., 2]

    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = F.normalize(vec1_xyz, dim=-1)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = F.normalize(vec2_xyz, dim=-1)

    aa_tp1_to_t = vec2_xyz.cross(vec1_xyz, dim=-1)
    aa_tp1_to_t_angle = torch.acos(torch.clamp((vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True), -1.0, 1.0))
    aa_tp1_to_t = F.normalize(aa_tp1_to_t, dim=-1) * aa_tp1_to_t_angle

    aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)  # Smooth
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT  # (B, L, 3)

    # Get R_t_to_0
    R_t_to_0 = [torch.eye(3)[None].expand(B, -1, -1).to(R_t_to_tp1)]
    for i in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, i])
    R_t_to_0 = torch.stack(R_t_to_0, dim=1)  # (B, L, 3, 3)
    R_t_to_0 = as_identity(R_t_to_0)

    global_orient = matrix_to_axis_angle(R_t_to_0 @ R_gv)

    # Rollout to global transl
    # Start from transl0, in gv0 -> flip y-axis of gv0
    transl = rollout_local_transl_vel(local_transl_vel, global_orient)
    global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="any->ay")

    smpl_params_w_Rt = {"global_orient": global_orient, "transl": transl}
    return smpl_params_w_Rt
