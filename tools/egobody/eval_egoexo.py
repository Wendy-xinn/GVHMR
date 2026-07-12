import argparse
import json
from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hmr4d.configs import register_store_gvhmr
from hmr4d.datamodule.mocap_trainX_testY import collate_fn
from hmr4d.dataset.egobody.egobody_egoexo_v1 import EgoBodyEgoExoV1Dataset
from hmr4d.model.gvhmr.gvhmr_pl import _body_smpl_params
from hmr4d.utils.geo.hmr_cam import normalize_kp2d
from hmr4d.utils.net_utils import load_pretrained_model


def masked_mean(x, mask):
    mask = mask.to(x.device).bool()
    while mask.ndim < x.ndim:
        mask = mask[..., None]
    return (x * mask).sum() / mask.sum().clamp_min(1)


def procrustes_align(pred, gt, eps=1e-8):
    """Batched similarity alignment for (..., J, 3)."""
    orig_shape = pred.shape
    pred = pred.reshape(-1, orig_shape[-2], 3)
    gt = gt.reshape(-1, orig_shape[-2], 3)

    mu_pred = pred.mean(dim=1, keepdim=True)
    mu_gt = gt.mean(dim=1, keepdim=True)
    pred0 = pred - mu_pred
    gt0 = gt - mu_gt

    norm_pred = torch.linalg.norm(pred0.reshape(pred0.shape[0], -1), dim=1, keepdim=True).clamp_min(eps)
    norm_gt = torch.linalg.norm(gt0.reshape(gt0.shape[0], -1), dim=1, keepdim=True).clamp_min(eps)
    pred0n = pred0 / norm_pred[:, None]
    gt0n = gt0 / norm_gt[:, None]

    H = pred0n.transpose(1, 2) @ gt0n
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)
    R = V @ U.transpose(1, 2)
    det = torch.det(R)
    sign = torch.ones((R.shape[0], 3), device=R.device, dtype=R.dtype)
    sign[:, -1] = torch.where(det < 0, -1.0, 1.0)
    R = V @ torch.diag_embed(sign) @ U.transpose(1, 2)

    scale = (norm_gt / norm_pred).reshape(-1, 1, 1)
    aligned = scale * (pred0 @ R) + mu_gt
    return aligned.reshape(orig_shape)


def fk_joints(model, params):
    body_params = {k: v.float() for k, v in _body_smpl_params(params).items()}
    return model.pipeline.endecoder.fk_v2(**body_params).float()


def metrics_for_params(model, pred, gt, mask, prefix):
    pred_j = fk_joints(model, pred)
    gt_j = fk_joints(model, gt)
    n = min(pred_j.shape[2], gt_j.shape[2])
    pred_j = pred_j[:, :, :n]
    gt_j = gt_j[:, :, :n]

    global_err = torch.linalg.norm(pred_j - gt_j, dim=-1).mean(dim=-1)
    pred_ra = pred_j - pred_j[:, :, :1]
    gt_ra = gt_j - gt_j[:, :, :1]
    ra_err = torch.linalg.norm(pred_ra - gt_ra, dim=-1).mean(dim=-1)
    pa = procrustes_align(pred_j, gt_j)
    pa_err = torch.linalg.norm(pa - gt_j, dim=-1).mean(dim=-1)

    transl_err = torch.linalg.norm(pred["transl"] - gt["transl"], dim=-1)
    head_id = min(15, n - 1)
    head_err = torch.linalg.norm(pred_j[:, :, head_id] - gt_j[:, :, head_id], dim=-1)

    return {
        f"{prefix}_mpjpe_mm": masked_mean(global_err, mask).item() * 1000.0,
        f"{prefix}_ra_mpjpe_mm": masked_mean(ra_err, mask).item() * 1000.0,
        f"{prefix}_pa_mpjpe_mm": masked_mean(pa_err, mask).item() * 1000.0,
        f"{prefix}_transl_err_mm": masked_mean(transl_err, mask).item() * 1000.0,
        f"{prefix}_head_err_mm": masked_mean(head_err, mask).item() * 1000.0,
    }


def update_sum(sums, weights, vals, weight):
    for k, v in vals.items():
        sums[k] = sums.get(k, 0.0) + float(v) * weight
        weights[k] = weights.get(k, 0) + weight


def make_eval_batch(batch, branch_mode):
    is_paired = "exo" in batch and "ego" in batch
    if is_paired:
        primary = batch["ego"] if branch_mode == "ego" else batch["exo"]
        batch["smpl_params_c"] = primary["smpl_params_c"]
        batch["smpl_params_w"] = primary["smpl_params_w"]
        batch["interactee_smpl_params_c"] = primary["smpl_params_c"]
        batch["interactee_smpl_params_w"] = primary["smpl_params_w"]
        if branch_mode == "ego":
            batch["bbx_xys"] = primary.get("bbx_body_xys", primary["bbx_xys"])
            batch["kp2d"] = primary.get("kp2d_body", primary["kp2d"])
            batch["f_imgseq"] = primary.get("f_body_imgseq", primary["f_imgseq"])
            batch["K_fullimg"] = batch.get("K_ego", batch["K_fullimg"])
            batch["mask"]["valid"] = batch["mask"].get("ego_valid", batch["mask"]["valid"])
        else:
            batch["bbx_xys"] = primary["bbx_xys"]
            batch["kp2d"] = primary["kp2d"]
            batch["f_imgseq"] = primary["f_imgseq"]
            batch["mask"]["valid"] = batch["mask"].get("exo_valid", batch["mask"]["valid"])

    obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
    obs[~batch["mask"]["valid"]] = 0
    out = {
        "length": batch["length"],
        "obs": obs,
        "bbx_xys": batch["bbx_xys"],
        "K_fullimg": batch["K_fullimg"],
        "cam_angvel": batch["cam_angvel"],
        "R_c2gv": batch["R_c2gv"],
        "f_imgseq": batch["f_imgseq"],
        "mask": batch["mask"],
    }
    for key in ("T_world_cam", "T_world_exo_cam"):
        if key in batch:
            out[key] = batch[key]
    if is_paired:
        out["exo"] = batch["exo"]
        out["ego"] = batch["ego"]
        out["ego_cond"] = batch.get("ego_cond", {})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="gvhmr/egobody_egoexo_stage1")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="/public/home/wenxin/GVHMR/data")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--branch-mode", default="ego", choices=["ego", "exo", "both"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--motion-frames", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    register_store_gvhmr()
    config_dir = str((Path(__file__).resolve().parents[2] / "hmr4d" / "configs").resolve())
    with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = hydra.compose(config_name="train", overrides=[f"exp={args.exp}"])
    cfg.pipeline.args.branch_mode = args.branch_mode
    cfg.pipeline.args.enable_frozen_ego_image_exo = False
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, args.ckpt)
    model.eval().cuda()

    dataset = EgoBodyEgoExoV1Dataset(
        output_root=args.data_root,
        split=args.split,
        motion_frames=args.motion_frames,
        world_coord="kinect12",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    sums = {}
    weights = {}
    total_ego_weight = 0
    device = torch.device("cuda")
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc=f"eval {args.branch_mode}/{args.split}")):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            batch = move_to_device(batch, device)
            eval_batch = make_eval_batch(batch, args.branch_mode)
            outputs = model.pipeline.forward(eval_batch, train=False, postproc=False)

            if "pred_smpl_params_global_ego" in outputs:
                ego_mask = batch["mask"].get("ego_valid", batch["mask"]["valid"])
                vals = metrics_for_params(model, outputs["pred_smpl_params_global_ego"], batch["ego"]["smpl_params_w"], ego_mask, "ego")
                weight = int(ego_mask.sum().item())
                update_sum(sums, weights, vals, weight)
                total_ego_weight += weight

            if args.branch_mode in ("exo", "both"):
                exo_mask = batch["mask"].get("exo_valid", batch["mask"]["valid"])
                weight = int(exo_mask.sum().item())
                if "pred_smpl_params_incam" in outputs:
                    vals = metrics_for_params(
                        model, outputs["pred_smpl_params_incam"], batch["exo"]["smpl_params_c"], exo_mask, "exo_incam"
                    )
                    update_sum(sums, weights, vals, weight)
                if "pred_smpl_params_kinect_from_incam" in outputs:
                    vals = metrics_for_params(
                        model, outputs["pred_smpl_params_kinect_from_incam"], batch["exo"]["smpl_params_w"], exo_mask, "exo_world_from_incam"
                    )
                    update_sum(sums, weights, vals, weight)

    results = {k: v / max(weights.get(k, 0), 1) for k, v in sums.items()}
    results.update(
        {
            "ckpt": args.ckpt,
            "exp": args.exp,
            "split": args.split,
            "branch_mode": args.branch_mode,
            "num_samples": len(dataset),
            "weighted_ego_frames": total_ego_weight,
            "metric_weights": weights,
        }
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, sort_keys=True))


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    return x


if __name__ == "__main__":
    main()
