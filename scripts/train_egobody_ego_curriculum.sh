#!/usr/bin/env bash
set -euo pipefail

# EgoBody ego curriculum training for the ifcam branch.
#
# Stages:
#   1) pose   : learn local/body pose first, with weak global/locomotion terms.
#   2) loco   : continue from pose, focus on camera-anchored locomotion/contact.
#               By default freezes the shared GVHMR backbone/transformer and trains ego heads + new condition embedders.
#   3) finetune: continue from loco, unfreeze backbone with a smaller learning rate.
#
# Common usage:
#   cd /public/home/wenxin/GVHMR_ifcam
#   bash scripts/train_egobody_ego_curriculum.sh
#
# Useful overrides:
#   RUN_SUFFIX=curriculum2 POSE_EPOCHS=20 LOCO_EPOCHS=80 FT_EPOCHS=50 bash scripts/train_egobody_ego_curriculum.sh
#   DEVICES=1 BATCH_SIZE=16 NUM_WORKERS=8 bash scripts/train_egobody_ego_curriculum.sh
#   POSE_INIT_CKPT=/path/to/init.ckpt bash scripts/train_egobody_ego_curriculum.sh
#   SKIP_POSE=1 POSE_CKPT=/path/to/pose.ckpt bash scripts/train_egobody_ego_curriculum.sh
#   REUSE_POSE_RUN=curriculum1 RUN_SUFFIX=residual_coarse2 SKIP_FINETUNE=1 bash scripts/train_egobody_ego_curriculum.sh
#   REUSE_POSE_RUN=curriculum1 RUN_SUFFIX=exo_pose_init1 COPY_EXO_TO_EGO_AFTER_LOAD=true LOCO_UNFREEZE_BLOCKS=2 bash scripts/train_egobody_ego_curriculum.sh
#   REUSE_POSE_RUN=curriculum1 RUN_SUFFIX=cross_attn_teacher1 USE_CROSS_VIEW_TEACHER=true COPY_EXO_TO_EGO_AFTER_LOAD=true LOCO_UNFREEZE_BLOCKS=2 bash scripts/train_egobody_ego_curriculum.sh
#   SKIP_LOCO=1 LOCO_CKPT=/path/to/loco.ckpt bash scripts/train_egobody_ego_curriculum.sh
#   EXTRA_ARGS="model.val_vis_every_n_batches=80" bash scripts/train_egobody_ego_curriculum.sh

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN=${PYTHON_BIN:-/public/home/wenxin/miniconda3/envs/gvhmr/bin/python}
DEFAULT_DATA_ROOT="${REPO_ROOT}/data"
LEGACY_DATA_ROOT=/public/home/wenxin/GVHMR/data
DATA_ROOT=${DATA_ROOT:-${DEFAULT_DATA_ROOT}}
if [[ ! -d "${DATA_ROOT}" && "${DATA_ROOT}" == "${DEFAULT_DATA_ROOT}" && -d "${LEGACY_DATA_ROOT}" ]]; then
  echo "DATA_ROOT ${DATA_ROOT} not found; reading preprocessed data from ${LEGACY_DATA_ROOT}"
  DATA_ROOT=${LEGACY_DATA_ROOT}
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs}
DATA_NAME=${DATA_NAME:-egobody_egoexo_ifcam}
RUN_SUFFIX=${RUN_SUFFIX:-curriculum1}
DEVICES=${DEVICES:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}
VIS_EVERY=${VIS_EVERY:-500}
VAL_EVERY_EPOCHS=${VAL_EVERY_EPOCHS:-5}
VAL_VIS_EVERY=${VAL_VIS_EVERY:-80}

POSE_EPOCHS=${POSE_EPOCHS:-20}
LOCO_EPOCHS=${LOCO_EPOCHS:-80}
FT_EPOCHS=${FT_EPOCHS:-50}

POSE_INIT_CKPT=${POSE_INIT_CKPT:-}
POSE_CKPT=${POSE_CKPT:-}
REUSE_POSE_RUN=${REUSE_POSE_RUN:-}
LOCO_CKPT=${LOCO_CKPT:-}
SKIP_POSE=${SKIP_POSE:-0}
SKIP_LOCO=${SKIP_LOCO:-0}
SKIP_FINETUNE=${SKIP_FINETUNE:-0}

POSE_FREEZE_BACKBONE=${POSE_FREEZE_BACKBONE:-true}
POSE_UNFREEZE_BLOCKS=${POSE_UNFREEZE_BLOCKS:-2}
POSE_BACKBONE_LR_SCALE=${POSE_BACKBONE_LR_SCALE:-0.01}

LOCO_FREEZE_BACKBONE=${LOCO_FREEZE_BACKBONE:-true}
LOCO_UNFREEZE_BLOCKS=${LOCO_UNFREEZE_BLOCKS:-0}
LOCO_BACKBONE_LR_SCALE=${LOCO_BACKBONE_LR_SCALE:-0.003}

FT_FREEZE_BACKBONE=${FT_FREEZE_BACKBONE:-false}
FT_UNFREEZE_BLOCKS=${FT_UNFREEZE_BLOCKS:-2}
FT_LR=${FT_LR:-5e-5}
FT_BACKBONE_LR_SCALE=${FT_BACKBONE_LR_SCALE:-0.1}

COPY_EXO_TO_EGO=${COPY_EXO_TO_EGO:-false}
COPY_EXO_TO_EGO_AFTER_LOAD=${COPY_EXO_TO_EGO_AFTER_LOAD:-false}
COPY_EXO_TO_EGO_MODE=${COPY_EXO_TO_EGO_MODE:-pose}
USE_CROSS_VIEW_TEACHER=${USE_CROSS_VIEW_TEACHER:-false}
CROSS_VIEW_HEADS=${CROSS_VIEW_HEADS:-4}

# Optional whitespace-separated Hydra overrides applied to all stages.
# Example: EXTRA_ARGS="pl_trainer.limit_val_batches=8 model.val_vis_every_n_batches=20"
EXTRA_ARGS=${EXTRA_ARGS:-}
USER_OVERRIDES=()
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  USER_OVERRIDES=(${EXTRA_ARGS})
fi
for arg in "$@"; do
  if [[ "${arg}" == ckpt_path=* ]]; then
    POSE_INIT_CKPT="${arg#ckpt_path=}"
  else
    USER_OVERRIDES+=("${arg}")
  fi
done

COMMON_OVERRIDES=(
  data_name=${DATA_NAME}
  ++data.dataset_opts.train.egobody_egoexo_train.output_root=${DATA_ROOT}
  ++data.dataset_opts.val.egobody_egoexo_val.output_root=${DATA_ROOT}
  data.loader_opts.train.batch_size=${BATCH_SIZE}
  data.loader_opts.val.batch_size=1
  data.loader_opts.train.num_workers=${NUM_WORKERS}
  data.loader_opts.val.num_workers=${NUM_WORKERS}
  pl_trainer.devices=${DEVICES}
  pl_trainer.check_val_every_n_epoch=${VAL_EVERY_EPOCHS}
  model.vis_every_n_steps=${VIS_EVERY}
  model.copy_exo_to_ego=${COPY_EXO_TO_EGO}
  model.copy_exo_to_ego_after_load=${COPY_EXO_TO_EGO_AFTER_LOAD}
  model.copy_exo_to_ego_mode=${COPY_EXO_TO_EGO_MODE}
  model.val_vis_every_n_batches=${VAL_VIS_EVERY}
  pipeline.args.branch_mode=both
  pipeline.args.input_role=ego
  pipeline.args.train_input_role=ego
  pipeline.args.supervise_role=ego
  pipeline.args.enable_frozen_ego_image_exo=false
  pipeline.args.use_interaction_condition=true
  pipeline.args.use_cross_view_teacher=${USE_CROSS_VIEW_TEACHER}
  network.cross_view_fusion=${USE_CROSS_VIEW_TEACHER}
  network.cross_view_heads=${CROSS_VIEW_HEADS}
)

POSE_OVERRIDES=(
  model.freeze_backbone=${POSE_FREEZE_BACKBONE}
  model.freeze_exo_head=true
  model.freeze_ego_head=false
  model.unfreeze_last_n_blocks=${POSE_UNFREEZE_BLOCKS}
  model.backbone_lr_scale=${POSE_BACKBONE_LR_SCALE}
  pl_trainer.max_epochs=${POSE_EPOCHS}
  pipeline.args.weights.cr_j3d=350.
  pipeline.args.weights.cr_verts=250.
  pipeline.args.weights.transl_w=0.05
  pipeline.args.weights.static_conf_bce=0.05
  pipeline.args.weights.ego_head_trans=20.
  pipeline.args.weights.ego_upper_j3d=250.
  pipeline.args.weights.ego_upper_limb=150.
  pipeline.args.weights.ego_first_transl_w=5.
  pipeline.args.weights.ego_floor=0.
  pipeline.args.weights.ego_floor_penetration=0.
  pipeline.args.weights.ego_foot_sliding=0.
  pipeline.args.weights.ego_root_accel=0.
  pipeline.args.weights.ego_pose_accel=0.
  pipeline.args.weights.ego_local_transl_vel=0.
  pipeline.args.weights.ego_direct_vel_consistency=0.
)

LOCO_OVERRIDES=(
  model.freeze_backbone=${LOCO_FREEZE_BACKBONE}
  model.freeze_exo_head=true
  model.freeze_ego_head=false
  model.unfreeze_last_n_blocks=${LOCO_UNFREEZE_BLOCKS}
  model.backbone_lr_scale=${LOCO_BACKBONE_LR_SCALE}
  pl_trainer.max_epochs=${LOCO_EPOCHS}
  pipeline.args.weights.cr_j3d=150.
  pipeline.args.weights.cr_verts=100.
  pipeline.args.weights.transl_w=0.05
  pipeline.args.weights.static_conf_bce=0.5
  pipeline.args.weights.ego_head_trans=50.
  pipeline.args.weights.ego_upper_j3d=150.
  pipeline.args.weights.ego_upper_limb=80.
  pipeline.args.weights.ego_first_transl_w=20.
  pipeline.args.weights.ego_floor=0.5
  pipeline.args.weights.ego_floor_penetration=1.0
  pipeline.args.weights.ego_foot_sliding=50.0
  pipeline.args.weights.ego_root_accel=10.0
  pipeline.args.weights.ego_pose_accel=2.0
  pipeline.args.weights.ego_local_transl_vel=100.
  pipeline.args.weights.ego_direct_vel_consistency=20.
)

FT_OVERRIDES=(
  optimizer.lr=${FT_LR}
  model.freeze_backbone=${FT_FREEZE_BACKBONE}
  model.freeze_exo_head=true
  model.freeze_ego_head=false
  model.unfreeze_last_n_blocks=${FT_UNFREEZE_BLOCKS}
  model.backbone_lr_scale=${FT_BACKBONE_LR_SCALE}
  pl_trainer.max_epochs=${FT_EPOCHS}
  pipeline.args.weights.cr_j3d=220.
  pipeline.args.weights.cr_verts=160.
  pipeline.args.weights.transl_w=0.05
  pipeline.args.weights.static_conf_bce=0.5
  pipeline.args.weights.ego_head_trans=50.
  pipeline.args.weights.ego_upper_j3d=180.
  pipeline.args.weights.ego_upper_limb=100.
  pipeline.args.weights.ego_first_transl_w=20.
  pipeline.args.weights.ego_floor=0.5
  pipeline.args.weights.ego_floor_penetration=1.0
  pipeline.args.weights.ego_foot_sliding=35.0
  pipeline.args.weights.ego_root_accel=6.0
  pipeline.args.weights.ego_pose_accel=1.0
  pipeline.args.weights.ego_local_transl_vel=100.
  pipeline.args.weights.ego_direct_vel_consistency=20.
)

RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${DATA_NAME}/${RUN_SUFFIX}"
POSE_EXP_NAME=pose
LOCO_EXP_NAME=loco
FT_EXP_NAME=finetune
POSE_OUTPUT_DIR="${RUN_OUTPUT_DIR}/pose"
LOCO_OUTPUT_DIR="${RUN_OUTPUT_DIR}/loco"
FT_OUTPUT_DIR="${RUN_OUTPUT_DIR}/finetune"
mkdir -p "${RUN_OUTPUT_DIR}"

latest_ckpt() {
  local ckpt_dir="$1"
  find "${ckpt_dir}" -maxdepth 1 -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true
}

if [[ -n "${REUSE_POSE_RUN}" ]]; then
  REUSE_POSE_DIR="${OUTPUT_ROOT}/${DATA_NAME}/${REUSE_POSE_RUN}/pose/checkpoints"
  POSE_CKPT="$(latest_ckpt "${REUSE_POSE_DIR}")"
  if [[ -z "${POSE_CKPT}" ]]; then
    echo "ERROR: no reusable pose checkpoint found under ${REUSE_POSE_DIR}" >&2
    exit 1
  fi
  SKIP_POSE=1
  echo "Reusing pose checkpoint from ${REUSE_POSE_RUN}: ${POSE_CKPT}"
fi

run_train() {
  local stage_name="$1"
  local exp_name="$2"
  local output_dir="$3"
  shift 3
  echo "========== ${stage_name} =========="
  echo "exp_name: ${exp_name}"
  echo "output_dir: ${output_dir}"
  "${PYTHON_BIN}" tools/train.py \
    exp=gvhmr/egobody_egoexo_stage1 \
    exp_name=${exp_name} \
    output_dir=${output_dir} \
    "$@" \
    "${COMMON_OVERRIDES[@]}" \
    "${USER_OVERRIDES[@]}"
}

echo "Repo root: ${REPO_ROOT}"
echo "Data root: ${DATA_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run suffix: ${RUN_SUFFIX}"
echo "Run output: ${RUN_OUTPUT_DIR}"
"${PYTHON_BIN}" - <<'PYIMPORT'
import hmr4d
print(f"hmr4d import: {hmr4d.__file__}")
assert hmr4d.__file__.startswith("/public/home/wenxin/GVHMR_ifcam/"), hmr4d.__file__
PYIMPORT

if [[ "${SKIP_POSE}" != "1" ]]; then
  POSE_CKPT_OVERRIDE=()
  if [[ -n "${POSE_INIT_CKPT}" ]]; then
    POSE_CKPT_OVERRIDE=(ckpt_path=${POSE_INIT_CKPT})
    echo "Pose init checkpoint: ${POSE_INIT_CKPT}"
  fi
  run_train "Stage 1/3: pose warmup" "${POSE_EXP_NAME}" "${POSE_OUTPUT_DIR}" \
    "${POSE_CKPT_OVERRIDE[@]}" \
    "${POSE_OVERRIDES[@]}"
  POSE_CKPT=$(latest_ckpt "${POSE_OUTPUT_DIR}/checkpoints")
fi
if [[ -z "${POSE_CKPT}" ]]; then
  echo "ERROR: no pose checkpoint found. Set POSE_CKPT=... or run pose stage." >&2
  exit 1
fi
echo "Using pose checkpoint: ${POSE_CKPT}"

if [[ "${SKIP_LOCO}" != "1" ]]; then
  run_train "Stage 2/3: locomotion/contact" "${LOCO_EXP_NAME}" "${LOCO_OUTPUT_DIR}" \
    ckpt_path="${POSE_CKPT}" \
    "${LOCO_OVERRIDES[@]}"
  LOCO_CKPT=$(latest_ckpt "${LOCO_OUTPUT_DIR}/checkpoints")
fi
if [[ -z "${LOCO_CKPT}" ]]; then
  echo "ERROR: no locomotion checkpoint found. Set LOCO_CKPT=... or run locomotion stage." >&2
  exit 1
fi
echo "Using locomotion checkpoint: ${LOCO_CKPT}"

if [[ "${SKIP_FINETUNE}" != "1" ]]; then
  run_train "Stage 3/3: low-lr finetune" "${FT_EXP_NAME}" "${FT_OUTPUT_DIR}" \
    ckpt_path="${LOCO_CKPT}" \
    "${FT_OVERRIDES[@]}"
  FT_CKPT=$(latest_ckpt "${FT_OUTPUT_DIR}/checkpoints")
  echo "Latest finetune checkpoint: ${FT_CKPT:-none yet}"
else
  echo "Skipped finetune. Latest locomotion checkpoint: ${LOCO_CKPT}"
fi
