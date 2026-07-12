#!/usr/bin/env bash
set -euo pipefail

# Two-stage EgoBody training:
#   Stage 1: ego-only on preprocessed EgoBody ego/exo data, train ego branch with last N shared blocks unfrozen.
#   Stage 2: ego+exo joint training on the same preprocessed EgoBody paired data, starting from Stage 1 ckpt.
#
# Useful overrides:
#   DATA_ROOT=/public/home/wenxin/GVHMR/data
#   DEVICES=1
#   STAGE1_EPOCHS=80 STAGE2_EPOCHS=80
#   UNFREEZE_BLOCKS=2 STAGE1_BACKBONE_LR_SCALE=0.02 STAGE2_BACKBONE_LR_SCALE=0.005
#   STAGE1_INIT_CKPT=/path/to/stage1.ckpt   # optional; also accepts ckpt_path=... for stage 1 only

PYTHON_BIN=${PYTHON_BIN:-/public/home/wenxin/miniconda3/envs/gvhmr/bin/python}
DATA_ROOT=${DATA_ROOT:-/public/home/wenxin/GVHMR/data}
DATA_NAME=${DATA_NAME:-egobody_egoexo_v2}
DEVICES=${DEVICES:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}
STAGE1_EPOCHS=${STAGE1_EPOCHS:-120}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-120}
UNFREEZE_BLOCKS=${UNFREEZE_BLOCKS:-2}
STAGE1_BACKBONE_LR_SCALE=${STAGE1_BACKBONE_LR_SCALE:-0.01}
STAGE2_BACKBONE_LR_SCALE=${STAGE2_BACKBONE_LR_SCALE:-0.003}
VIS_EVERY=${VIS_EVERY:-500}
VAL_EVERY_EPOCHS=${VAL_EVERY_EPOCHS:-5}
VAL_VIS_EVERY=${VAL_VIS_EVERY:-}
STAGE1_INIT_CKPT=${STAGE1_INIT_CKPT:-}

USER_OVERRIDES=()
for arg in "$@"; do
  if [[ "${arg}" == ckpt_path=* ]]; then
    STAGE1_INIT_CKPT="${arg#ckpt_path=}"
  else
    USER_OVERRIDES+=("${arg}")
  fi
done

COMMON_OVERRIDES=(
  data_name=${DATA_NAME}
  data.dataset_opts.train.egobody_egoexo_train.output_root=${DATA_ROOT}
  data.dataset_opts.val.egobody_egoexo_val.output_root=${DATA_ROOT}
  data.loader_opts.train.batch_size=${BATCH_SIZE}
  data.loader_opts.val.batch_size=1
  data.loader_opts.train.num_workers=${NUM_WORKERS}
  data.loader_opts.val.num_workers=${NUM_WORKERS}
  pl_trainer.devices=${DEVICES}
  model.unfreeze_last_n_blocks=${UNFREEZE_BLOCKS}
  model.vis_every_n_steps=${VIS_EVERY}
  pl_trainer.check_val_every_n_epoch=${VAL_EVERY_EPOCHS}
)
if [[ -n "${VAL_VIS_EVERY}" ]]; then
  COMMON_OVERRIDES+=(model.val_vis_every_n_batches=${VAL_VIS_EVERY})
fi

STAGE1_CKPT_OVERRIDE=()
if [[ -n "${STAGE1_INIT_CKPT}" ]]; then
  STAGE1_CKPT_OVERRIDE=(ckpt_path=${STAGE1_INIT_CKPT})
  echo "Stage 1 init checkpoint: ${STAGE1_INIT_CKPT}"
fi

echo "========== Stage 1: Ego input, ego-supervised =========="
${PYTHON_BIN} tools/train.py   exp=gvhmr/egobody_egoexo_stage1   exp_name=egobody_egoexo_stage1_ego_only_long   pipeline.args.branch_mode=both   pipeline.args.input_role=ego   pipeline.args.train_input_role=ego   pipeline.args.supervise_role=ego   pipeline.args.enable_frozen_ego_image_exo=false   model.freeze_backbone=true   model.freeze_exo_head=true   model.freeze_ego_head=false   model.backbone_lr_scale=${STAGE1_BACKBONE_LR_SCALE}   pl_trainer.max_epochs=${STAGE1_EPOCHS}   "${STAGE1_CKPT_OVERRIDE[@]}"   "${COMMON_OVERRIDES[@]}"   "${USER_OVERRIDES[@]}"

STAGE1_CKPT_DIR="outputs/${DATA_NAME}/egobody_egoexo_stage1_ego_only_long/checkpoints"
STAGE1_CKPT=$(find "${STAGE1_CKPT_DIR}" -maxdepth 1 -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
if [[ -z "${STAGE1_CKPT}" ]]; then
  echo "ERROR: no Stage 1 checkpoint found in ${STAGE1_CKPT_DIR}" >&2
  echo "Hint: SimpleCkptSaver saves every 10 epochs by default; set STAGE1_EPOCHS>=10." >&2
  exit 1
fi

echo "Using Stage 1 checkpoint: ${STAGE1_CKPT}"
echo "========== Stage 2: Ego+Exo joint on EgoBody paired data =========="
${PYTHON_BIN} tools/train.py   exp=gvhmr/egobody_egoexo_stage2_both   exp_name=egobody_egoexo_stage2_both_long   ckpt_path="${STAGE1_CKPT}"   pipeline.args.branch_mode=both   model.freeze_backbone=true   model.freeze_exo_head=false   model.freeze_ego_head=false   model.backbone_lr_scale=${STAGE2_BACKBONE_LR_SCALE}   pl_trainer.max_epochs=${STAGE2_EPOCHS}   "${COMMON_OVERRIDES[@]}"   "${USER_OVERRIDES[@]}"
