#!/usr/bin/env bash
set -euo pipefail

# Single-window overfit sanity check for EgoBody ego branch.
# This should drive training losses close to zero and make val/world_mesh_floor
# align ego_pred with ego_gt on the same train sample.

PYTHON_BIN=${PYTHON_BIN:-/public/home/wenxin/miniconda3/envs/gvhmr/bin/python}
DATA_ROOT=${DATA_ROOT:-/public/home/wenxin/GVHMR/data}
DEVICES=${DEVICES:-1}
MAX_STEPS=${MAX_STEPS:-1000}
VAL_EVERY_EPOCHS=${VAL_EVERY_EPOCHS:-100}
VIS_EVERY=${VIS_EVERY:-100}

${PYTHON_BIN} tools/train.py \
  exp=gvhmr/egobody_egoexo_stage1_overfit1 \
  data.dataset_opts.train.egobody_egoexo_train.output_root=${DATA_ROOT} \
  data.dataset_opts.val.egobody_egoexo_val.output_root=${DATA_ROOT} \
  model.vis_every_n_steps=${VIS_EVERY} \
  pl_trainer.max_steps=${MAX_STEPS} \
  pl_trainer.check_val_every_n_epoch=${VAL_EVERY_EPOCHS} \
  pl_trainer.devices=${DEVICES} \
  "$@"
