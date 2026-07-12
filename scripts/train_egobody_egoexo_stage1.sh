#!/usr/bin/env bash
set -euo pipefail

# Stage-1 EgoBody ego/exo training.
# Default behavior:
#   - load GVHMR release checkpoint
#   - freeze original exo backbone/head
#   - train ego image/head/hand condition embedders and ego SMPL head
#   - use preprocessed EgoBody records under /public/home/wenxin/GVHMR/data

PYTHON_BIN=${PYTHON_BIN:-/public/home/wenxin/miniconda3/envs/gvhmr/bin/python}
DATA_ROOT=${DATA_ROOT:-/public/home/wenxin/GVHMR/data}
DEVICES=${DEVICES:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}

${PYTHON_BIN} tools/train.py \
  exp=gvhmr/egobody_egoexo_stage1 \
  data.dataset_opts.train.egobody_egoexo_train.output_root=${DATA_ROOT} \
  data.dataset_opts.val.egobody_egoexo_val.output_root=${DATA_ROOT} \
  data.loader_opts.train.batch_size=${BATCH_SIZE} \
  data.loader_opts.val.batch_size=1 \
  data.loader_opts.train.num_workers=${NUM_WORKERS} \
  data.loader_opts.val.num_workers=${NUM_WORKERS} \
  pl_trainer.devices=${DEVICES} \
  "$@"
