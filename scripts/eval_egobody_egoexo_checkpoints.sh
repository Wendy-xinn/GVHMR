#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/public/home/wenxin/miniconda3/envs/gvhmr/bin/python}
DATA_ROOT=${DATA_ROOT:-/public/home/wenxin/GVHMR/data}
OUT_DIR=${OUT_DIR:-outputs/egobody_egoexo_v1/eval}
NUM_WORKERS=${NUM_WORKERS:-4}
MOTION_FRAMES=${MOTION_FRAMES:-128}

EGO_CKPT=${EGO_CKPT:-outputs/egobody_egoexo_v1/egobody_egoexo_stage1_ego_only_long/checkpoints/e019-s000980.ckpt}
BOTH_CKPT=${BOTH_CKPT:-outputs/egobody_egoexo_v1/egobody_egoexo_stage2_both_continue/checkpoints/e099-s004900.ckpt}

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" tools/egobody/eval_egoexo.py \
  --exp gvhmr/egobody_egoexo_stage1 \
  --ckpt "$EGO_CKPT" \
  --branch-mode ego \
  --split test \
  --data-root "$DATA_ROOT" \
  --motion-frames "$MOTION_FRAMES" \
  --num-workers "$NUM_WORKERS" \
  --out "$OUT_DIR/ego_only_test.json"

"$PYTHON_BIN" tools/egobody/eval_egoexo.py \
  --exp gvhmr/egobody_egoexo_stage2_both \
  --ckpt "$BOTH_CKPT" \
  --branch-mode both \
  --split test \
  --data-root "$DATA_ROOT" \
  --motion-frames "$MOTION_FRAMES" \
  --num-workers "$NUM_WORKERS" \
  --out "$OUT_DIR/both_test.json"
