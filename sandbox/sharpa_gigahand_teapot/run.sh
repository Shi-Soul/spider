#!/bin/bash
# Benchmark p36-tea (gigahand teapot) for xhand (single seed) and sharpa (5 seeds).
# Run on amazon16 (4 GPUs). The dev box GPU is too slow for mjwp_fast.
#
# Usage:
#   bash sandbox/sharpa_gigahand_teapot/run.sh
#
# Pre-reqs (verify before running):
#   - Branch feedback/sharpa-gigahand checked out
#   - example_datasets/raw/gigahand exists
#   - spider/assets/robots/sharpa/ exists (NOT in tree as of 2026-05-31; user
#     must add sharpa MJCF + embodiment mappings before running the sharpa half)
set -e
cd "$(git rev-parse --show-toplevel)"

OUT=sandbox/sharpa_gigahand_teapot
mkdir -p "$OUT/xhand" "$OUT/sharpa"

# 1) Process gigahand raw -> standard NPZ for p36-tea, sequence 0010 -> data_id=10
uv run spider/process_datasets/gigahand.py \
  --participant=p36 --scene=tea --sequence-id=0010 \
  --embodiment-type=right --no-show-viewer

# data_id used by downstream scripts is the integer of the sequence_id (here, 10).
DATA_ID=10
TASK="p36-tea-0010"

# 2) Convex decompose with CoACD (handles -> use decompose.py, NOT decompose_fast.py)
uv run spider/preprocess/decompose.py \
  --task="$TASK" --dataset-name=gigahand --data-id=$DATA_ID --embodiment-type=right

# 3) Detect contacts
uv run spider/preprocess/detect_contact.py \
  --task="$TASK" --dataset-name=gigahand --data-id=$DATA_ID --embodiment-type=right

# ---------- xhand ----------
uv run spider/preprocess/generate_xml.py \
  --task="$TASK" --dataset-name=gigahand --data-id=$DATA_ID --embodiment-type=right \
  --robot-type=xhand
uv run spider/preprocess/ik.py \
  --task="$TASK" --dataset-name=gigahand --data-id=$DATA_ID --embodiment-type=right \
  --robot-type=xhand --open-hand --save-video --no-show-viewer
uv run examples/run_mjwp_fast.py \
  +override=gigahand_fast task="$TASK" data_id=$DATA_ID robot_type=xhand \
  embodiment_type=right seed=0 2>&1 | tee "$OUT/xhand/run_seed0.log"

# ---------- sharpa (5 seeds) ----------
uv run spider/preprocess/generate_xml.py \
  --task="$TASK" --dataset-name=gigahand --data-id=$DATA_ID --embodiment-type=right \
  --robot-type=sharpa
uv run spider/preprocess/ik.py \
  --task="$TASK" --dataset-name=gigahand --data-id=$DATA_ID --embodiment-type=right \
  --robot-type=sharpa --open-hand --save-video --no-show-viewer
for seed in 0 1 2 3 4; do
  mkdir -p "$OUT/sharpa/seed${seed}"
  uv run examples/run_mjwp_fast.py \
    +override=gigahand_fast task="$TASK" data_id=$DATA_ID robot_type=sharpa \
    embodiment_type=right seed=$seed 2>&1 | tee "$OUT/sharpa/seed${seed}/run.log"
done

echo "Done. Artifacts in $OUT (videos may live under recordings/ — see MANIFEST.txt)."
