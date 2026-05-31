#!/usr/bin/env bash
# Run the full arcticv2 pipeline (process -> decompose -> generate_xml ->
# ik_fast -> mjwp_fast) on the 6 dexmachina-pinned tasks. Saves viser
# exports for IK and MJWP-Fast for later inspection.
#
# Heavy mjwp_fast runs - execute on amazon16 (4 GPUs), not the dev box.
#
# Usage:
#   bash scripts/run_arcticv2_pipeline.sh
set -e

cd "$(dirname "$0")/.."

# Headless OpenGL for mujoco offscreen rendering (video export).
export MUJOCO_GL=${MUJOCO_GL:-egl}

PYTHON=.venv/bin/python
ROBOT=${ROBOT:-xhand}
EMB=bimanual
ARCTIC_ROOT=${ARCTIC_ROOT:-~/arctic}
MANO_ROOT=${MANO_ROOT:-$ARCTIC_ROOT/unpack/body_models}

# 6 dexmachina-pinned objects (subject s01, "use" sub-task, take 01).
TASKS="s01-box_use_01 s01-ketchup_use_01 s01-laptop_use_01 s01-mixer_use_01 s01-notebook_use_01 s01-waffleiron_use_01"

# Per-robot IK weights. Sharpa benefits from a higher wrist_pos_cost for tighter
# wrist tracking; xhand uses the default.
case "$ROBOT" in
    sharpa)
        IK_WRIST_POS_COST=2.0
        IK_WRIST_ORI_COST=5.0
        ;;
    *)
        IK_WRIST_POS_COST=0.3
        IK_WRIST_ORI_COST=3.0
        ;;
esac
# Per-task wall-clock cap for mjwp_fast.
MJWP_TIMEOUT=300

for TASK in $TASKS; do
    echo "================================================================="
    echo "Task: $TASK"
    echo "================================================================="

    SUBJECT=${TASK%%-*}
    SEQUENCE=${TASK#*-}

    echo ">>> [1/4] process_datasets/arcticv2.py"
    $PYTHON spider/process_datasets/arcticv2.py \
        --arctic-root=$ARCTIC_ROOT --subject=$SUBJECT --sequence=$SEQUENCE \
        --data-id=0 --embodiment-type=$EMB --robot-type=$ROBOT \
        --no-show-viewer --mano-assets-root=$MANO_ROOT \
        2>&1 | grep -E '(clipped frame range|Saved qpos|active_hand|frame-0)' | head -3

    echo ">>> [2/4] preprocess/decompose_fast.py (with floor support + stability check)"
    $PYTHON spider/preprocess/decompose_fast.py \
        --dataset-name=arcticv2 --task=$TASK \
        --embodiment-type=$EMB --data-id=0 --add-floor --check-stability \
        2>&1 | grep -E '(Stability|Updated|Failed)' | head -3

    echo ">>> [3/4] preprocess/generate_xml.py"
    $PYTHON spider/preprocess/generate_xml.py \
        --dataset-name=arcticv2 --task=$TASK \
        --embodiment-type=$EMB --data-id=0 --robot-type=$ROBOT 2>&1 \
        | grep -E '(Saved model|Failed)' || true

    echo ">>> [4a/4] preprocess/ik_fast.py (save .viser + .mp4)"
    $PYTHON spider/preprocess/ik_fast.py \
        --dataset-name=arcticv2 --task=$TASK --robot-type=$ROBOT \
        --embodiment-type=$EMB --data-id=0 \
        --wrist-pos-cost=$IK_WRIST_POS_COST \
        --wrist-ori-cost=$IK_WRIST_ORI_COST \
        --no-show-viewer --save-video --save-viser \
        --mano-assets-root=$MANO_ROOT \
        2>&1 | grep -E '(Saved viser|Saved video|Saved /local|Failed)' | head -6

    echo ">>> [4b/4] examples/run_mjwp_fast.py (save .viser + .mp4, timeout=${MJWP_TIMEOUT}s)"
    timeout --preserve-status --signal=KILL $MJWP_TIMEOUT \
        $PYTHON examples/run_mjwp_fast.py +override=arcticv2_fast \
        task=$TASK robot_type=$ROBOT embodiment_type=$EMB data_id=0 \
        viewer=viser save_viser=true +wait_on_finish=false \
        save_video=true show_viewer=true 2>&1 \
        | grep -E '(Saved viser|Saved video|Final object tracking|Saved info|Attempt)' || true
done

echo "================================================================="
echo "All 6 tasks complete. Viser exports saved:"
find example_datasets/processed/arcticv2 -name '*.viser' 2>/dev/null | sort
