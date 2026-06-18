# G1 WBC Bench Data Phase 1 Benchmark Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Spider-native phase-1 benchmark runner for G1 WBC on the `bench_data/` IsaacLab-format motion set. The runner must create a reproducible package, evaluate `no_mpc` plus three MPC methods with the fixed BC checkpoint and v14 MPC settings, summarize metrics/failures, and render one four-panel video per motion.

**Architecture:** Keep GPU/MuJoCo execution in existing `evaluate.py` and `visualize_g1_wbc.py`. Add pure benchmark helper code for manifest parsing, command construction, package metadata, and metric summaries so CPU tests can validate behavior without importing simulation dependencies.

**Tech Stack:** Python stdlib, NumPy for `.npz` metadata inspection, existing Spider G1 WBC CLIs, `unittest` tests. Do not add dependencies or update `uv.lock`.

---

## Constraints

- Work in `spider` on branch `dev-wjs`.
- Preserve existing dirty user state: modified `uv.lock` and untracked `AGENTS.md`.
- Fixed checkpoint: `/home/wujs/Projects/BrainStorm/Learning/RL/model-based/wxy/0608_ckpt_bc/model_8000.pt`.
- Fixed reward weights: `/home/wujs/Projects/BrainStorm/Learning/RL/model-based/g1_wbc_testbed_motion_package_20260617/metadata/g1_wbc_reward_weights_method_specific_v14_20260612.json`.
- Methods: `no_mpc`, `g1_wbc_joint_global`, `g1_wbc_joint`, `g1_wbc_ee`.
- Motion type: `isaaclab`.
- Per-motion steps: `min(800, frames - 1)`.
- Keep `bench_data/` files in place; package `input_motions/bench_data.yaml` references repo-relative paths.
- No H.264 transcode; use existing `visualize_g1_wbc.py` `mp4v` output.

## Target Files

- [x] Add `spider/spider/tasks/g1_wbc/bench_data_benchmark.py`.
- [x] Add `spider/scripts/run_g1_wbc_bench_data_phase1.py` as a thin wrapper.
- [x] Add `spider/tests/tasks/g1_wbc/test_bench_data_benchmark.py`.
- [x] Add this plan file under `spider/docs/superpowers/plans/`.

## Implementation Tasks

- [x] Manifest parsing and motion metadata
  - Parse `bench_data/benchmark.yaml`, preserving category comments.
  - Reject motion entries before the first category.
  - Build stable IDs as `{category_slug}__{source_group_slug}__{motion_stem_slug}` and add a short hash suffix on collisions.
  - Read `.npz` metadata: `fps`, frame count, duration, and validate `frames >= 2`.
  - Validate phase-1 `fps == 50`.
  - Write `input_motions/bench_data.yaml` with repo-relative paths.

- [x] Command generation and dry run
  - Build eval command argv for existing `python -m spider.tasks.g1_wbc.evaluate`.
  - Use `outputs/no_mpc/<motion_id>` for baseline and `outputs/spider/<motion_id>/<method>` for MPC.
  - Use `logs/eval/<motion_id>/<method>.log`.
  - Add `--save-rollout` for all runs and `--num-envs 1` for `no_mpc`.
  - Add fixed v14 MPC flags only to MPC methods.
  - Support `--mode smoke|full`, `--manifest-only`, `--dry-run`, `--limit`, `--skip-existing`, `--timeout-sec`, and `--device`.

- [x] Package metadata and videos
  - Build one render command per motion using `visualize_g1_wbc.py --method saved`.
  - Keep saved rollout order: `no_mpc`, `g1_wbc_joint_global`, `g1_wbc_joint`, `g1_wbc_ee`.
  - Use `--panel-layout 2x2`, `--camera-mode ref-follow`, `--show-root-error`, `--width 960`, `--height 540`, `--fps 50`.
  - Write `metadata/render_commands.sh`, `metadata/bench_data_sources.csv`, `metadata/benchmark_config.json`, `metadata/package_summary.json`, `metadata/manifest.csv`, `metadata/manifest.sha256`, and `videos/manifest.csv`.

- [x] Metric summaries
  - Read each `metrics.json` emitted by `evaluate.py`.
  - Alias `metrics.success` to `success_current`.
  - Generate benchmark CSVs and `benchmark/benchmark_summary.md`.
  - Track failure labels: `missing_required_metrics`, `root_translation_drift`, `root_orientation_drift`, `ee_global_tracking_failure`, `ee_local_tracking_failure`, `contact_schedule_failure`, `baseline_regression`, and `stability_regression`.
  - Track statuses: `tracked`, `borderline`, `failed`, `unknown`.

- [x] Verification
  - [x] First add tests and observe expected failures.
  - [x] Implement minimal code until tests pass.
  - [x] Run `PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py'` from `spider`.
  - [x] Run a dry run: `PYTHONPATH=. python scripts/run_g1_wbc_bench_data_phase1.py --benchmark-yaml ../bench_data/benchmark.yaml --package-dir /tmp/g1_wbc_bench_dryrun --date 20260618 --dry-run --limit 2`.
  - [x] Run `uv run --frozen --extra dev ruff check scripts/run_g1_wbc_bench_data_phase1.py spider/tasks/g1_wbc/bench_data_benchmark.py tests/tasks/g1_wbc` if local dependencies allow.

## Acceptance Criteria

- `--dry-run --limit 2` creates a package skeleton with full input manifest and no GPU rollout.
- Smoke mode plans 3 motions x 4 methods; full mode plans all `bench_data` motions x 4 methods.
- Render commands are generated only after rollout paths can be referenced and are reproducible from `metadata/render_commands.sh`.
- Summaries tolerate missing `metrics.json` files and mark those runs as `unknown` rather than crashing.
- Unit tests cover manifest parsing, ID generation, command construction, package metadata, and summary classification.
