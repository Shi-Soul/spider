from __future__ import annotations

# ruff: noqa: D101,D102
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spider.tasks.g1_wbc import bench_data_benchmark as bench


class BenchDataBenchmarkTest(unittest.TestCase):
    def test_parse_benchmark_yaml_preserves_comment_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.yaml"
            path.write_text(
                "\n".join(
                    [
                        "# locomotion_walk",
                        "- /tmp/walk.npz",
                        "",
                        "# jump_hop",
                        "- relative/jump.npz",
                    ]
                )
                + "\n"
            )

            sources = bench.parse_benchmark_yaml(path)

        self.assertEqual(
            [(source.category, str(source.path), source.line_no) for source in sources],
            [
                ("locomotion_walk", "/tmp/walk.npz", 2),
                ("jump_hop", "relative/jump.npz", 5),
            ],
        )

    def test_parse_benchmark_yaml_rejects_motion_before_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.yaml"
            path.write_text("- /tmp/walk.npz\n")

            with self.assertRaisesRegex(ValueError, "before any category"):
                bench.parse_benchmark_yaml(path)

    def test_slugify_removes_spaces_case_slashes_and_punctuation(self) -> None:
        self.assertEqual(
            bench.slugify("Loop/Backward Walk 001__A017!"),
            "loop_backward_walk_001_a017",
        )

    def test_motion_id_uses_category_source_group_and_stem(self) -> None:
        self.assertEqual(
            bench.build_motion_id(
                "locomotion_walk",
                "sonic_filtered/220705",
                "Loop_Backward_Walk_001__A017",
            ),
            "locomotion_walk__sonic_filtered_220705__loop_backward_walk_001_a017",
        )

    def test_motion_id_collision_appends_sha1_suffix(self) -> None:
        motion_a = _motion(
            motion_id="same__base__motion",
            path=Path("/tmp/a/motion.npz"),
        )
        motion_b = _motion(
            motion_id="same__base__motion",
            path=Path("/tmp/b/motion.npz"),
        )

        resolved = bench.resolve_motion_ids([motion_a, motion_b])

        self.assertEqual(len({motion.id for motion in resolved}), 2)
        self.assertRegex(resolved[0].id, r"^same__base__motion__[0-9a-f]{8}$")
        self.assertRegex(resolved[1].id, r"^same__base__motion__[0-9a-f]{8}$")

    def test_load_motion_metadata_reads_frames_fps_duration_and_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            motion_path = root / "bench_data" / "sonic_filtered" / "220705" / "Walk.npz"
            _write_motion_npz(motion_path, frames=11, fps=50)
            source = bench.BenchmarkSource(
                category="locomotion_walk",
                path=motion_path,
                line_no=1,
            )

            motion = bench.load_motion_metadata(source, workspace_root=root)

        self.assertEqual(motion.id, "locomotion_walk__sonic_filtered_220705__walk")
        self.assertEqual(motion.source_group, "sonic_filtered/220705")
        self.assertEqual(motion.repo_relative_path, "bench_data/sonic_filtered/220705/Walk.npz")
        self.assertEqual(motion.fps, 50.0)
        self.assertEqual(motion.frames, 11)
        self.assertEqual(motion.duration_sec, 0.2)
        self.assertEqual(motion.max_steps, 10)

    def test_max_steps_is_min_800_frames_minus_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            motion_path = root / "bench_data" / "sonic_filtered" / "220705" / "Long.npz"
            _write_motion_npz(motion_path, frames=1200, fps=50)
            source = bench.BenchmarkSource("locomotion_walk", motion_path, 1)

            motion = bench.load_motion_metadata(source, workspace_root=root)

        self.assertEqual(motion.max_steps, 800)

    def test_write_input_manifest_uses_repo_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "package" / "input_motions" / "bench_data.yaml"
            motion = _motion(
                motion_id="locomotion_walk__sonic_filtered_220705__walk",
                path=root / "bench_data" / "sonic_filtered" / "220705" / "Walk.npz",
                repo_relative_path="bench_data/sonic_filtered/220705/Walk.npz",
            )

            bench.write_input_manifest([motion], manifest_path, workspace_root=root)

            text = manifest_path.read_text()

        self.assertIn("path_mode: repo_relative", text)
        self.assertIn("path: bench_data/sonic_filtered/220705/Walk.npz", text)
        self.assertNotIn(f"path: {root}", text)

    def test_build_eval_command_no_mpc_has_no_mpc_flags_and_no_mpc_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            motion = _motion(path=root / "bench_data" / "motion.npz")

            command = bench.build_eval_command(motion, "no_mpc", config)

        self.assertEqual(command.output_dir, config.package_dir / "outputs" / "no_mpc" / motion.id)
        self.assertEqual(command.log_path, config.package_dir / "logs" / "eval" / motion.id / "no_mpc.log")
        self.assertIn("--num-envs", command.argv)
        self.assertIn("--save-rollout", command.argv)
        self.assertNotIn("--mpc-samples", command.argv)
        self.assertEqual(command.env["PYTHONPATH"], str(config.spider_root))

    def test_build_eval_command_mpc_has_fixed_phase1_flags_and_spider_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            motion = _motion(path=root / "bench_data" / "motion.npz")

            command = bench.build_eval_command(motion, "g1_wbc_joint", config)

        self.assertEqual(
            command.output_dir,
            config.package_dir / "outputs" / "spider" / motion.id / "g1_wbc_joint",
        )
        self.assertIn("--mpc-preset", command.argv)
        self.assertIn("aggressive", command.argv)
        self.assertIn("--mpc-sampling-mode", command.argv)
        self.assertIn("knot", command.argv)
        self.assertIn("--mpc-knot-count", command.argv)
        self.assertIn("8", command.argv)
        self.assertIn("--mpc-samples", command.argv)
        self.assertIn("8192", command.argv)
        self.assertIn("--mpc-smooth-passes", command.argv)
        self.assertIn("0", command.argv)
        self.assertIn("--mpc-command-reg-weight", command.argv)
        self.assertIn("0.0", command.argv)
        self.assertIn("--mpc-command-smooth-weight", command.argv)
        self.assertIn("--mpc-guided-root-pos-gain", command.argv)
        self.assertIn("--mpc-guided-joint-clip", command.argv)
        self.assertIn("0.35", command.argv)
        self.assertIn("--mpc-guided-candidate", command.argv)
        self.assertIn("--mpc-acceptance-gate", command.argv)
        self.assertNotIn("--num-envs", command.argv)

    def test_invalid_methods_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "Unsupported method"):
                _config(root, methods=("no_mpc", "not_a_method"))

    def test_partial_methods_are_rejected_for_phase1_four_panel_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "requires methods"):
                _config(root, methods=("no_mpc", "g1_wbc_joint"))

    def test_smoke_mode_builds_12_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            motions = [
                _motion(motion_id=motion_id, path=root / f"{idx}.npz")
                for idx, motion_id in enumerate(bench.DEFAULT_SMOKE_MOTION_IDS)
            ]

            commands = bench.build_eval_commands(motions, config, mode="smoke")

        self.assertEqual(len(commands), 12)
        self.assertEqual([command.motion.id for command in commands[::4]], bench.DEFAULT_SMOKE_MOTION_IDS)

    def test_full_mode_builds_112_commands_for_28_motions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            motions = [
                _motion(motion_id=f"motion_{idx:02d}", path=root / f"{idx}.npz")
                for idx in range(28)
            ]

            commands = bench.build_eval_commands(motions, config, mode="full")

        self.assertEqual(len(commands), 112)

    def test_dry_run_does_not_call_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_yaml, package_dir = _single_motion_benchmark(root)
            calls: list[bench.EvalCommand] = []

            exit_code = bench.run_benchmark(
                [
                    "--workspace-root",
                    str(root),
                    "--benchmark-yaml",
                    str(benchmark_yaml),
                    "--package-dir",
                    str(package_dir),
                    "--dry-run",
                    "--mode",
                    "full",
                ],
                run_eval=lambda command, timeout_sec: calls.append(command),
                run_render=lambda command, timeout_sec: calls.append(command),  # type: ignore[arg-type]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, [])
            self.assertTrue((package_dir / "input_motions" / "bench_data.yaml").exists())
            summary = (package_dir / "benchmark" / "benchmark_summary.md").read_text()
            self.assertIn("- Unknown runs: 4", summary)
            self.assertNotIn("- no_mpc: 1", summary)
            eval_script = (package_dir / "metadata" / "eval_commands.sh").read_text()
            render_script = (package_dir / "metadata" / "render_commands.sh").read_text()
            self.assertIn(f"cd {root / 'spider'}", eval_script)
            self.assertIn("export PYTHONPATH=", eval_script)
            self.assertIn(f"cd {root / 'spider'}", render_script)
            self.assertIn("export PYTHONPATH=", render_script)

    def test_build_render_command_uses_saved_rollouts_in_panel_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _config(root)
            motion = _motion(path=root / "bench_data" / "motion.npz", max_steps=218)

            command = bench.build_render_command(motion, config)

        rollout_args = [
            value for idx, value in enumerate(command.argv) if command.argv[idx - 1] == "--saved-rollout"
        ]
        self.assertEqual(
            [arg.split(":", 1)[0] for arg in rollout_args],
            ["no_mpc", "g1_wbc_joint_global", "g1_wbc_joint", "g1_wbc_ee"],
        )
        self.assertIn("--panel-layout", command.argv)
        self.assertIn("2x2", command.argv)
        self.assertIn("--device", command.argv)
        self.assertIn("cuda:0", command.argv)
        self.assertIn("--max-steps", command.argv)
        self.assertIn("218", command.argv)

    def test_timeout_is_recorded_as_failed_run_and_benchmark_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_yaml, package_dir = _single_motion_benchmark(root)
            calls = 0

            def timeout_eval(
                command: bench.EvalCommand,
                timeout_sec: int | None,
            ) -> subprocess.CompletedProcess[str]:
                nonlocal calls
                calls += 1
                raise subprocess.TimeoutExpired(command.argv, timeout_sec)

            exit_code = bench.run_benchmark(
                [
                    "--workspace-root",
                    str(root),
                    "--benchmark-yaml",
                    str(benchmark_yaml),
                    "--package-dir",
                    str(package_dir),
                    "--mode",
                    "full",
                    "--timeout-sec",
                    "1",
                ],
                run_eval=timeout_eval,
                run_render=lambda command, timeout_sec: subprocess.CompletedProcess(
                    command.argv,
                    0,
                ),
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(calls, 4)

    def test_classify_run_metrics_marks_missing_required_metrics_unknown(self) -> None:
        status = bench.classify_run_metrics({"score": -1.0})

        self.assertEqual(status.track_status, "unknown")
        self.assertIn("missing_required_metrics", status.failure_labels)

    def test_classify_run_metrics_marks_threshold_edges_failed_and_near_edges_borderline(self) -> None:
        failed = bench.classify_run_metrics(
            _metrics(root_pos_error_mean=0.25, contact_mismatch_rate=0.10)
        )
        borderline = bench.classify_run_metrics(
            _metrics(root_pos_error_mean=0.20, contact_mismatch_rate=0.10)
        )

        self.assertEqual(failed.track_status, "failed")
        self.assertIn("root_translation_drift", failed.failure_labels)
        self.assertEqual(borderline.track_status, "borderline")

    def test_compare_baseline_and_mpc_marks_regression_and_stability_regression(self) -> None:
        baseline = _metrics(score=-1.0, control_delta_mean=0.10)
        mpc = _metrics(score=-2.0, control_delta_mean=0.20)

        comparison = bench.compare_baseline_and_mpc(baseline, mpc)

        self.assertEqual(comparison.score_status, "regressed")
        self.assertIn("baseline_regression", comparison.failure_labels)
        self.assertIn("stability_regression", comparison.failure_labels)

    def test_checksum_writer_is_stable_and_excludes_manifest_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = Path(tmpdir)
            (package_dir / "metadata").mkdir()
            (package_dir / "b.txt").write_text("b")
            (package_dir / "a.txt").write_text("a")

            bench.write_package_manifests(package_dir)

            manifest_csv = (package_dir / "metadata" / "manifest.csv").read_text()
            manifest_sha = (package_dir / "metadata" / "manifest.sha256").read_text()

        self.assertLess(manifest_csv.index("a.txt"), manifest_csv.index("b.txt"))
        self.assertIn("metadata/manifest.sha256", manifest_csv)
        self.assertNotIn("metadata/manifest.csv", manifest_csv)
        self.assertNotIn("metadata/manifest.csv", manifest_sha)
        self.assertNotIn("metadata/manifest.sha256", manifest_sha)


def _write_motion_npz(path: Path, *, frames: int, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        fps=np.array(fps),
        joint_pos=np.zeros((frames, 29), dtype=np.float32),
    )


def _motion(
    *,
    motion_id: str = "locomotion_walk__sonic_filtered_220705__walk",
    path: Path = Path("/tmp/bench_data/sonic_filtered/220705/Walk.npz"),
    repo_relative_path: str = "bench_data/sonic_filtered/220705/Walk.npz",
    max_steps: int = 10,
) -> bench.BenchMotion:
    return bench.BenchMotion(
        id=motion_id,
        category="locomotion_walk",
        source_group="sonic_filtered/220705",
        path=path,
        repo_relative_path=repo_relative_path,
        motion_type="isaaclab",
        fps=50.0,
        frames=max_steps + 1,
        duration_sec=max_steps / 50.0,
        max_steps=max_steps,
    )


def _config(
    root: Path,
    *,
    methods: tuple[str, ...] = bench.DEFAULT_METHODS,
) -> bench.RunnerConfig:
    return bench.RunnerConfig(
        workspace_root=root,
        package_dir=root / "package",
        python_executable="/usr/bin/python",
        device="cuda:0",
        checkpoint=root / "wxy" / "0608_ckpt_bc" / "model_8000.pt",
        reward_weights=root
        / "g1_wbc_testbed_motion_package_20260617"
        / "metadata"
        / "g1_wbc_reward_weights_method_specific_v14_20260612.json",
        methods=methods,
    )


def _single_motion_benchmark(root: Path) -> tuple[Path, Path]:
    motion_path = root / "bench_data" / "sonic_filtered" / "220705" / "Walk.npz"
    _write_motion_npz(motion_path, frames=11, fps=50)
    benchmark_yaml = root / "bench_data" / "benchmark.yaml"
    benchmark_yaml.write_text(f"# locomotion_walk\n- {motion_path}\n")
    return benchmark_yaml, root / "package"


def _metrics(**overrides: float | bool) -> dict[str, float | bool]:
    values: dict[str, float | bool] = {
        "score": -1.0,
        "success": True,
        "root_pos_error_mean": 0.10,
        "root_rot_error_mean": 0.10,
        "ee_global_pos_error_mean": 0.10,
        "ee_local_pos_error_mean": 0.10,
        "contact_mismatch_rate": 0.10,
        "contact_false_positive_rate": 0.02,
        "contact_false_negative_rate": 0.03,
        "contact_switch_rate": 0.10,
        "control_delta_mean": 0.10,
        "joint_acc_mean": 0.10,
        "joint_jerk_mean": 0.10,
    }
    values.update(overrides)
    return values


if __name__ == "__main__":
    unittest.main()
