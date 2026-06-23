from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from spider.tasks.g1_wbc import local_first_stage as stage

RUNNER_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "run_g1_wbc_local_first_stage.py"
)


class LocalFirstTransferRunnerTest(unittest.TestCase):
    def test_selected_candidate_names_supports_transfer_and_recovery_sets(self) -> None:
        runner = _load_runner()

        self.assertEqual(
            runner.selected_candidate_names(None, candidate_set="transfer"),
            stage.TRANSFER_CANDIDATE_NAMES,
        )
        self.assertEqual(
            runner.selected_candidate_names(None, candidate_set="recovery"),
            stage.RECOVERY_CANDIDATE_NAMES,
        )

    def test_selected_candidate_names_supports_pareto_set(self) -> None:
        runner = _load_runner()

        self.assertEqual(
            runner.selected_candidate_names(None, candidate_set="pareto"),
            stage.PARETO_CANDIDATE_NAMES,
        )

    def test_include_contact_is_rejected_for_exact_transfer_and_recovery_sets(
        self,
    ) -> None:
        runner = _load_runner()

        with self.assertRaisesRegex(ValueError, "include-contact"):
            runner.selected_candidate_names(
                None,
                candidate_set="transfer",
                include_contact=True,
            )
        with self.assertRaisesRegex(ValueError, "include-contact"):
            runner.selected_candidate_names(
                None,
                candidate_set="recovery",
                include_contact=True,
            )

    def test_parse_args_infers_transfer_defaults_and_keeps_recovery_defaults(
        self,
    ) -> None:
        runner = _load_runner()

        transfer_args = runner.parse_args(["--candidate-set", "transfer"])
        recovery_args = runner.parse_args(["--candidate-set", "recovery"])
        explicit_motion_args = runner.parse_args(
            ["--candidate-set", "transfer", "--motions", "jump"]
        )

        self.assertEqual(transfer_args.assessment_mode, "transfer")
        self.assertEqual(transfer_args.motions, ["walk", "qixing"])
        self.assertEqual(recovery_args.assessment_mode, "promotion")
        self.assertEqual(recovery_args.motions, ["jump"])
        self.assertEqual(explicit_motion_args.assessment_mode, "transfer")
        self.assertEqual(explicit_motion_args.motions, ["jump"])

    def test_transfer_dry_run_writes_plan_and_planned_summary(self) -> None:
        runner = _load_runner()
        base = {"g1_wbc_joint_global": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            base_path = output_root / "base_reward_weights.json"
            base_path.write_text(json.dumps(base))

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = runner.main(
                    [
                        "--candidate-set",
                        "transfer",
                        "--dry-run",
                        "--output-root",
                        str(output_root),
                        "--base-reward-weights",
                        str(base_path),
                        "--python-executable",
                        "/opt/python",
                        "--device",
                        "cpu",
                    ]
                )

            experiment_plan = json.loads((output_root / "experiment_plan.json").read_text())
            with (output_root / "transfer_summary.csv").open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames

        self.assertEqual(exit_code, 0)
        self.assertEqual(experiment_plan["candidate_set"], "transfer")
        self.assertEqual(experiment_plan["assessment_mode"], "transfer")
        self.assertEqual(experiment_plan["motions"], ["walk", "qixing"])
        self.assertEqual(
            experiment_plan["candidates"],
            list(stage.TRANSFER_CANDIDATE_NAMES),
        )
        self.assertEqual(len(rows), len(stage.TRANSFER_CANDIDATE_NAMES) * 2)
        self.assertEqual({row["motion"] for row in rows}, {"walk", "qixing"})
        self.assertEqual({row["status"] for row in rows}, {"planned"})
        self.assertEqual({row["transfer_class"] for row in rows}, {"invalid"})
        self.assertEqual(fieldnames, runner.TRANSFER_SUMMARY_COLUMNS)

    def test_pareto_dry_run_writes_ladder_plan_and_planned_summary(self) -> None:
        runner = _load_runner()
        base = {"g1_wbc_joint_global": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            base_path = output_root / "base_reward_weights.json"
            base_path.write_text(json.dumps(base))

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = runner.main(
                    [
                        "--candidate-set",
                        "pareto",
                        "--dry-run",
                        "--output-root",
                        str(output_root),
                        "--base-reward-weights",
                        str(base_path),
                        "--python-executable",
                        "/opt/python",
                        "--device",
                        "cpu",
                    ]
                )

            experiment_plan = json.loads((output_root / "experiment_plan.json").read_text())
            planned_commands = (output_root / "planned_commands.sh").read_text()
            with (output_root / "pareto_summary.csv").open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames
            decision_report = (output_root / "pareto_decision.md").read_text()

        self.assertEqual(exit_code, 0)
        self.assertEqual(experiment_plan["candidate_set"], "pareto")
        self.assertEqual(experiment_plan["assessment_mode"], "pareto")
        self.assertEqual(experiment_plan["motions"], ["jump"])
        self.assertEqual(
            experiment_plan["fixed_parameters"]["samples"],
            [64, 96, 128, 192, 256],
        )
        self.assertEqual(experiment_plan["fixed_parameters"]["seeds"], [0, 1, 2])
        self.assertIn("--samples 64 96 128 192 256", planned_commands)
        self.assertIn("--seeds 0 1 2", planned_commands)
        self.assertEqual(len(rows), len(stage.PARETO_SAMPLE_COUNTS))
        self.assertEqual({row["status"] for row in rows}, {"planned"})
        self.assertEqual({row["pareto_class"] for row in rows}, {"invalid"})
        self.assertEqual(fieldnames, runner.PARETO_SUMMARY_COLUMNS)
        self.assertIn("Conclusion: pending", decision_report)
        self.assertIn("rollout execution is still required", decision_report)

    def test_pareto_dry_run_writes_explicit_samples_and_seeds_as_lists(self) -> None:
        runner = _load_runner()
        base = {"g1_wbc_joint_global": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            base_path = output_root / "base_reward_weights.json"
            base_path.write_text(json.dumps(base))

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = runner.main(
                    [
                        "--candidate-set",
                        "pareto",
                        "--dry-run",
                        "--samples",
                        "192",
                        "--seeds",
                        "0",
                        "--output-root",
                        str(output_root),
                        "--base-reward-weights",
                        str(base_path),
                        "--python-executable",
                        "/opt/python",
                        "--device",
                        "cpu",
                    ]
                )

            experiment_plan = json.loads((output_root / "experiment_plan.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(experiment_plan["fixed_parameters"]["samples"], [192])
        self.assertEqual(experiment_plan["fixed_parameters"]["seeds"], [0])

    def test_recovery_dry_run_writes_jump_promotion_plan(self) -> None:
        runner = _load_runner()
        base = {"g1_wbc_joint_global": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            base_path = output_root / "base_reward_weights.json"
            base_path.write_text(json.dumps(base))

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = runner.main(
                    [
                        "--candidate-set",
                        "recovery",
                        "--assessment-mode",
                        "promotion",
                        "--dry-run",
                        "--output-root",
                        str(output_root),
                        "--base-reward-weights",
                        str(base_path),
                        "--python-executable",
                        "/opt/python",
                        "--device",
                        "cpu",
                    ]
                )

            experiment_plan = json.loads((output_root / "experiment_plan.json").read_text())
            planned_commands = (output_root / "planned_commands.sh").read_text()

        self.assertEqual(exit_code, 0)
        self.assertEqual(experiment_plan["candidate_set"], "recovery")
        self.assertEqual(experiment_plan["assessment_mode"], "promotion")
        self.assertEqual(experiment_plan["motions"], ["jump"])
        self.assertEqual(
            experiment_plan["candidates"],
            list(stage.RECOVERY_CANDIDATE_NAMES),
        )
        self.assertEqual(experiment_plan["command_count"], len(stage.RECOVERY_CANDIDATE_NAMES))
        self.assertIn("--motions jump", planned_commands)
        for candidate_name in stage.RECOVERY_CANDIDATE_NAMES:
            self.assertIn(f"# candidate={candidate_name}", planned_commands)

    def test_transfer_summary_sorts_by_transfer_class_before_metrics(self) -> None:
        runner = _load_runner()
        candidate_names = (
            "jg_s128_v14_control",
            "jg_s128_L148_posture",
            "jg_s128_L142_smooth005",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            for candidate_name, score in zip(candidate_names, ("1.0", "2.0", "3.0")):
                _write_csv(
                    output_root / "candidates" / candidate_name / "summary.csv",
                    [
                        _summary_row(
                            "walk",
                            _candidate_metrics("walk", score=score),
                            _accepted_mpc(),
                            duration_sec="100.0",
                        )
                    ],
                )

            with mock.patch.object(
                runner.stage,
                "assess_transfer_candidate",
                side_effect=_fake_transfer_assessment,
            ):
                rows = runner.write_transfer_summary(
                    output_root=output_root,
                    candidate_names=candidate_names,
                    motion_names=("walk",),
                )
            with (output_root / "transfer_summary.csv").open() as f:
                transfer_rows = list(csv.DictReader(f))

        self.assertEqual(
            [row["candidate"] for row in rows],
            [
                "jg_s128_v14_control",
                "jg_s128_L142_smooth005",
                "jg_s128_L148_posture",
            ],
        )
        self.assertEqual([row["rank"] for row in rows], ["1", "2", "3"])
        self.assertEqual(
            [row["transfer_class"] for row in rows],
            ["transfer_pass", "recovery_candidate", "invalid"],
        )
        self.assertEqual({row["status"] for row in rows}, {"completed"})
        self.assertEqual(transfer_rows, rows)

    def test_pareto_summary_aggregates_by_motion_and_samples(self) -> None:
        runner = _load_runner()
        baseline = stage.BEST_S128_BASELINES["jump"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_csv(
                output_root / "candidates" / "jg_s128_v14_control" / "summary.csv",
                [
                    _summary_row(
                        "jump",
                        _candidate_metrics(
                            "jump",
                            success="True",
                            score=str(baseline["score"] + 0.1),
                            root_pos_error_mean=str(
                                baseline["root_pos_error_mean"] * 0.9
                            ),
                            body_global_pos_error_mean=str(
                                baseline["body_global_pos_error_mean"] * 0.9
                            ),
                            ee_global_pos_error_mean=str(
                                baseline["ee_global_pos_error_mean"] * 0.9
                            ),
                        ),
                        _accepted_mpc(),
                        samples="192",
                        seed=str(seed),
                        duration_sec=str(120 + seed),
                        ref_score=str(baseline["score"] + 0.5),
                        ref_root_pos_error_mean=str(
                            baseline["root_pos_error_mean"] * 0.9
                        ),
                        ref_body_global_pos_error_mean=str(
                            baseline["body_global_pos_error_mean"] * 0.9
                        ),
                        ref_ee_global_pos_error_mean=str(
                            baseline["ee_global_pos_error_mean"] * 0.9
                        ),
                    )
                    for seed in (0, 1, 2)
                ],
            )

            rows = runner.write_pareto_summary(
                output_root=output_root,
                candidate_names=stage.PARETO_CANDIDATE_NAMES,
                motion_names=("jump",),
                sample_counts=(192,),
                seed_values=stage.PARETO_REPEAT_SEEDS,
            )
            with (output_root / "pareto_summary.csv").open() as f:
                csv_rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["motion"], "jump")
        self.assertEqual(rows[0]["samples"], "192")
        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[0]["repeat_count"], "3")
        self.assertEqual(rows[0]["pareto_class"], "promising_budget")
        self.assertEqual(rows[0]["duration_sec_mean"], "121")
        self.assertEqual(csv_rows, rows)

    def test_pareto_summary_marks_missing_expected_seeds_unstable(self) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_pareto_rows(output_root, seeds=(0, 1))

            rows = runner.write_pareto_summary(
                output_root=output_root,
                candidate_names=stage.PARETO_CANDIDATE_NAMES,
                motion_names=("jump",),
                sample_counts=(192,),
                seed_values=(0, 1, 2),
            )

        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[0]["repeat_count"], "2")
        self.assertEqual(rows[0]["pareto_class"], "unstable")
        self.assertIn("missing_seeds", rows[0]["failure_labels"])

    def test_pareto_summary_marks_duplicate_expected_seeds_unstable(self) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_pareto_rows(output_root, seeds=(0, 1, 1, 2))

            rows = runner.write_pareto_summary(
                output_root=output_root,
                candidate_names=stage.PARETO_CANDIDATE_NAMES,
                motion_names=("jump",),
                sample_counts=(192,),
                seed_values=(0, 1, 2),
            )

        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[0]["repeat_count"], "4")
        self.assertEqual(rows[0]["pareto_class"], "unstable")
        self.assertIn("duplicate_seeds", rows[0]["failure_labels"])

    def test_pareto_summary_ignores_extra_seeds_outside_expected_set(self) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_pareto_rows(output_root, seeds=(0, 1, 2, 9))

            rows = runner.write_pareto_summary(
                output_root=output_root,
                candidate_names=stage.PARETO_CANDIDATE_NAMES,
                motion_names=("jump",),
                sample_counts=(192,),
                seed_values=(0, 1, 2),
            )

        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[0]["repeat_count"], "3")
        self.assertEqual(rows[0]["pareto_class"], "promising_budget")
        self.assertNotIn("missing_seeds", rows[0]["failure_labels"])
        self.assertNotIn("duplicate_seeds", rows[0]["failure_labels"])

    def test_pareto_decision_report_recommends_fastest_baseline_close_sample(
        self,
    ) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            report_path = runner.write_pareto_decision_report(
                output_root,
                (
                    _pareto_decision_row("256", "baseline_close"),
                    _pareto_decision_row("192", "baseline_close"),
                ),
            )
            report = report_path.read_text()

        self.assertEqual(report_path.name, "pareto_decision.md")
        self.assertIn("Conclusion: sample_budget_likely", report)
        self.assertIn("Recommended next step", report)
        self.assertIn("192", report)
        self.assertNotIn("256", report)

    def test_pareto_decision_report_recommends_two_promising_samples(self) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            report_path = runner.write_pareto_decision_report(
                output_root,
                (
                    _pareto_decision_row("192", "promising_budget"),
                    _pareto_decision_row("128", "promising_budget"),
                ),
            )
            report = report_path.read_text()

        self.assertIn("Conclusion: sample_budget_partial", report)
        self.assertIn("128", report)
        self.assertIn("192", report)

    def test_pareto_decision_report_recommends_stability_repair_for_unstable(
        self,
    ) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            report_path = runner.write_pareto_decision_report(
                output_root,
                (_pareto_decision_row("192", "unstable"),),
            )
            report = report_path.read_text()

        self.assertIn("Conclusion: stability_likely", report)
        self.assertIn("stability/acceptance repair", report)

    def test_pareto_decision_report_recommends_reward_gating_repair_for_low_quality(
        self,
    ) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            report_path = runner.write_pareto_decision_report(
                output_root,
                (
                    _pareto_decision_row("128", "stable_but_low_quality"),
                    _pareto_decision_row("256", "stable_but_low_quality"),
                ),
            )
            report = report_path.read_text()

        self.assertIn("Conclusion: reward_gating_likely", report)
        self.assertIn("best_s128", report)

    def test_transfer_summary_rejects_non_finite_duration_values(self) -> None:
        runner = _load_runner()
        baseline = stage.BEST_S128_BASELINES["walk"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_csv(
                output_root / "candidates" / "jg_s128_v14_control" / "summary.csv",
                [
                    _summary_row(
                        "walk",
                        _candidate_metrics(
                            "walk",
                            joint_pos_error_mean=str(
                                baseline["joint_pos_error_mean"] * 0.96
                            ),
                        ),
                        _accepted_mpc(),
                        duration_sec="inf",
                    )
                ],
            )

            rows = runner.write_transfer_summary(
                output_root=output_root,
                candidate_names=("jg_s128_v14_control",),
                motion_names=("walk",),
            )

        self.assertEqual(rows[0]["transfer_class"], "invalid")
        self.assertEqual(rows[0]["runtime_guardrail_passed"], "false")
        self.assertIn("runtime_guardrail", rows[0]["failure_labels"])

    def test_transfer_summary_does_not_serialize_non_finite_diagnostics(
        self,
    ) -> None:
        runner = _load_runner()
        baseline = stage.BEST_S128_BASELINES["walk"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_csv(
                output_root / "candidates" / "jg_s128_v14_control" / "summary.csv",
                [
                    _summary_row(
                        "walk",
                        _candidate_metrics(
                            "walk",
                            joint_pos_error_mean=str(
                                baseline["joint_pos_error_mean"] * 0.96
                            ),
                            ee_local_pos_error_mean="nan",
                        ),
                        _accepted_mpc(),
                        duration_sec="100.0",
                    )
                ],
            )

            rows = runner.write_transfer_summary(
                output_root=output_root,
                candidate_names=("jg_s128_v14_control",),
                motion_names=("walk",),
            )
            with (output_root / "transfer_summary.csv").open() as f:
                csv_rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["transfer_class"], "invalid")
        self.assertIn("required_metrics", rows[0]["failure_labels"])
        for row in csv_rows:
            for value in row.values():
                self.assertNotIn(value.lower(), {"nan", "inf", "-inf"})

    def test_transfer_summary_sorting_uses_same_class_tie_breakers(self) -> None:
        runner = _load_runner()
        candidate_names = (
            "jg_s128_v14_control",
            "jg_s128_L148_posture",
            "jg_s128_L142_smooth005",
            "jg_s128_L145_smooth005",
            "jg_s128_L148_smooth005",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            for candidate_name, score in zip(
                candidate_names,
                ("1.0", "2.0", "3.0", "4.0", "5.0"),
            ):
                _write_csv(
                    output_root / "candidates" / candidate_name / "summary.csv",
                    [
                        _summary_row(
                            "walk",
                            _candidate_metrics("walk", score=score),
                            _accepted_mpc(),
                            duration_sec="100.0",
                        )
                    ],
                )

            with mock.patch.object(
                runner.stage,
                "assess_transfer_candidate",
                side_effect=_fake_transfer_tie_assessment,
            ):
                rows = runner.write_transfer_summary(
                    output_root=output_root,
                    candidate_names=candidate_names,
                    motion_names=("walk",),
                )

        self.assertEqual(
            [row["candidate"] for row in rows],
            [
                "jg_s128_L148_posture",
                "jg_s128_L148_smooth005",
                "jg_s128_L145_smooth005",
                "jg_s128_L142_smooth005",
                "jg_s128_v14_control",
            ],
        )


def _fake_transfer_assessment(
    motion_name: str,
    metrics: dict[str, str],
    mpc: dict[str, str],
    *,
    duration_sec: float | None,
    control_duration_sec: float | None,
) -> dict[str, object]:
    del motion_name, mpc, duration_sec, control_duration_sec
    by_score = {
        "1.0": _transfer_result("transfer_pass", local_count_3=1, score_delta=0.2),
        "2.0": _transfer_result(
            "invalid",
            failure_labels=["global_guardrail"],
            global_guardrail_passed=False,
            local_count_3=3,
            score_delta=0.5,
            max_global_ratio=1.06,
        ),
        "3.0": _transfer_result(
            "recovery_candidate",
            failure_labels=["contact_guardrail"],
            contact_guardrail_passed=False,
            local_count_3=2,
            score_delta=0.1,
        ),
    }
    return by_score[str(metrics["score"])]


def _fake_transfer_tie_assessment(
    motion_name: str,
    metrics: dict[str, str],
    mpc: dict[str, str],
    *,
    duration_sec: float | None,
    control_duration_sec: float | None,
) -> dict[str, object]:
    del motion_name, mpc, duration_sec, control_duration_sec
    by_score = {
        "1.0": _transfer_result(
            "transfer_pass",
            local_count_3=1,
            max_local_regression=0.0,
            score_delta=0.5,
            max_global_ratio=0.8,
        ),
        "2.0": _transfer_result(
            "transfer_pass",
            local_count_3=3,
            max_local_regression=0.02,
            score_delta=0.0,
            max_global_ratio=1.0,
        ),
        "3.0": _transfer_result(
            "transfer_pass",
            local_count_3=2,
            max_local_regression=0.01,
            score_delta=0.1,
            max_global_ratio=1.0,
        ),
        "4.0": _transfer_result(
            "transfer_pass",
            local_count_3=2,
            max_local_regression=0.01,
            score_delta=0.2,
            max_global_ratio=1.0,
        ),
        "5.0": _transfer_result(
            "transfer_pass",
            local_count_3=2,
            max_local_regression=0.01,
            score_delta=0.2,
            max_global_ratio=0.9,
        ),
    }
    return by_score[str(metrics["score"])]


def _transfer_result(
    transfer_class: str,
    *,
    failure_labels: list[str] | None = None,
    score_guardrail_passed: bool = True,
    global_guardrail_passed: bool = True,
    smooth_guardrail_passed: bool = True,
    contact_guardrail_passed: bool = True,
    local_transfer_passed: bool = True,
    mpc_guardrail_passed: bool = True,
    runtime_guardrail_passed: bool = True,
    local_count_3: int = 1,
    max_local_regression: float = 0.0,
    score_delta: float = 0.0,
    max_global_ratio: float = 1.0,
    contact_delta: float = 0.0,
    control_delta_regression: float = 0.0,
    joint_acc_regression: float = 0.0,
) -> dict[str, object]:
    return {
        "transfer_class": transfer_class,
        "failure_labels": failure_labels or [],
        "score_guardrail_passed": score_guardrail_passed,
        "global_guardrail_passed": global_guardrail_passed,
        "smooth_guardrail_passed": smooth_guardrail_passed,
        "contact_guardrail_passed": contact_guardrail_passed,
        "local_transfer_passed": local_transfer_passed,
        "mpc_guardrail_passed": mpc_guardrail_passed,
        "runtime_guardrail_passed": runtime_guardrail_passed,
        "local_count_3": local_count_3,
        "max_local_regression": max_local_regression,
        "score_delta": score_delta,
        "max_global_ratio": max_global_ratio,
        "contact_delta": contact_delta,
        "control_delta_regression": control_delta_regression,
        "joint_acc_regression": joint_acc_regression,
    }


def _candidate_metrics(motion: str, **overrides: str) -> dict[str, str]:
    values = {key: str(value) for key, value in stage.BEST_S128_BASELINES[motion].items()}
    values["num_steps"] = "800.0"
    values.update(overrides)
    return values


def _accepted_mpc() -> dict[str, str]:
    return {
        "accepted": "True",
        "accepted_windows": "40",
        "used_baseline_fallback": "False",
    }


def _summary_row(
    motion: str,
    metrics: dict[str, str],
    mpc: dict[str, str],
    **values: str,
) -> dict[str, str]:
    row = {"motion_name": motion, "status": "ok", **values}
    row.update({f"metric_{key}": value for key, value in metrics.items()})
    row.update({f"mpc_{key}": value for key, value in mpc.items()})
    return row


def _pareto_decision_row(samples: str, pareto_class: str) -> dict[str, str]:
    return {
        "candidate": "jg_s128_v14_control",
        "motion": "jump",
        "samples": samples,
        "status": "completed",
        "pareto_class": pareto_class,
    }


def _write_pareto_rows(output_root: Path, *, seeds: tuple[int, ...]) -> None:
    baseline = stage.BEST_S128_BASELINES["jump"]
    _write_csv(
        output_root / "candidates" / "jg_s128_v14_control" / "summary.csv",
        [
            _summary_row(
                "jump",
                _candidate_metrics(
                    "jump",
                    success="True",
                    score=str(baseline["score"] + 0.1),
                    root_pos_error_mean=str(baseline["root_pos_error_mean"] * 0.9),
                    body_global_pos_error_mean=str(
                        baseline["body_global_pos_error_mean"] * 0.9
                    ),
                    ee_global_pos_error_mean=str(
                        baseline["ee_global_pos_error_mean"] * 0.9
                    ),
                ),
                _accepted_mpc(),
                samples="192",
                seed=str(seed),
                duration_sec=str(120 + seed),
                ref_score=str(baseline["score"] + 0.5),
                ref_root_pos_error_mean=str(baseline["root_pos_error_mean"] * 0.9),
                ref_body_global_pos_error_mean=str(
                    baseline["body_global_pos_error_mean"] * 0.9
                ),
                ref_ee_global_pos_error_mean=str(
                    baseline["ee_global_pos_error_mean"] * 0.9
                ),
            )
            for seed in seeds
        ],
    )


def _load_runner():
    if not RUNNER_PATH.exists():
        raise AssertionError(f"missing runner script: {RUNNER_PATH}")
    module_name = "_g1_wbc_local_first_stage_transfer_runner_test"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load runner script: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
