from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from spider.tasks.g1_wbc import local_first_stage as stage

RUNNER_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "run_g1_wbc_local_first_stage.py"
)


class LocalFirstCandidateTest(unittest.TestCase):
    def test_candidate_reward_weights_overrides_joint_global_only(self) -> None:
        base = {
            "g1_wbc_joint_global": {
                "body_local_pos_error": 5.0,
                "body_global_pos_error": 58.0,
            },
            "g1_wbc_joint": {"body_local_pos_error": 99.0},
        }

        weights = stage.candidate_reward_weights(base, "jg_s128_L125_posture")

        self.assertEqual(
            weights["g1_wbc_joint_global"]["body_local_pos_error"],
            6.25,
        )
        self.assertEqual(
            weights["g1_wbc_joint_global"]["body_global_pos_error"],
            58.0,
        )
        self.assertEqual(weights["g1_wbc_joint_global"]["joint_pos_error"], 0.25)
        self.assertEqual(weights["g1_wbc_joint"], base["g1_wbc_joint"])
        self.assertNotIn("joint_pos_error", base["g1_wbc_joint_global"])

    def test_candidate_definitions_are_immutable(self) -> None:
        candidate = stage.CANDIDATE_DEFINITIONS["jg_s128_L150_posture"]

        with self.assertRaises(TypeError):
            candidate.overrides["joint_pos_error"] = 123.0

    def test_next_stage_candidate_names_are_exact_order(self) -> None:
        self.assertEqual(
            stage.NEXT_STAGE_CANDIDATE_NAMES,
            (
                "jg_s128_v14_control",
                "jg_s128_v14_smooth015",
                "jg_s128_v14_smooth025",
                "jg_s128_v14_contact015",
                "jg_s128_v14_smooth015_contact015",
                "jg_s128_L135_posture",
                "jg_s128_L140_posture",
                "jg_s128_L145_posture",
                "jg_s128_L140_smooth015",
                "jg_s128_L140_contact015",
                "jg_s128_L145_smooth015_contact015",
            ),
        )

    def test_upper_bound_candidate_names_are_exact_order(self) -> None:
        self.assertEqual(
            stage.UPPER_BOUND_CANDIDATE_NAMES,
            (
                "jg_s128_v14_control",
                "jg_s128_L138_posture",
                "jg_s128_L140_posture",
                "jg_s128_L142_posture",
                "jg_s128_L145_posture",
                "jg_s128_L148_posture",
                "jg_s128_L150_posture",
                "jg_s128_L155_posture",
                "jg_s128_L142_smooth005",
                "jg_s128_L145_smooth005",
                "jg_s128_L148_smooth005",
                "jg_s128_L150_smooth005",
            ),
        )

    def test_candidate_reward_weights_include_next_stage_matrix(self) -> None:
        smooth015 = {
            "control_delta": 1.035,
            "action_delta": 0.345,
            "joint_acc": 0.00345,
            "joint_jerk": 0.00092,
        }
        smooth025 = {
            "control_delta": 1.125,
            "action_delta": 0.375,
            "joint_acc": 0.00375,
            "joint_jerk": 0.001,
        }
        contact015 = {
            "contact_switch": 6.9,
            "contact_force_delta": 1.15,
            "contact_false_positive": 0.69,
            "contact_false_negative": 0.2875,
        }
        posture135 = {
            "body_local_pos_error": 6.75,
            "body_local_rot_error": 1.19,
            "ee_local_pos_error": 5.4,
            "ee_local_rot_error": 0.833,
            "hand_local_pos_error": 4.05,
            "joint_pos_error": 0.31,
        }
        posture140 = {
            "body_local_pos_error": 7.0,
            "body_local_rot_error": 1.21,
            "ee_local_pos_error": 5.6,
            "ee_local_rot_error": 0.847,
            "hand_local_pos_error": 4.2,
            "joint_pos_error": 0.34,
        }
        posture145 = {
            "body_local_pos_error": 7.25,
            "body_local_rot_error": 1.23,
            "ee_local_pos_error": 5.8,
            "ee_local_rot_error": 0.861,
            "hand_local_pos_error": 4.35,
            "joint_pos_error": 0.37,
        }
        expected = {
            "jg_s128_v14_control": {},
            "jg_s128_v14_smooth015": smooth015,
            "jg_s128_v14_smooth025": smooth025,
            "jg_s128_v14_contact015": contact015,
            "jg_s128_v14_smooth015_contact015": {**smooth015, **contact015},
            "jg_s128_L135_posture": posture135,
            "jg_s128_L140_posture": posture140,
            "jg_s128_L145_posture": posture145,
            "jg_s128_L140_smooth015": {**posture140, **smooth015},
            "jg_s128_L140_contact015": {**posture140, **contact015},
            "jg_s128_L145_smooth015_contact015": {
                **posture145,
                **smooth015,
                **contact015,
            },
        }

        for candidate_name, overrides in expected.items():
            with self.subTest(candidate_name=candidate_name):
                weights = stage.candidate_reward_weights(
                    {stage.JOINT_GLOBAL_METHOD: {}},
                    candidate_name,
                )

                self.assertEqual(weights[stage.JOINT_GLOBAL_METHOD], overrides)

    def test_candidate_reward_weights_include_upper_bound_matrix(self) -> None:
        smooth005 = {
            "control_delta": 0.945,
            "action_delta": 0.315,
            "joint_acc": 0.00315,
            "joint_jerk": 0.00084,
        }
        posture138 = {
            "body_local_pos_error": 6.9,
            "body_local_rot_error": 1.202,
            "ee_local_pos_error": 5.52,
            "ee_local_rot_error": 0.8414,
            "hand_local_pos_error": 4.14,
            "joint_pos_error": 0.328,
        }
        posture142 = {
            "body_local_pos_error": 7.1,
            "body_local_rot_error": 1.218,
            "ee_local_pos_error": 5.68,
            "ee_local_rot_error": 0.8526,
            "hand_local_pos_error": 4.26,
            "joint_pos_error": 0.352,
        }
        posture145 = {
            "body_local_pos_error": 7.25,
            "body_local_rot_error": 1.23,
            "ee_local_pos_error": 5.8,
            "ee_local_rot_error": 0.861,
            "hand_local_pos_error": 4.35,
            "joint_pos_error": 0.37,
        }
        posture148 = {
            "body_local_pos_error": 7.4,
            "body_local_rot_error": 1.242,
            "ee_local_pos_error": 5.92,
            "ee_local_rot_error": 0.8694,
            "hand_local_pos_error": 4.44,
            "joint_pos_error": 0.388,
        }
        posture150 = {
            "body_local_pos_error": 7.5,
            "body_local_rot_error": 1.25,
            "ee_local_pos_error": 6.0,
            "ee_local_rot_error": 0.875,
            "hand_local_pos_error": 4.5,
            "joint_pos_error": 0.40,
        }
        posture155 = {
            "body_local_pos_error": 7.75,
            "body_local_rot_error": 1.27,
            "ee_local_pos_error": 6.2,
            "ee_local_rot_error": 0.889,
            "hand_local_pos_error": 4.65,
            "joint_pos_error": 0.43,
        }
        expected = {
            "jg_s128_L138_posture": posture138,
            "jg_s128_L142_posture": posture142,
            "jg_s128_L148_posture": posture148,
            "jg_s128_L155_posture": posture155,
            "jg_s128_L142_smooth005": {**posture142, **smooth005},
            "jg_s128_L145_smooth005": {**posture145, **smooth005},
            "jg_s128_L148_smooth005": {**posture148, **smooth005},
            "jg_s128_L150_smooth005": {**posture150, **smooth005},
        }

        for candidate_name, overrides in expected.items():
            with self.subTest(candidate_name=candidate_name):
                weights = stage.candidate_reward_weights(
                    {stage.JOINT_GLOBAL_METHOD: {}},
                    candidate_name,
                )

                self.assertEqual(weights[stage.JOINT_GLOBAL_METHOD], overrides)

    def test_write_candidate_reward_files_writes_selected_json(self) -> None:
        base = {
            "g1_wbc_joint_global": {
                "body_local_pos_error": 5.0,
                "control_delta": 1.8,
            },
            "g1_wbc_joint": {"body_local_pos_error": 99.0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.json"
            base_path.write_text(json.dumps(base))

            paths = stage.write_candidate_reward_files(
                base_path,
                root / "reward_weights",
                candidate_names=("jg_s128_v14_control", "jg_s128_L150_smooth"),
            )

            control = json.loads(paths["jg_s128_v14_control"].read_text())
            smooth = json.loads(paths["jg_s128_L150_smooth"].read_text())

        self.assertEqual(
            set(paths),
            {"jg_s128_v14_control", "jg_s128_L150_smooth"},
        )
        self.assertEqual(control, base)
        self.assertEqual(smooth["g1_wbc_joint_global"]["body_local_pos_error"], 7.5)
        self.assertEqual(smooth["g1_wbc_joint_global"]["joint_pos_error"], 0.40)
        self.assertEqual(smooth["g1_wbc_joint_global"]["control_delta"], 1.08)
        self.assertEqual(smooth["g1_wbc_joint"]["body_local_pos_error"], 99.0)


class LocalFirstGuardrailTest(unittest.TestCase):
    def test_assess_candidate_passes_jump_with_local_improvement_and_global_guardrail(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            score=baseline["score"] - 0.01,
            root_pos_error_mean=baseline["root_pos_error_mean"] * 1.04,
            body_global_pos_error_mean=baseline["body_global_pos_error_mean"] * 1.04,
            ee_global_pos_error_mean=baseline["ee_global_pos_error_mean"] * 1.04,
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.94,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.94,
            joint_acc_mean=baseline["joint_acc_mean"] * 1.01,
        )

        result = stage.assess_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["failure_labels"], [])
        self.assertEqual(result["improved_local_count"], 2)
        self.assertTrue(result["global_guardrail_passed"])
        self.assertTrue(result["smooth_guardrail_passed"])
        self.assertTrue(result["contact_guardrail_passed"])
        self.assertTrue(result["score_guardrail_passed"])

    def test_assess_candidate_fails_jump_when_global_guardrail_regresses(self) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            root_pos_error_mean=baseline["root_pos_error_mean"] * 1.06,
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.94,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.94,
        )

        result = stage.assess_candidate("jump", metrics, _accepted_mpc())

        self.assertFalse(result["passed"])
        self.assertFalse(result["global_guardrail_passed"])
        self.assertIn("global_guardrail", result["failure_labels"])
        self.assertEqual(result["improved_local_count"], 2)

    def test_assess_candidate_fails_qixing_when_contact_target_is_missed(self) -> None:
        metrics = _candidate_metrics("qixing", contact_mismatch_rate=0.221)

        result = stage.assess_candidate("qixing", metrics, _accepted_mpc())

        self.assertFalse(result["passed"])
        self.assertFalse(result["contact_guardrail_passed"])
        self.assertIn("contact_guardrail", result["failure_labels"])

    def test_assess_candidate_fails_when_mpc_used_baseline_fallback(self) -> None:
        metrics = _candidate_metrics("walk")

        result = stage.assess_candidate(
            "walk",
            metrics,
            {
                "accepted": True,
                "accepted_windows": 40,
                "used_baseline_fallback": True,
            },
        )

        self.assertFalse(result["passed"])
        self.assertIn("mpc_guardrail", result["failure_labels"])

    def test_assess_candidate_fails_when_accepted_windows_is_not_full_count(
        self,
    ) -> None:
        metrics = _candidate_metrics("walk")

        result = stage.assess_candidate(
            "walk",
            metrics,
            {
                "accepted": True,
                "accepted_windows": 39,
                "used_baseline_fallback": False,
            },
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["mpc_guardrail_passed"])
        self.assertIn("mpc_guardrail", result["failure_labels"])

    def test_assess_candidate_fails_when_num_steps_is_missing(self) -> None:
        metrics = _candidate_metrics("walk")
        metrics.pop("num_steps")

        result = stage.assess_candidate("walk", metrics, _accepted_mpc())

        self.assertFalse(result["passed"])
        self.assertFalse(result["mpc_guardrail_passed"])
        self.assertIn("mpc_guardrail", result["failure_labels"])

    def test_assess_candidate_fails_when_num_steps_is_short(self) -> None:
        metrics = _candidate_metrics("walk", num_steps=799.0)

        result = stage.assess_candidate("walk", metrics, _accepted_mpc())

        self.assertFalse(result["passed"])
        self.assertFalse(result["mpc_guardrail_passed"])
        self.assertIn("mpc_guardrail", result["failure_labels"])

    def test_assess_candidate_fails_when_accepted_windows_is_non_integral(
        self,
    ) -> None:
        metrics = _candidate_metrics("walk")

        result = stage.assess_candidate(
            "walk",
            metrics,
            {
                "accepted": True,
                "accepted_windows": "40.9",
                "used_baseline_fallback": False,
            },
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["mpc_guardrail_passed"])
        self.assertIn("mpc_guardrail", result["failure_labels"])

    def test_assess_candidate_fails_without_control_duration(self) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.94,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.94,
        )

        result = stage.assess_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["runtime_guardrail_passed"])
        self.assertIn("runtime_guardrail", result["failure_labels"])

    def test_assess_candidate_uses_configurable_local_improvement_multiplier(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.96,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.96,
        )

        result = stage.assess_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
            local_improvement_multiplier=stage.MIDDLE_GATE_LOCAL_IMPROVEMENT_MULTIPLIER,
        )
        strict = stage.assess_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
            local_improvement_multiplier=stage.STRICT_LOCAL_IMPROVEMENT_MULTIPLIER,
        )
        default_strict = stage.assess_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(
            stage.LOCAL_IMPROVEMENT_MULTIPLIER,
            stage.STRICT_LOCAL_IMPROVEMENT_MULTIPLIER,
        )
        self.assertTrue(result["local_guardrail_passed"])
        self.assertEqual(result["improved_local_count"], 2)
        self.assertFalse(strict["local_guardrail_passed"])
        self.assertEqual(strict["improved_local_count"], 0)
        self.assertFalse(default_strict["local_guardrail_passed"])
        self.assertEqual(default_strict["improved_local_count"], 0)

    def test_assess_candidate_does_not_count_zero_baselines_as_local_improvements(
        self,
    ) -> None:
        jump_baseline = dict(stage.BEST_S128_BASELINES["jump"])
        jump_baseline.update(
            {metric: 0.0 for metric in stage.LOCAL_IMPROVEMENT_METRICS}
        )
        baselines = dict(stage.BEST_S128_BASELINES)
        baselines["jump"] = jump_baseline

        with mock.patch.object(stage, "BEST_S128_BASELINES", baselines):
            metrics = _candidate_metrics(
                "jump",
                joint_pos_error_mean=0.0,
                body_local_pos_error_mean=0.0,
                ee_local_pos_error_mean=0.0,
            )

            result = stage.assess_candidate(
                "jump",
                metrics,
                _accepted_mpc(),
                duration_sec=100.0,
                control_duration_sec=100.0,
            )

        self.assertEqual(result["improved_local_count"], 0)
        self.assertFalse(result["local_guardrail_passed"])
        self.assertTrue(result["score_guardrail_passed"])
        self.assertTrue(result["global_guardrail_passed"])
        self.assertTrue(result["smooth_guardrail_passed"])
        self.assertTrue(result["contact_guardrail_passed"])
        self.assertTrue(result["mpc_guardrail_passed"])
        self.assertTrue(result["runtime_guardrail_passed"])
        self.assertIn("local_improvement_guardrail", result["failure_labels"])

    def test_assess_candidate_rejects_non_finite_required_metric(self) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.90,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.90,
            ee_local_pos_error_mean=math.nan,
        )

        result = stage.assess_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
            local_improvement_multiplier=stage.MIDDLE_GATE_LOCAL_IMPROVEMENT_MULTIPLIER,
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["required_metrics_passed"])
        self.assertIn("required_metrics", result["failure_labels"])


class LocalFirstUpperBoundAssessmentTest(unittest.TestCase):
    def test_assess_upper_bound_candidate_classifies_two_5pct_locals_as_frontier(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.95,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.95,
        )

        result = stage.assess_upper_bound_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["frontier_class"], "local_frontier")
        self.assertEqual(result["failure_labels"], [])
        self.assertTrue(result["hard_guardrail_passed"])
        self.assertEqual(result["local_count_5"], 2)
        self.assertGreater(result["local_composite_improvement"], 0.0)

    def test_assess_upper_bound_candidate_classifies_two_3pct_locals_as_near_frontier(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.97,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.97,
        )

        result = stage.assess_upper_bound_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["frontier_class"], "near_frontier")
        self.assertEqual(result["failure_labels"], [])
        self.assertEqual(result["local_count_3"], 2)
        self.assertEqual(result["local_count_5"], 0)

    def test_assess_upper_bound_candidate_does_not_count_zero_baselines_as_local_improvements(
        self,
    ) -> None:
        jump_baseline = dict(stage.BEST_S128_BASELINES["jump"])
        jump_baseline.update(
            {metric: 0.0 for metric in stage.LOCAL_IMPROVEMENT_METRICS}
        )
        baselines = dict(stage.BEST_S128_BASELINES)
        baselines["jump"] = jump_baseline

        with mock.patch.object(stage, "BEST_S128_BASELINES", baselines):
            metrics = _candidate_metrics(
                "jump",
                joint_pos_error_mean=0.0,
                body_local_pos_error_mean=0.0,
                ee_local_pos_error_mean=0.0,
            )

            result = stage.assess_upper_bound_candidate(
                "jump",
                metrics,
                _accepted_mpc(),
                duration_sec=100.0,
                control_duration_sec=100.0,
            )

        self.assertEqual(result["local_count_3"], 0)
        self.assertEqual(result["local_count_5"], 0)
        self.assertEqual(result["local_count_8"], 0)
        self.assertNotEqual(result["frontier_class"], "local_frontier")

    def test_assess_upper_bound_candidate_invalidates_relaxed_global_regression(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            root_pos_error_mean=baseline["root_pos_error_mean"] * 1.36,
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.95,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.95,
        )

        result = stage.assess_upper_bound_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["frontier_class"], "invalid")
        self.assertFalse(result["global_relaxed_passed"])
        self.assertIn("relaxed_global_guardrail", result["failure_labels"])
        self.assertGreater(result["max_global_ratio"], 1.35)

    def test_assess_upper_bound_candidate_invalidates_relaxed_score_regression(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            score=baseline["score"] - 0.26,
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.95,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.95,
        )

        result = stage.assess_upper_bound_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["frontier_class"], "invalid")
        self.assertFalse(result["score_relaxed_passed"])
        self.assertIn("relaxed_score_guardrail", result["failure_labels"])
        self.assertLess(result["score_delta"], -0.25)

    def test_assess_upper_bound_candidate_keeps_hard_guardrail_to_mpc_and_runtime(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            score=baseline["score"] - 0.26,
            root_pos_error_mean=baseline["root_pos_error_mean"] * 1.36,
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.95,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.95,
        )

        result = stage.assess_upper_bound_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["frontier_class"], "invalid")
        self.assertTrue(result["hard_guardrail_passed"])
        self.assertFalse(result["score_relaxed_passed"])
        self.assertFalse(result["global_relaxed_passed"])
        self.assertTrue(result["mpc_guardrail_passed"])
        self.assertTrue(result["runtime_guardrail_passed"])


class FastQualityParetoAssessmentTest(unittest.TestCase):
    def test_pareto_group_classifies_baseline_close_budget(self) -> None:
        reference = {
            "score": -2.078,
            "root_pos_error_mean": 0.0425,
            "body_global_pos_error_mean": 0.0530,
            "ee_global_pos_error_mean": 0.0608,
        }
        rows = [
            _pareto_source_row(
                "jump",
                samples=192,
                seed=seed,
                score=-2.12,
                root_pos_error_mean=0.050,
                body_global_pos_error_mean=0.060,
                ee_global_pos_error_mean=0.070,
                reference=reference,
            )
            for seed in (0, 1, 2)
        ]

        result = stage.assess_pareto_group("jump", rows)

        self.assertEqual(result["pareto_class"], "baseline_close")
        self.assertEqual(result["repeat_count"], 3)
        self.assertEqual(result["ok_count"], 3)
        self.assertEqual(result["success_count"], 3)
        self.assertEqual(result["accepted_count"], 3)
        self.assertEqual(result["fallback_count"], 0)
        self.assertEqual(result["full_run_count"], 3)
        self.assertEqual(result["accepted_windows_count"], 3)
        self.assertAlmostEqual(result["score_delta_vs_best_mean"], 0.1457276578247549)
        self.assertAlmostEqual(result["score_delta_vs_baseline_mean"], -0.042)
        self.assertLessEqual(result["max_global_ratio_vs_baseline_mean"], 1.35)
        self.assertTrue(_expected_pareto_keys() <= set(result))

    def test_pareto_group_classifies_promising_budget_against_best_s128(
        self,
    ) -> None:
        best = stage.BEST_S128_BASELINES["jump"]
        reference = {
            "score": -2.078,
            "root_pos_error_mean": 0.0425,
            "body_global_pos_error_mean": 0.0530,
            "ee_global_pos_error_mean": 0.0608,
        }
        rows = [
            _pareto_source_row(
                "jump",
                samples=192,
                seed=seed,
                score=best["score"] + 0.10,
                root_pos_error_mean=best["root_pos_error_mean"] * 0.95,
                body_global_pos_error_mean=best["body_global_pos_error_mean"] * 0.95,
                ee_global_pos_error_mean=best["ee_global_pos_error_mean"] * 0.95,
                reference=reference,
            )
            for seed in (0, 1, 2)
        ]

        result = stage.assess_pareto_group("jump", rows)

        self.assertEqual(result["pareto_class"], "promising_budget")
        self.assertGreater(result["score_delta_vs_best_mean"], 0.0)
        self.assertLessEqual(result["max_global_ratio_vs_best_mean"], 1.0)
        self.assertGreater(result["max_global_ratio_vs_baseline_mean"], 1.35)

    def test_pareto_group_accepts_canonical_string_booleans_for_stable_classes(
        self,
    ) -> None:
        best = stage.BEST_S128_BASELINES["jump"]
        strict_reference = {
            "score": -2.078,
            "root_pos_error_mean": 0.0425,
            "body_global_pos_error_mean": 0.0530,
            "ee_global_pos_error_mean": 0.0608,
        }
        cases = (
            (
                "baseline_close",
                [
                    _pareto_source_row("jump", samples=192, seed=seed)
                    for seed in (0, 1, 2)
                ],
            ),
            (
                "promising_budget",
                [
                    _pareto_source_row(
                        "jump",
                        samples=192,
                        seed=seed,
                        score=best["score"] + 0.10,
                        root_pos_error_mean=best["root_pos_error_mean"] * 0.95,
                        body_global_pos_error_mean=(
                            best["body_global_pos_error_mean"] * 0.95
                        ),
                        ee_global_pos_error_mean=(
                            best["ee_global_pos_error_mean"] * 0.95
                        ),
                        reference=strict_reference,
                    )
                    for seed in (0, 1, 2)
                ],
            ),
        )

        for expected_class, rows in cases:
            with self.subTest(expected_class=expected_class):
                for index, row in enumerate(rows):
                    row["metric_success"] = "True" if index % 2 == 0 else "true"
                    row["mpc_accepted"] = "true" if index % 2 == 0 else "True"
                    row["mpc_used_baseline_fallback"] = (
                        "False" if index % 2 == 0 else "false"
                    )

                result = stage.assess_pareto_group("jump", rows)

                self.assertEqual(result["pareto_class"], expected_class)
                self.assertEqual(result["failure_labels"], [])
                self.assertEqual(result["success_count"], 3)
                self.assertEqual(result["accepted_count"], 3)
                self.assertEqual(result["fallback_count"], 0)

    def test_pareto_group_marks_missing_or_non_finite_primary_metric_unstable(
        self,
    ) -> None:
        cases = ("missing", "nan")

        for case in cases:
            with self.subTest(case=case):
                rows = [
                    _pareto_source_row("jump", samples=192, seed=seed)
                    for seed in (0, 1, 2)
                ]
                if case == "missing":
                    del rows[1]["metric_score"]
                else:
                    rows[1]["metric_score"] = "nan"

                result = stage.assess_pareto_group("jump", rows)

                self.assertEqual(result["pareto_class"], "unstable")
                self.assertEqual(result["ok_count"], 3)
                self.assertIn("required_metrics", result["failure_labels"])

    def test_pareto_group_marks_missing_reference_metrics_unstable(self) -> None:
        for key in ("ref_root_pos_error_mean", "ref_score"):
            with self.subTest(key=key):
                rows = [
                    _pareto_source_row("jump", samples=192, seed=seed)
                    for seed in (0, 1, 2)
                ]
                del rows[1][key]

                result = stage.assess_pareto_group("jump", rows)

                self.assertEqual(result["pareto_class"], "unstable")
                self.assertEqual(result["ok_count"], 3)
                self.assertIn("reference_metrics", result["failure_labels"])

    def test_pareto_group_rejects_malformed_csv_booleans_as_hard_constraints(
        self,
    ) -> None:
        cases = (
            ("metric_success", "yes", "metric_success"),
            ("mpc_accepted", "yes", "mpc_accepted"),
            ("mpc_used_baseline_fallback", "no", "baseline_fallback"),
        )

        for key, value, expected_label in cases:
            with self.subTest(key=key):
                rows = [
                    _pareto_source_row("jump", samples=192, seed=seed)
                    for seed in (0, 1, 2)
                ]
                rows[1][key] = value

                result = stage.assess_pareto_group("jump", rows)

                self.assertEqual(result["pareto_class"], "unstable")
                self.assertIn(expected_label, result["failure_labels"])

    def test_pareto_group_classifies_unstable_when_a_repeat_fails(self) -> None:
        rows = [
            _pareto_source_row("jump", samples=128, seed=0),
            _pareto_source_row("jump", samples=128, seed=1, metric_success=False),
            _pareto_source_row("jump", samples=128, seed=2),
        ]

        result = stage.assess_pareto_group("jump", rows)

        self.assertEqual(result["pareto_class"], "unstable")
        self.assertEqual(result["success_count"], 2)
        self.assertIn("metric_success", result["failure_labels"])

    def test_pareto_group_classifies_invalid_without_usable_repeats(self) -> None:
        result = stage.assess_pareto_group("jump", [])

        self.assertEqual(result["pareto_class"], "invalid")
        self.assertEqual(result["repeat_count"], 0)
        self.assertIn("no_repeats", result["failure_labels"])


class LocalFirstRunnerTest(unittest.TestCase):
    def test_selected_candidate_names_defaults_to_next_stage_matrix(self) -> None:
        runner = _load_runner()

        self.assertEqual(
            runner.selected_candidate_names(None),
            stage.NEXT_STAGE_CANDIDATE_NAMES,
        )

    def test_selected_candidate_names_defaults_to_upper_bound_matrix(self) -> None:
        runner = _load_runner()

        self.assertEqual(
            runner.selected_candidate_names(None, candidate_set="upper-bound"),
            stage.UPPER_BOUND_CANDIDATE_NAMES,
        )

    def test_selected_candidate_names_include_contact_appends_once(self) -> None:
        runner = _load_runner()

        default_with_contact = runner.selected_candidate_names(
            None,
            include_contact=True,
        )
        explicit_with_contact = runner.selected_candidate_names(
            ("jg_s128_L125_posture", "jg_s128_L150_contact"),
            include_contact=True,
        )

        self.assertEqual(default_with_contact.count("jg_s128_L150_contact"), 1)
        self.assertEqual(default_with_contact[-1], "jg_s128_L150_contact")
        self.assertEqual(
            explicit_with_contact,
            ("jg_s128_L125_posture", "jg_s128_L150_contact"),
        )

    def test_plan_sweep_command_uses_fixed_local_first_stage_parameters(self) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            commands = runner.plan_sweep_commands(
                output_root=output_root,
                candidate_names=("jg_s128_L125_posture",),
                motion_names=("jump",),
                python_executable="/opt/python",
                device="cpu",
            )

        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[0], "/opt/python")
        self.assertEqual(Path(command[1]).name, "run_g1_wbc_low_sample_sweep.py")
        self.assertIn("--execute", command)
        self.assertEqual(_flag_value(command, "--samples"), "128")
        self.assertEqual(_flag_value(command, "--iterations"), "2")
        self.assertEqual(_flag_value(command, "--horizons"), "40")
        self.assertEqual(_flag_value(command, "--controls"), "20")
        self.assertEqual(_flag_value(command, "--knot-counts"), "8")
        self.assertEqual(_flag_value(command, "--sigma-triplets"), "0.04,0.1,0.18")
        self.assertEqual(_flag_value(command, "--seeds"), "0")
        self.assertEqual(_flag_value(command, "--max-steps"), "800")
        self.assertEqual(_flag_value(command, "--device"), "cpu")
        self.assertEqual(_flag_value(command, "--motions"), "jump")
        self.assertEqual(
            Path(_flag_value(command, "--mpc-reward-weights")),
            output_root / "reward_weights" / "jg_s128_L125_posture.json",
        )
        self.assertEqual(
            Path(_flag_value(command, "--output-root")),
            output_root / "candidates" / "jg_s128_L125_posture",
        )

    def test_dry_run_writes_deterministic_plan_commands_and_reward_files(self) -> None:
        runner = _load_runner()
        base = {
            "g1_wbc_joint_global": {
                "body_local_pos_error": 5.0,
                "body_global_pos_error": 58.0,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            base_path = output_root / "base_reward_weights.json"
            base_path.write_text(json.dumps(base))

            artifacts = runner.write_dry_run_artifacts(
                output_root=output_root,
                base_reward_weights_path=base_path,
                candidate_names=("jg_s128_L125_posture",),
                motion_names=("jump",),
                python_executable="/opt/python",
                device="cpu",
            )

            experiment_plan = json.loads((output_root / "experiment_plan.json").read_text())
            planned_commands = (output_root / "planned_commands.sh").read_text()
            reward_weights = json.loads(
                (
                    output_root
                    / "reward_weights"
                    / "jg_s128_L125_posture.json"
                ).read_text()
            )

        self.assertEqual(
            artifacts["experiment_plan"],
            output_root / "experiment_plan.json",
        )
        self.assertEqual(experiment_plan["dry_run"], True)
        self.assertEqual(experiment_plan["candidate_set"], "middle")
        self.assertEqual(experiment_plan["assessment_mode"], "promotion")
        self.assertEqual(experiment_plan["candidates"], ["jg_s128_L125_posture"])
        self.assertEqual(experiment_plan["motions"], ["jump"])
        self.assertEqual(experiment_plan["command_count"], 1)
        self.assertEqual(experiment_plan["local_improvement_pct"], 3.0)
        self.assertEqual(experiment_plan["local_improvement_multiplier"], 0.97)
        self.assertIn("--samples 128", planned_commands)
        self.assertIn("--mpc-reward-weights", planned_commands)
        self.assertEqual(
            reward_weights["g1_wbc_joint_global"]["body_local_pos_error"],
            6.25,
        )
        self.assertEqual(
            reward_weights["g1_wbc_joint_global"]["body_global_pos_error"],
            58.0,
        )

    def test_upper_bound_dry_run_plan_records_mode_and_all_candidates(self) -> None:
        runner = _load_runner()
        base = {"g1_wbc_joint_global": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            base_path = output_root / "base_reward_weights.json"
            base_path.write_text(json.dumps(base))

            runner.write_dry_run_artifacts(
                output_root=output_root,
                base_reward_weights_path=base_path,
                candidate_names=stage.UPPER_BOUND_CANDIDATE_NAMES,
                motion_names=("jump",),
                python_executable="/opt/python",
                device="cpu",
                candidate_set="upper-bound",
                assessment_mode="upper-bound",
            )

            experiment_plan = json.loads((output_root / "experiment_plan.json").read_text())

        self.assertEqual(experiment_plan["candidate_set"], "upper-bound")
        self.assertEqual(experiment_plan["assessment_mode"], "upper-bound")
        self.assertEqual(experiment_plan["candidates"], list(stage.UPPER_BOUND_CANDIDATE_NAMES))
        self.assertEqual(experiment_plan["command_count"], 12)

    def test_main_infers_upper_bound_assessment_for_upper_bound_candidate_set(
        self,
    ) -> None:
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
                        "upper-bound",
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

        self.assertEqual(exit_code, 0)
        self.assertEqual(experiment_plan["candidate_set"], "upper-bound")
        self.assertEqual(experiment_plan["assessment_mode"], "upper-bound")

    def test_upper_bound_dry_run_writes_planned_frontier_summary(self) -> None:
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
                        "upper-bound",
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

            with (output_root / "frontier_summary.csv").open() as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(rows), len(stage.UPPER_BOUND_CANDIDATE_NAMES))
        self.assertEqual({row["status"] for row in rows}, {"planned"})
        self.assertEqual({row["frontier_class"] for row in rows}, {"invalid"})

    def test_main_preserves_explicit_promotion_mode_for_upper_bound_candidate_set(
        self,
    ) -> None:
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
                        "upper-bound",
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

        self.assertEqual(exit_code, 0)
        self.assertEqual(experiment_plan["candidate_set"], "upper-bound")
        self.assertEqual(experiment_plan["assessment_mode"], "promotion")

    def test_dry_run_writes_promotion_placeholder_without_commands(self) -> None:
        runner = _load_runner()
        base = {
            "g1_wbc_joint_global": {
                "body_local_pos_error": 5.0,
                "body_global_pos_error": 58.0,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            base_path = output_root / "base_reward_weights.json"
            base_path.write_text(json.dumps(base))

            artifacts = runner.write_dry_run_artifacts(
                output_root=output_root,
                base_reward_weights_path=base_path,
                candidate_names=("jg_s128_L125_posture",),
                motion_names=("jump",),
                promoted_motion_names=("walk", "qixing"),
                python_executable="/opt/python",
                device="cpu",
            )
            promoted_commands = (output_root / "promoted_commands.sh").read_text()

        self.assertEqual(
            artifacts["promoted_commands"],
            output_root / "promoted_commands.sh",
        )
        self.assertIn("No promoted candidates yet", promoted_commands)
        self.assertIn("walk qixing", promoted_commands)
        self.assertNotIn("run_g1_wbc_low_sample_sweep.py --execute", promoted_commands)

    def test_execute_mode_writes_experiment_plan_as_not_dry_run(self) -> None:
        runner = _load_runner()
        base = {
            "g1_wbc_joint_global": {
                "body_local_pos_error": 5.0,
                "body_global_pos_error": 58.0,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            base_path = output_root / "base_reward_weights.json"
            base_path.write_text(json.dumps(base))

            with mock.patch.object(
                runner.subprocess,
                "run",
                return_value=_completed_process(0),
            ) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = runner.main(
                        [
                            "--execute",
                            "--output-root",
                            str(output_root),
                            "--base-reward-weights",
                            str(base_path),
                            "--candidates",
                            "jg_s128_L125_posture",
                            "--motions",
                            "jump",
                            "--python-executable",
                            "/opt/python",
                            "--device",
                            "cpu",
                            "--local-improvement-pct",
                            "5.0",
                        ]
                    )

            experiment_plan = json.loads((output_root / "experiment_plan.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(experiment_plan["dry_run"], False)
        self.assertEqual(experiment_plan["local_improvement_pct"], 5.0)
        self.assertEqual(experiment_plan["local_improvement_multiplier"], 0.95)

    def test_parse_args_supports_required_runner_flags(self) -> None:
        runner = _load_runner()

        args = runner.parse_args(
            [
                "--execute",
                "--output-root",
                "/tmp/local-first",
                "--candidates",
                "jg_s128_L125_posture",
                "--motions",
                "jump",
                "--python-executable",
                "/opt/python",
                "--device",
                "cpu",
                "--include-contact",
                "--summarize-only",
                "--local-improvement-pct",
                "5.0",
                "--candidate-set",
                "upper-bound",
                "--assessment-mode",
                "upper-bound",
            ]
        )

        self.assertTrue(args.execute)
        self.assertEqual(args.output_root, "/tmp/local-first")
        self.assertEqual(args.candidates, ["jg_s128_L125_posture"])
        self.assertEqual(args.motions, ["jump"])
        self.assertEqual(args.python_executable, "/opt/python")
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.include_contact)
        self.assertTrue(args.summarize_only)
        self.assertEqual(args.local_improvement_pct, 5.0)
        self.assertEqual(args.candidate_set, "upper-bound")
        self.assertEqual(args.assessment_mode, "upper-bound")

    def test_parse_args_defaults_to_middle_promotion_mode(self) -> None:
        runner = _load_runner()

        args = runner.parse_args([])

        self.assertEqual(args.local_improvement_pct, 3.0)
        self.assertEqual(args.candidate_set, "middle")
        self.assertEqual(args.assessment_mode, "promotion")

    def test_parse_args_rejects_abbreviated_assessment_mode_flag(self) -> None:
        runner = _load_runner()

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.parse_args(["--assessment", "promotion"])

    def test_parse_args_rejects_invalid_local_improvement_pct(self) -> None:
        runner = _load_runner()

        for value in ("100", "-0.1", "nan"):
            with self.subTest(value=value):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(["--local-improvement-pct", value])

    def test_local_improvement_multiplier_rejects_non_finite_pct(self) -> None:
        runner = _load_runner()

        for value in (math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    runner.local_improvement_multiplier_from_pct(value)

    def test_guardrail_summary_marks_planned_and_missing_rows_as_not_passed(self) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            summary_path = (
                output_root
                / "candidates"
                / "jg_s128_L125_posture"
                / "summary.csv"
            )
            _write_csv(
                summary_path,
                [
                    {
                        "motion_name": "jump",
                        "status": "ok",
                        "metric_score": "-2.26",
                        "metric_num_steps": "800",
                        "mpc_accepted": "True",
                        "mpc_accepted_windows": "40",
                        "mpc_used_baseline_fallback": "False",
                    }
                ],
            )

            rows = runner.write_guardrail_summary(
                output_root=output_root,
                candidate_names=("jg_s128_L125_posture", "jg_s128_L150_posture"),
                motion_names=("jump",),
            )
            with (output_root / "guardrail_summary.csv").open() as f:
                guardrail_rows = list(csv.DictReader(f))

        by_candidate = {row["candidate"]: row for row in rows}
        self.assertEqual(by_candidate["jg_s128_L125_posture"]["status"], "missing")
        self.assertEqual(by_candidate["jg_s128_L125_posture"]["passed"], "false")
        self.assertEqual(by_candidate["jg_s128_L150_posture"]["status"], "planned")
        self.assertEqual(by_candidate["jg_s128_L150_posture"]["passed"], "false")
        self.assertEqual(guardrail_rows, rows)

    def test_guardrail_summary_normalizes_missing_metrics_status_to_missing(self) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            summary_path = (
                output_root
                / "candidates"
                / "jg_s128_L125_posture"
                / "summary.csv"
            )
            _write_csv(
                summary_path,
                [
                    {
                        "motion_name": "jump",
                        "status": "missing_metrics",
                    }
                ],
            )

            rows = runner.write_guardrail_summary(
                output_root=output_root,
                candidate_names=("jg_s128_L125_posture",),
                motion_names=("jump",),
            )

        self.assertEqual(rows[0]["status"], "missing")
        self.assertEqual(rows[0]["passed"], "false")

    def test_guardrail_summary_uses_configured_local_improvement_pct(self) -> None:
        runner = _load_runner()
        baseline = stage.BEST_S128_BASELINES["jump"]
        four_pct_local_improvement = _candidate_metrics(
            "jump",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.96,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.96,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_v14_control"
                / "summary.csv",
                [
                    _summary_row(
                        "jump",
                        _candidate_metrics("jump"),
                        _accepted_mpc(),
                        duration_sec="10.0",
                    )
                ],
            )
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_L140_posture"
                / "summary.csv",
                [
                    _summary_row(
                        "jump",
                        four_pct_local_improvement,
                        _accepted_mpc(),
                        duration_sec="10.5",
                    )
                ],
            )

            middle_gate_rows = runner.write_guardrail_summary(
                output_root=output_root,
                candidate_names=("jg_s128_L140_posture",),
                motion_names=("jump",),
                local_improvement_pct=3.0,
            )
            strict_rows = runner.write_guardrail_summary(
                output_root=output_root,
                candidate_names=("jg_s128_L140_posture",),
                motion_names=("jump",),
                local_improvement_pct=5.0,
            )

        self.assertEqual(middle_gate_rows[0]["status"], "passed")
        self.assertEqual(middle_gate_rows[0]["local_guardrail_passed"], "true")
        self.assertEqual(middle_gate_rows[0]["improved_local_count"], "2")
        self.assertEqual(strict_rows[0]["status"], "failed")
        self.assertEqual(strict_rows[0]["local_guardrail_passed"], "false")
        self.assertEqual(strict_rows[0]["improved_local_count"], "0")

    def test_guardrail_summary_writes_promoted_commands_for_passed_screening_rows(
        self,
    ) -> None:
        runner = _load_runner()
        baseline = stage.BEST_S128_BASELINES["jump"]
        passing_metrics = _candidate_metrics(
            "jump",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.94,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.94,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_v14_control"
                / "summary.csv",
                [
                    _summary_row(
                        "jump",
                        passing_metrics,
                        _accepted_mpc(),
                        duration_sec="10.0",
                    )
                ],
            )
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_L125_posture"
                / "summary.csv",
                [
                    _summary_row(
                        "jump",
                        passing_metrics,
                        _accepted_mpc(),
                        duration_sec="10.5",
                    )
                ],
            )
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_L150_posture"
                / "summary.csv",
                [
                    {
                        "motion_name": "jump",
                        "status": "failed",
                    }
                ],
            )
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_L150_smooth"
                / "summary.csv",
                [
                    _summary_row(
                        "jump",
                        passing_metrics,
                        _accepted_mpc(),
                        duration_sec="10.6",
                    )
                ],
            )

            rows = runner.write_guardrail_summary(
                output_root=output_root,
                candidate_names=(
                    "jg_s128_L125_posture",
                    "jg_s128_L150_posture",
                    "jg_s128_L150_smooth",
                ),
                motion_names=("jump",),
                promoted_motion_names=("walk", "qixing"),
                python_executable="/opt/python",
                device="cpu",
            )
            promoted_commands = (output_root / "promoted_commands.sh").read_text()

        self.assertEqual([row["candidate"] for row in rows if row["passed"] == "true"], [
            "jg_s128_L125_posture",
            "jg_s128_L150_smooth",
        ])
        self.assertIn("--motions walk qixing", promoted_commands)
        self.assertIn(
            str(output_root / "promoted" / "jg_s128_L125_posture"),
            promoted_commands,
        )
        self.assertIn(
            str(output_root / "promoted" / "jg_s128_L150_smooth"),
            promoted_commands,
        )
        self.assertNotIn(
            str(output_root / "promoted" / "jg_s128_L150_posture"),
            promoted_commands,
        )

    def test_control_durations_ignore_non_ok_control_rows(self) -> None:
        runner = _load_runner()
        baseline = stage.BEST_S128_BASELINES["jump"]
        candidate_metrics = _candidate_metrics(
            "jump",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.94,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.94,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_v14_control"
                / "summary.csv",
                [
                    {
                        "motion_name": "jump",
                        "status": "failed",
                        "duration_sec": "1.0",
                    }
                ],
            )
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_L125_posture"
                / "summary.csv",
                [
                    _summary_row(
                        "jump",
                        candidate_metrics,
                        _accepted_mpc(),
                        duration_sec="1.2",
                    )
                ],
            )

            rows = runner.write_guardrail_summary(
                output_root=output_root,
                candidate_names=("jg_s128_L125_posture",),
                motion_names=("jump",),
            )

        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["passed"], "false")
        self.assertEqual(rows[0]["runtime_guardrail_passed"], "false")
        self.assertIn("runtime_guardrail", rows[0]["failure_labels"])

    def test_frontier_summary_sorts_local_frontier_before_near_frontier(self) -> None:
        runner = _load_runner()
        baseline = stage.BEST_S128_BASELINES["jump"]
        local_frontier_metrics = _candidate_metrics(
            "jump",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.95,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.95,
        )
        near_frontier_metrics = _candidate_metrics(
            "jump",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.97,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.97,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_v14_control"
                / "summary.csv",
                [
                    _summary_row(
                        "jump",
                        _candidate_metrics("jump"),
                        _accepted_mpc(),
                        duration_sec="100.0",
                    )
                ],
            )
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_L142_posture"
                / "summary.csv",
                [
                    _summary_row(
                        "jump",
                        near_frontier_metrics,
                        _accepted_mpc(),
                        duration_sec="100.0",
                    )
                ],
            )
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_L148_posture"
                / "summary.csv",
                [
                    _summary_row(
                        "jump",
                        local_frontier_metrics,
                        _accepted_mpc(),
                        duration_sec="100.0",
                    )
                ],
            )

            rows = runner.write_frontier_summary(
                output_root=output_root,
                candidate_names=("jg_s128_L142_posture", "jg_s128_L148_posture"),
                motion_names=("jump",),
            )
            with (output_root / "frontier_summary.csv").open() as f:
                frontier_rows = list(csv.DictReader(f))

        self.assertEqual(
            [row["candidate"] for row in rows],
            ["jg_s128_L148_posture", "jg_s128_L142_posture"],
        )
        self.assertEqual([row["rank"] for row in rows], ["1", "2"])
        self.assertEqual(rows[0]["frontier_class"], "local_frontier")
        self.assertEqual(rows[1]["frontier_class"], "near_frontier")
        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[1]["status"], "completed")
        self.assertEqual(frontier_rows, rows)

    def test_frontier_summary_marks_absent_motion_row_as_planned(self) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)

            rows = runner.write_frontier_summary(
                output_root=output_root,
                candidate_names=("jg_s128_L125_posture",),
                motion_names=("jump",),
            )

        self.assertEqual(rows[0]["status"], "planned")
        self.assertEqual(rows[0]["frontier_class"], "invalid")
        self.assertEqual(rows[0]["failure_labels"], "planned")

    def test_frontier_summary_marks_missing_metrics_status_as_missing(self) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_L125_posture"
                / "summary.csv",
                [
                    {
                        "motion_name": "jump",
                        "status": "missing_metrics",
                    }
                ],
            )

            rows = runner.write_frontier_summary(
                output_root=output_root,
                candidate_names=("jg_s128_L125_posture",),
                motion_names=("jump",),
            )

        self.assertEqual(rows[0]["status"], "missing")
        self.assertEqual(rows[0]["frontier_class"], "invalid")
        self.assertEqual(rows[0]["failure_labels"], "missing")

    def test_frontier_summary_preserves_failed_status(self) -> None:
        runner = _load_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            _write_csv(
                output_root
                / "candidates"
                / "jg_s128_L125_posture"
                / "summary.csv",
                [
                    {
                        "motion_name": "jump",
                        "status": "failed",
                    }
                ],
            )

            rows = runner.write_frontier_summary(
                output_root=output_root,
                candidate_names=("jg_s128_L125_posture",),
                motion_names=("jump",),
            )

        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["frontier_class"], "invalid")
        self.assertEqual(rows[0]["failure_labels"], "failed")


def _candidate_metrics(motion: str, **overrides: float | bool) -> dict[str, float | bool]:
    values: dict[str, float | bool] = dict(stage.BEST_S128_BASELINES[motion])
    values["num_steps"] = 800.0
    values.update(overrides)
    return values


def _accepted_mpc() -> dict[str, float | bool]:
    return {
        "accepted": True,
        "accepted_windows": 40,
        "used_baseline_fallback": False,
    }


def _summary_row(
    motion: str,
    metrics: dict[str, float | bool],
    mpc: dict[str, float | bool],
    **values: str,
) -> dict[str, str]:
    row = {"motion_name": motion, "status": "ok", **values}
    row.update({f"metric_{key}": str(value) for key, value in metrics.items()})
    row.update({f"mpc_{key}": str(value) for key, value in mpc.items()})
    return row


def _pareto_source_row(
    motion: str,
    *,
    samples: int,
    seed: int,
    status: str = "ok",
    metric_success: bool = True,
    score: float | None = None,
    root_pos_error_mean: float | None = None,
    body_global_pos_error_mean: float | None = None,
    ee_global_pos_error_mean: float | None = None,
    reference: dict[str, float] | None = None,
) -> dict[str, object]:
    baseline = stage.BEST_S128_BASELINES[motion]
    reference_values = reference or {
        "score": baseline["score"] + 0.20,
        "root_pos_error_mean": baseline["root_pos_error_mean"] * 0.80,
        "body_global_pos_error_mean": baseline["body_global_pos_error_mean"] * 0.80,
        "ee_global_pos_error_mean": baseline["ee_global_pos_error_mean"] * 0.80,
    }
    return {
        "motion_name": motion,
        "samples": samples,
        "seed": seed,
        "status": status,
        "duration_sec": 100.0 + samples / 100.0,
        "metric_success": metric_success,
        "metric_num_steps": 800,
        "metric_score": baseline["score"] if score is None else score,
        "metric_root_pos_error_mean": (
            baseline["root_pos_error_mean"]
            if root_pos_error_mean is None
            else root_pos_error_mean
        ),
        "metric_body_global_pos_error_mean": (
            baseline["body_global_pos_error_mean"]
            if body_global_pos_error_mean is None
            else body_global_pos_error_mean
        ),
        "metric_ee_global_pos_error_mean": (
            baseline["ee_global_pos_error_mean"]
            if ee_global_pos_error_mean is None
            else ee_global_pos_error_mean
        ),
        "metric_ee_local_pos_error_mean": baseline["ee_local_pos_error_mean"],
        "metric_contact_mismatch_rate": baseline["contact_mismatch_rate"],
        "metric_control_delta_mean": baseline["control_delta_mean"],
        "metric_joint_acc_mean": baseline["joint_acc_mean"],
        "mpc_accepted": True,
        "mpc_accepted_windows": 40,
        "mpc_used_baseline_fallback": False,
        "ref_score": reference_values["score"],
        "ref_root_pos_error_mean": reference_values["root_pos_error_mean"],
        "ref_body_global_pos_error_mean": reference_values[
            "body_global_pos_error_mean"
        ],
        "ref_ee_global_pos_error_mean": reference_values[
            "ee_global_pos_error_mean"
        ],
    }


def _expected_pareto_keys() -> set[str]:
    return {
        "pareto_class",
        "failure_labels",
        "repeat_count",
        "ok_count",
        "success_count",
        "accepted_count",
        "fallback_count",
        "full_run_count",
        "accepted_windows_count",
        "duration_sec_mean",
        "score_mean",
        "root_pos_error_mean_mean",
        "body_global_pos_error_mean_mean",
        "ee_global_pos_error_mean_mean",
        "ee_local_pos_error_mean_mean",
        "contact_mismatch_rate_mean",
        "control_delta_mean_mean",
        "joint_acc_mean_mean",
        "score_delta_vs_best_mean",
        "score_delta_vs_baseline_mean",
        "max_global_ratio_vs_best_mean",
        "max_global_ratio_vs_baseline_mean",
    }


def _completed_process(returncode: int):
    return type("CompletedProcess", (), {"returncode": returncode})()


def _load_runner():
    if not RUNNER_PATH.exists():
        raise AssertionError(f"missing runner script: {RUNNER_PATH}")
    module_name = "_g1_wbc_local_first_stage_runner_test"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load runner script: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _flag_value(command: list[str], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError as exc:
        raise AssertionError(f"missing flag {flag!r} in command {command!r}") from exc
    try:
        return command[index + 1]
    except IndexError as exc:
        raise AssertionError(f"missing value for flag {flag!r}") from exc


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
