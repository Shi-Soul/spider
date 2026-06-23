from __future__ import annotations

import math
import unittest

from spider.tasks.g1_wbc import local_first_stage as stage


class LocalTransferCandidateTest(unittest.TestCase):
    def test_transfer_candidate_names_are_exact_order(self) -> None:
        self.assertEqual(
            stage.TRANSFER_CANDIDATE_NAMES,
            (
                "jg_s128_v14_control",
                "jg_s128_L148_posture",
                "jg_s128_L142_smooth005",
                "jg_s128_L145_smooth005",
            ),
        )

    def test_recovery_candidate_names_are_exact_order(self) -> None:
        self.assertEqual(
            stage.RECOVERY_CANDIDATE_NAMES,
            (
                "jg_s128_v14_control",
                "jg_s128_L148_posture",
                "jg_s128_L148_smooth010",
                "jg_s128_L148_contact010",
                "jg_s128_L148_smooth010_contact010",
                "jg_s128_L142_smooth005_contact010",
                "jg_s128_L145_smooth010_contact010",
            ),
        )

    def test_transfer_reward_weights_include_recovery_matrix(self) -> None:
        smooth010 = {
            "control_delta": 0.99,
            "action_delta": 0.33,
            "joint_acc": 0.0033,
            "joint_jerk": 0.00088,
        }
        contact010 = {
            "contact_switch": 6.6,
            "contact_force_delta": 1.1,
            "contact_false_positive": 0.66,
            "contact_false_negative": 0.275,
        }
        posture148 = stage.CANDIDATE_DEFINITIONS["jg_s128_L148_posture"].overrides
        posture142 = stage.CANDIDATE_DEFINITIONS["jg_s128_L142_posture"].overrides
        posture145 = stage.CANDIDATE_DEFINITIONS["jg_s128_L145_posture"].overrides
        smooth005 = {
            "control_delta": 0.945,
            "action_delta": 0.315,
            "joint_acc": 0.00315,
            "joint_jerk": 0.00084,
        }
        expected = {
            "jg_s128_L148_smooth010": {**posture148, **smooth010},
            "jg_s128_L148_contact010": {**posture148, **contact010},
            "jg_s128_L148_smooth010_contact010": {
                **posture148,
                **smooth010,
                **contact010,
            },
            "jg_s128_L142_smooth005_contact010": {
                **posture142,
                **smooth005,
                **contact010,
            },
            "jg_s128_L145_smooth010_contact010": {
                **posture145,
                **smooth010,
                **contact010,
            },
        }

        for candidate_name, overrides in expected.items():
            with self.subTest(candidate_name=candidate_name):
                weights = stage.candidate_reward_weights(
                    {stage.JOINT_GLOBAL_METHOD: {}},
                    candidate_name,
                )

                self.assertEqual(weights[stage.JOINT_GLOBAL_METHOD], overrides)


class LocalTransferAssessmentTest(unittest.TestCase):
    def test_transfer_assessment_passes_walk_with_one_local_improvement(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["walk"]
        metrics = _candidate_metrics(
            "walk",
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.96,
        )

        result = stage.assess_transfer_candidate(
            "walk",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["transfer_class"], "transfer_pass")
        self.assertEqual(result["failure_labels"], [])
        self.assertTrue(result["local_transfer_passed"])
        self.assertEqual(result["local_count_3"], 1)
        self.assertLessEqual(result["max_local_regression"], 0.02)
        self.assertTrue(_expected_transfer_keys() <= set(result))

    def test_transfer_assessment_marks_contact_failure_as_recovery_candidate(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["qixing"]
        metrics = _candidate_metrics(
            "qixing",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.96,
            contact_mismatch_rate=stage.QIXING_CONTACT_TARGET + 0.001,
        )

        result = stage.assess_transfer_candidate(
            "qixing",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["transfer_class"], "recovery_candidate")
        self.assertFalse(result["contact_guardrail_passed"])
        self.assertTrue(result["score_guardrail_passed"])
        self.assertTrue(result["global_guardrail_passed"])
        self.assertTrue(result["mpc_guardrail_passed"])
        self.assertTrue(result["runtime_guardrail_passed"])
        self.assertIn("contact_guardrail", result["failure_labels"])

    def test_transfer_assessment_marks_global_failure_as_invalid(self) -> None:
        baseline = stage.BEST_S128_BASELINES["walk"]
        metrics = _candidate_metrics(
            "walk",
            root_pos_error_mean=baseline["root_pos_error_mean"] * 1.06,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.96,
        )

        result = stage.assess_transfer_candidate(
            "walk",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["transfer_class"], "invalid")
        self.assertFalse(result["global_guardrail_passed"])
        self.assertIn("global_guardrail", result["failure_labels"])

    def test_transfer_assessment_requires_two_local_improvements_for_jump(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.96,
        )

        result = stage.assess_transfer_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["transfer_class"], "recovery_candidate")
        self.assertEqual(result["local_count_3"], 1)
        self.assertFalse(result["local_transfer_passed"])
        self.assertIn("local_transfer_guardrail", result["failure_labels"])

    def test_transfer_assessment_marks_non_finite_required_metric_as_invalid(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["walk"]
        metrics = _candidate_metrics(
            "walk",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.96,
            ee_local_pos_error_mean=math.nan,
        )

        result = stage.assess_transfer_candidate(
            "walk",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["transfer_class"], "invalid")
        self.assertIn("required_metrics", result["failure_labels"])
        self.assertFalse(result["local_transfer_passed"])

    def test_transfer_assessment_marks_missing_mpc_data_as_invalid(self) -> None:
        baseline = stage.BEST_S128_BASELINES["walk"]
        metrics = _candidate_metrics(
            "walk",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.96,
        )
        mpc = _accepted_mpc()
        del mpc["used_baseline_fallback"]

        result = stage.assess_transfer_candidate(
            "walk",
            metrics,
            mpc,
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["transfer_class"], "invalid")
        self.assertIn("required_mpc_data", result["failure_labels"])
        self.assertIn("mpc_guardrail", result["failure_labels"])

    def test_transfer_assessment_rejects_malformed_baseline_fallback(self) -> None:
        baseline = stage.BEST_S128_BASELINES["walk"]
        metrics = _candidate_metrics(
            "walk",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.96,
        )
        mpc = _accepted_mpc()
        mpc["used_baseline_fallback"] = "unexpected"

        result = stage.assess_transfer_candidate(
            "walk",
            metrics,
            mpc,
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["transfer_class"], "invalid")
        self.assertFalse(result["mpc_guardrail_passed"])
        self.assertIn("mpc_guardrail", result["failure_labels"])

    def test_transfer_assessment_rejects_non_canonical_mpc_boolean_aliases(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["walk"]
        metrics = _candidate_metrics(
            "walk",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.96,
        )

        for value in ("1", "0", "yes", "y", "no", "n", 1, 0):
            with self.subTest(value=value):
                mpc = _accepted_mpc()
                mpc["used_baseline_fallback"] = value

                result = stage.assess_transfer_candidate(
                    "walk",
                    metrics,
                    mpc,
                    duration_sec=100.0,
                    control_duration_sec=100.0,
                )

                self.assertEqual(result["transfer_class"], "invalid")
                self.assertFalse(result["mpc_guardrail_passed"])
                self.assertIn("mpc_guardrail", result["failure_labels"])

    def test_transfer_assessment_rejects_non_positive_runtime(self) -> None:
        baseline = stage.BEST_S128_BASELINES["walk"]
        metrics = _candidate_metrics(
            "walk",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.96,
        )

        result = stage.assess_transfer_candidate(
            "walk",
            metrics,
            _accepted_mpc(),
            duration_sec=-1.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["transfer_class"], "invalid")
        self.assertFalse(result["runtime_guardrail_passed"])
        self.assertIn("runtime_guardrail", result["failure_labels"])

    def test_upper_bound_assessment_marks_non_finite_required_metric_as_invalid(
        self,
    ) -> None:
        baseline = stage.BEST_S128_BASELINES["jump"]
        metrics = _candidate_metrics(
            "jump",
            joint_pos_error_mean=baseline["joint_pos_error_mean"] * 0.90,
            body_local_pos_error_mean=baseline["body_local_pos_error_mean"] * 0.90,
            ee_local_pos_error_mean=math.inf,
        )

        result = stage.assess_upper_bound_candidate(
            "jump",
            metrics,
            _accepted_mpc(),
            duration_sec=100.0,
            control_duration_sec=100.0,
        )

        self.assertEqual(result["frontier_class"], "invalid")
        self.assertIn("required_metrics", result["failure_labels"])


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


def _expected_transfer_keys() -> set[str]:
    return {
        "transfer_class",
        "failure_labels",
        "score_guardrail_passed",
        "global_guardrail_passed",
        "smooth_guardrail_passed",
        "contact_guardrail_passed",
        "local_transfer_passed",
        "mpc_guardrail_passed",
        "runtime_guardrail_passed",
        "local_count_3",
        "max_local_regression",
        "score_delta",
        "max_global_ratio",
        "contact_delta",
        "control_delta_regression",
        "joint_acc_regression",
    }
