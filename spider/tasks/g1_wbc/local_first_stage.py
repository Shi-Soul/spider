"""Pure helpers for the G1 WBC local-first low-sample stage."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

JOINT_GLOBAL_METHOD = "g1_wbc_joint_global"

SCORE_TOLERANCE = 0.05
GLOBAL_REGRESSION_MULTIPLIER = 1.05
UPPER_BOUND_SCORE_TOLERANCE = 0.25
UPPER_BOUND_GLOBAL_REGRESSION_MULTIPLIER = 1.35
STRICT_LOCAL_IMPROVEMENT_MULTIPLIER = 0.95
MIDDLE_GATE_LOCAL_IMPROVEMENT_MULTIPLIER = 0.97
LOCAL_IMPROVEMENT_MULTIPLIER = STRICT_LOCAL_IMPROVEMENT_MULTIPLIER
TRANSFER_LOCAL_IMPROVEMENT_MULTIPLIER = 0.97
TRANSFER_LOCAL_REGRESSION_MULTIPLIER = 1.02
JOINT_ACC_REGRESSION_MULTIPLIER = 1.02
RUNTIME_REGRESSION_MULTIPLIER = 1.10
FULL_RUN_STEPS = 800.0
FULL_RUN_ACCEPTED_WINDOWS = 40
QIXING_CONTACT_TARGET = 0.220
PARETO_SAMPLE_COUNTS = (64, 96, 128, 192, 256)
PARETO_REPEAT_SEEDS = (0, 1, 2)
PARETO_BASELINE_SCORE_TOLERANCE = 0.25
PARETO_BASELINE_GLOBAL_RATIO_LIMIT = 1.35

GLOBAL_GUARDRAIL_METRICS = (
    "root_pos_error_mean",
    "body_global_pos_error_mean",
    "ee_global_pos_error_mean",
)
LOCAL_IMPROVEMENT_METRICS = (
    "joint_pos_error_mean",
    "body_local_pos_error_mean",
    "ee_local_pos_error_mean",
)
ASSESSMENT_REQUIRED_METRICS = (
    "score",
    "num_steps",
    *GLOBAL_GUARDRAIL_METRICS,
    *LOCAL_IMPROVEMENT_METRICS,
    "contact_mismatch_rate",
    "control_delta_mean",
    "joint_acc_mean",
)
ASSESSMENT_REQUIRED_MPC = ("accepted", "accepted_windows", "used_baseline_fallback")
PARETO_PRIMARY_METRICS = (
    "score",
    "root_pos_error_mean",
    "body_global_pos_error_mean",
    "ee_global_pos_error_mean",
    "ee_local_pos_error_mean",
    "contact_mismatch_rate",
    "control_delta_mean",
    "joint_acc_mean",
)


@dataclass(frozen=True)
class CandidateDefinition:
    """Reward-weight overrides for one local-first candidate."""

    name: str
    overrides: Mapping[str, float]


def _frozen_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(values))


def _candidate(name: str, overrides: Mapping[str, float]) -> CandidateDefinition:
    return CandidateDefinition(name=name, overrides=_frozen_mapping(overrides))


_L150_POSTURE_OVERRIDES = {
    "body_local_pos_error": 7.5,
    "body_local_rot_error": 1.25,
    "ee_local_pos_error": 6.0,
    "ee_local_rot_error": 0.875,
    "hand_local_pos_error": 4.5,
    "joint_pos_error": 0.40,
}

_SMOOTH015_OVERRIDES = {
    "control_delta": 1.035,
    "action_delta": 0.345,
    "joint_acc": 0.00345,
    "joint_jerk": 0.00092,
}

_SMOOTH005_OVERRIDES = {
    "control_delta": 0.945,
    "action_delta": 0.315,
    "joint_acc": 0.00315,
    "joint_jerk": 0.00084,
}

_SMOOTH010_OVERRIDES = {
    "control_delta": 0.99,
    "action_delta": 0.33,
    "joint_acc": 0.0033,
    "joint_jerk": 0.00088,
}

_SMOOTH025_OVERRIDES = {
    "control_delta": 1.125,
    "action_delta": 0.375,
    "joint_acc": 0.00375,
    "joint_jerk": 0.001,
}

_CONTACT015_OVERRIDES = {
    "contact_switch": 6.9,
    "contact_force_delta": 1.15,
    "contact_false_positive": 0.69,
    "contact_false_negative": 0.2875,
}

_CONTACT010_OVERRIDES = {
    "contact_switch": 6.6,
    "contact_force_delta": 1.1,
    "contact_false_positive": 0.66,
    "contact_false_negative": 0.275,
}

_L135_POSTURE_OVERRIDES = {
    "body_local_pos_error": 6.75,
    "body_local_rot_error": 1.19,
    "ee_local_pos_error": 5.4,
    "ee_local_rot_error": 0.833,
    "hand_local_pos_error": 4.05,
    "joint_pos_error": 0.31,
}

_L138_POSTURE_OVERRIDES = {
    "body_local_pos_error": 6.9,
    "body_local_rot_error": 1.202,
    "ee_local_pos_error": 5.52,
    "ee_local_rot_error": 0.8414,
    "hand_local_pos_error": 4.14,
    "joint_pos_error": 0.328,
}

_L140_POSTURE_OVERRIDES = {
    "body_local_pos_error": 7.0,
    "body_local_rot_error": 1.21,
    "ee_local_pos_error": 5.6,
    "ee_local_rot_error": 0.847,
    "hand_local_pos_error": 4.2,
    "joint_pos_error": 0.34,
}

_L142_POSTURE_OVERRIDES = {
    "body_local_pos_error": 7.1,
    "body_local_rot_error": 1.218,
    "ee_local_pos_error": 5.68,
    "ee_local_rot_error": 0.8526,
    "hand_local_pos_error": 4.26,
    "joint_pos_error": 0.352,
}

_L145_POSTURE_OVERRIDES = {
    "body_local_pos_error": 7.25,
    "body_local_rot_error": 1.23,
    "ee_local_pos_error": 5.8,
    "ee_local_rot_error": 0.861,
    "hand_local_pos_error": 4.35,
    "joint_pos_error": 0.37,
}

_L148_POSTURE_OVERRIDES = {
    "body_local_pos_error": 7.4,
    "body_local_rot_error": 1.242,
    "ee_local_pos_error": 5.92,
    "ee_local_rot_error": 0.8694,
    "hand_local_pos_error": 4.44,
    "joint_pos_error": 0.388,
}

_L155_POSTURE_OVERRIDES = {
    "body_local_pos_error": 7.75,
    "body_local_rot_error": 1.27,
    "ee_local_pos_error": 6.2,
    "ee_local_rot_error": 0.889,
    "hand_local_pos_error": 4.65,
    "joint_pos_error": 0.43,
}

CANDIDATE_DEFINITIONS: Mapping[str, CandidateDefinition] = MappingProxyType(
    {
        "jg_s128_v14_control": _candidate("jg_s128_v14_control", {}),
        "jg_s128_L125_posture": _candidate(
            "jg_s128_L125_posture",
            {
                "body_local_pos_error": 6.25,
                "body_local_rot_error": 1.15,
                "ee_local_pos_error": 5.0,
                "ee_local_rot_error": 0.805,
                "hand_local_pos_error": 3.75,
                "joint_pos_error": 0.25,
            },
        ),
        "jg_s128_L150_posture": _candidate(
            "jg_s128_L150_posture",
            _L150_POSTURE_OVERRIDES,
        ),
        "jg_s128_L150_smooth": _candidate(
            "jg_s128_L150_smooth",
            {
                **_L150_POSTURE_OVERRIDES,
                "control_delta": 1.08,
                "action_delta": 0.375,
                "joint_acc": 0.0045,
                "joint_jerk": 0.0012,
            },
        ),
        "jg_s128_L150_contact": _candidate(
            "jg_s128_L150_contact",
            {
                **_L150_POSTURE_OVERRIDES,
                "contact_switch": 7.5,
                "contact_force_delta": 1.25,
                "contact_false_positive": 0.72,
                "contact_false_negative": 0.375,
            },
        ),
        "jg_s128_L150_global090": _candidate(
            "jg_s128_L150_global090",
            {
                **_L150_POSTURE_OVERRIDES,
                "body_global_pos_error": 52.2,
                "body_global_rot_error": 6.48,
                "ee_global_pos_error": 5.4,
                "ee_global_rot_error": 0.99,
                "hand_global_pos_error": 4.5,
            },
        ),
        "jg_s128_v14_smooth015": _candidate(
            "jg_s128_v14_smooth015",
            _SMOOTH015_OVERRIDES,
        ),
        "jg_s128_v14_smooth025": _candidate(
            "jg_s128_v14_smooth025",
            _SMOOTH025_OVERRIDES,
        ),
        "jg_s128_v14_contact015": _candidate(
            "jg_s128_v14_contact015",
            _CONTACT015_OVERRIDES,
        ),
        "jg_s128_v14_smooth015_contact015": _candidate(
            "jg_s128_v14_smooth015_contact015",
            {
                **_SMOOTH015_OVERRIDES,
                **_CONTACT015_OVERRIDES,
            },
        ),
        "jg_s128_L135_posture": _candidate(
            "jg_s128_L135_posture",
            _L135_POSTURE_OVERRIDES,
        ),
        "jg_s128_L140_posture": _candidate(
            "jg_s128_L140_posture",
            _L140_POSTURE_OVERRIDES,
        ),
        "jg_s128_L145_posture": _candidate(
            "jg_s128_L145_posture",
            _L145_POSTURE_OVERRIDES,
        ),
        "jg_s128_L140_smooth015": _candidate(
            "jg_s128_L140_smooth015",
            {
                **_L140_POSTURE_OVERRIDES,
                **_SMOOTH015_OVERRIDES,
            },
        ),
        "jg_s128_L140_contact015": _candidate(
            "jg_s128_L140_contact015",
            {
                **_L140_POSTURE_OVERRIDES,
                **_CONTACT015_OVERRIDES,
            },
        ),
        "jg_s128_L145_smooth015_contact015": _candidate(
            "jg_s128_L145_smooth015_contact015",
            {
                **_L145_POSTURE_OVERRIDES,
                **_SMOOTH015_OVERRIDES,
                **_CONTACT015_OVERRIDES,
            },
        ),
        "jg_s128_L138_posture": _candidate(
            "jg_s128_L138_posture",
            _L138_POSTURE_OVERRIDES,
        ),
        "jg_s128_L142_posture": _candidate(
            "jg_s128_L142_posture",
            _L142_POSTURE_OVERRIDES,
        ),
        "jg_s128_L148_posture": _candidate(
            "jg_s128_L148_posture",
            _L148_POSTURE_OVERRIDES,
        ),
        "jg_s128_L148_smooth010": _candidate(
            "jg_s128_L148_smooth010",
            {
                **_L148_POSTURE_OVERRIDES,
                **_SMOOTH010_OVERRIDES,
            },
        ),
        "jg_s128_L148_contact010": _candidate(
            "jg_s128_L148_contact010",
            {
                **_L148_POSTURE_OVERRIDES,
                **_CONTACT010_OVERRIDES,
            },
        ),
        "jg_s128_L148_smooth010_contact010": _candidate(
            "jg_s128_L148_smooth010_contact010",
            {
                **_L148_POSTURE_OVERRIDES,
                **_SMOOTH010_OVERRIDES,
                **_CONTACT010_OVERRIDES,
            },
        ),
        "jg_s128_L155_posture": _candidate(
            "jg_s128_L155_posture",
            _L155_POSTURE_OVERRIDES,
        ),
        "jg_s128_L142_smooth005": _candidate(
            "jg_s128_L142_smooth005",
            {
                **_L142_POSTURE_OVERRIDES,
                **_SMOOTH005_OVERRIDES,
            },
        ),
        "jg_s128_L142_smooth005_contact010": _candidate(
            "jg_s128_L142_smooth005_contact010",
            {
                **_L142_POSTURE_OVERRIDES,
                **_SMOOTH005_OVERRIDES,
                **_CONTACT010_OVERRIDES,
            },
        ),
        "jg_s128_L145_smooth005": _candidate(
            "jg_s128_L145_smooth005",
            {
                **_L145_POSTURE_OVERRIDES,
                **_SMOOTH005_OVERRIDES,
            },
        ),
        "jg_s128_L145_smooth010_contact010": _candidate(
            "jg_s128_L145_smooth010_contact010",
            {
                **_L145_POSTURE_OVERRIDES,
                **_SMOOTH010_OVERRIDES,
                **_CONTACT010_OVERRIDES,
            },
        ),
        "jg_s128_L148_smooth005": _candidate(
            "jg_s128_L148_smooth005",
            {
                **_L148_POSTURE_OVERRIDES,
                **_SMOOTH005_OVERRIDES,
            },
        ),
        "jg_s128_L150_smooth005": _candidate(
            "jg_s128_L150_smooth005",
            {
                **_L150_POSTURE_OVERRIDES,
                **_SMOOTH005_OVERRIDES,
            },
        ),
    }
)
CANDIDATE_NAMES = tuple(CANDIDATE_DEFINITIONS)
NEXT_STAGE_CANDIDATE_NAMES = (
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
)
UPPER_BOUND_CANDIDATE_NAMES = (
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
)
TRANSFER_CANDIDATE_NAMES = (
    "jg_s128_v14_control",
    "jg_s128_L148_posture",
    "jg_s128_L142_smooth005",
    "jg_s128_L145_smooth005",
)
RECOVERY_CANDIDATE_NAMES = (
    "jg_s128_v14_control",
    "jg_s128_L148_posture",
    "jg_s128_L148_smooth010",
    "jg_s128_L148_contact010",
    "jg_s128_L148_smooth010_contact010",
    "jg_s128_L142_smooth005_contact010",
    "jg_s128_L145_smooth010_contact010",
)
PARETO_CANDIDATE_NAMES = ("jg_s128_v14_control",)


BEST_S128_BASELINES: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "jump": MappingProxyType(
            {
                "score": -2.2657276578247547,
                "num_steps": 800.0,
                "root_pos_error_mean": 0.08005901426076889,
                "body_global_pos_error_mean": 0.087236188352108,
                "ee_global_pos_error_mean": 0.0935337096452713,
                "joint_pos_error_mean": 0.6668357849121094,
                "body_local_pos_error_mean": 0.02871049754321575,
                "ee_local_pos_error_mean": 0.04581649228930473,
                "contact_mismatch_rate": 0.35374999046325684,
                "control_delta_mean": 0.4710128903388977,
                "joint_acc_mean": 223.22132873535156,
            }
        ),
        "walk": MappingProxyType(
            {
                "score": -1.267110737413168,
                "num_steps": 800.0,
                "root_pos_error_mean": 0.07323325425386429,
                "body_global_pos_error_mean": 0.077600859105587,
                "ee_global_pos_error_mean": 0.07914591580629349,
                "joint_pos_error_mean": 0.4450763761997223,
                "body_local_pos_error_mean": 0.02322342060506344,
                "ee_local_pos_error_mean": 0.035528190433979034,
                "contact_mismatch_rate": 0.14999999105930328,
                "control_delta_mean": 0.24414201080799103,
                "joint_acc_mean": 117.78897094726562,
            }
        ),
        "qixing": MappingProxyType(
            {
                "score": -1.5680181883275508,
                "num_steps": 800.0,
                "root_pos_error_mean": 0.06380528211593628,
                "body_global_pos_error_mean": 0.06895875185728073,
                "ee_global_pos_error_mean": 0.07506847381591797,
                "joint_pos_error_mean": 0.5439301133155823,
                "body_local_pos_error_mean": 0.02858331985771656,
                "ee_local_pos_error_mean": 0.04572489857673645,
                "contact_mismatch_rate": 0.22749999165534973,
                "control_delta_mean": 0.2371356189250946,
                "joint_acc_mean": 88.93841552734375,
            }
        ),
    }
)


def candidate_reward_weights(
    base_payload: Mapping[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    """Return reward weights for a candidate without mutating the base payload."""

    candidate = _get_candidate(candidate_name)
    if JOINT_GLOBAL_METHOD not in base_payload:
        raise KeyError(f"Missing {JOINT_GLOBAL_METHOD!r} reward weights.")

    payload = copy.deepcopy(dict(base_payload))
    target = payload[JOINT_GLOBAL_METHOD]
    if not isinstance(target, MutableMapping):
        raise TypeError(f"{JOINT_GLOBAL_METHOD!r} reward weights must be a mapping.")

    target.update(candidate.overrides)
    return payload


def write_candidate_reward_files(
    base_path: str | Path,
    output_dir: str | Path,
    candidate_names: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write one reward-weight JSON file per requested candidate."""

    base_path = Path(base_path)
    output_dir = Path(output_dir)
    base_payload = json.loads(base_path.read_text())
    names = tuple(candidate_names) if candidate_names is not None else CANDIDATE_NAMES

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for candidate_name in names:
        payload = candidate_reward_weights(base_payload, candidate_name)
        path = output_dir / f"{candidate_name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        paths[candidate_name] = path
    return paths


def assess_candidate(
    motion: str,
    metrics: Mapping[str, Any],
    mpc: Mapping[str, Any],
    duration_sec: float | None = None,
    control_duration_sec: float | None = None,
    *,
    local_improvement_multiplier: float = STRICT_LOCAL_IMPROVEMENT_MULTIPLIER,
) -> dict[str, Any]:
    """Assess a candidate against the local-first promotion guardrails."""

    baseline = _get_baseline(motion)
    score_guardrail_passed = _at_least(
        metrics,
        "score",
        baseline["score"] - SCORE_TOLERANCE,
    )
    global_guardrail_passed = all(
        _at_most(metrics, metric, baseline[metric] * GLOBAL_REGRESSION_MULTIPLIER)
        for metric in GLOBAL_GUARDRAIL_METRICS
    )
    required_metrics_passed = not _missing_required_metrics(metrics)
    required_mpc_passed = not _missing_required_mpc(mpc)
    smooth_guardrail_passed = _at_most(
        metrics,
        "control_delta_mean",
        baseline["control_delta_mean"],
    ) and _at_most(
        metrics,
        "joint_acc_mean",
        baseline["joint_acc_mean"] * JOINT_ACC_REGRESSION_MULTIPLIER,
    )
    contact_guardrail_passed = _at_most(
        metrics,
        "contact_mismatch_rate",
        _contact_limit(motion, baseline),
    )
    improved_local_count = sum(
        int(
            _local_threshold_improved(
                metrics,
                baseline,
                metric,
                local_improvement_multiplier,
            )
        )
        for metric in LOCAL_IMPROVEMENT_METRICS
    )
    local_guardrail_passed = motion != "jump" or improved_local_count >= 2
    mpc_guardrail_passed = _mpc_guardrail_passed(metrics, mpc)
    runtime_guardrail_passed = _runtime_guardrail_passed(
        duration_sec,
        control_duration_sec,
    )

    checks = (
        ("required_metrics", required_metrics_passed),
        ("required_mpc_data", required_mpc_passed),
        ("score_guardrail", score_guardrail_passed),
        ("global_guardrail", global_guardrail_passed),
        ("smooth_guardrail", smooth_guardrail_passed),
        ("contact_guardrail", contact_guardrail_passed),
        ("local_improvement_guardrail", local_guardrail_passed),
        ("mpc_guardrail", mpc_guardrail_passed),
        ("runtime_guardrail", runtime_guardrail_passed),
    )
    failure_labels = [label for label, passed in checks if not passed]

    return {
        "passed": not failure_labels,
        "failure_labels": failure_labels,
        "improved_local_count": improved_local_count,
        "required_metrics_passed": required_metrics_passed,
        "required_mpc_passed": required_mpc_passed,
        "global_guardrail_passed": global_guardrail_passed,
        "smooth_guardrail_passed": smooth_guardrail_passed,
        "contact_guardrail_passed": contact_guardrail_passed,
        "score_guardrail_passed": score_guardrail_passed,
        "local_guardrail_passed": local_guardrail_passed,
        "mpc_guardrail_passed": mpc_guardrail_passed,
        "runtime_guardrail_passed": runtime_guardrail_passed,
    }


def assess_transfer_candidate(
    motion: str,
    metrics: Mapping[str, Any],
    mpc: Mapping[str, Any],
    duration_sec: float | None = None,
    control_duration_sec: float | None = None,
) -> dict[str, Any]:
    """Classify a transfer/recovery candidate on non-local-frontier motions."""

    baseline = _get_baseline(motion)
    score = _as_float(metrics.get("score"))
    score_delta = score - baseline["score"] if score is not None else float("-inf")
    score_guardrail_passed = _at_least(
        metrics,
        "score",
        baseline["score"] - SCORE_TOLERANCE,
    )
    max_global_ratio = max(
        _metric_ratio(metrics, metric, baseline[metric])
        for metric in GLOBAL_GUARDRAIL_METRICS
    )
    global_guardrail_passed = max_global_ratio <= GLOBAL_REGRESSION_MULTIPLIER
    required_metrics_passed = not _missing_required_metrics(metrics)
    required_mpc_passed = not _missing_required_mpc(mpc)
    smooth_guardrail_passed = _at_most(
        metrics,
        "control_delta_mean",
        baseline["control_delta_mean"],
    ) and _at_most(
        metrics,
        "joint_acc_mean",
        baseline["joint_acc_mean"] * JOINT_ACC_REGRESSION_MULTIPLIER,
    )
    contact_guardrail_passed = _at_most(
        metrics,
        "contact_mismatch_rate",
        _contact_limit(motion, baseline),
    )
    local_count_3 = sum(
        int(
            _local_threshold_improved(
                metrics,
                baseline,
                metric,
                TRANSFER_LOCAL_IMPROVEMENT_MULTIPLIER,
            )
        )
        for metric in LOCAL_IMPROVEMENT_METRICS
    )
    local_regressions = [
        _metric_ratio(metrics, metric, baseline[metric]) - 1.0
        for metric in LOCAL_IMPROVEMENT_METRICS
    ]
    max_local_regression = max(local_regressions)
    required_local_count = 2 if motion == "jump" else 1
    local_transfer_passed = (
        local_count_3 >= required_local_count
        and max_local_regression <= TRANSFER_LOCAL_REGRESSION_MULTIPLIER - 1.0
    )
    mpc_guardrail_passed = _mpc_guardrail_passed(metrics, mpc)
    runtime_guardrail_passed = _runtime_guardrail_passed(
        duration_sec,
        control_duration_sec,
    )

    checks = (
        ("required_metrics", required_metrics_passed),
        ("required_mpc_data", required_mpc_passed),
        ("score_guardrail", score_guardrail_passed),
        ("global_guardrail", global_guardrail_passed),
        ("smooth_guardrail", smooth_guardrail_passed),
        ("contact_guardrail", contact_guardrail_passed),
        ("local_transfer_guardrail", local_transfer_passed),
        ("mpc_guardrail", mpc_guardrail_passed),
        ("runtime_guardrail", runtime_guardrail_passed),
    )
    failure_labels = [label for label, passed in checks if not passed]
    hard_failures = {
        "required_metrics",
        "required_mpc_data",
        "score_guardrail",
        "global_guardrail",
        "mpc_guardrail",
        "runtime_guardrail",
    }
    if hard_failures.intersection(failure_labels):
        transfer_class = "invalid"
    elif (
        smooth_guardrail_passed
        and contact_guardrail_passed
        and local_transfer_passed
    ):
        transfer_class = "transfer_pass"
    else:
        transfer_class = "recovery_candidate"

    return {
        "transfer_class": transfer_class,
        "failure_labels": failure_labels,
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
        "contact_delta": _metric_delta(
            metrics,
            "contact_mismatch_rate",
            baseline["contact_mismatch_rate"],
        ),
        "control_delta_regression": _metric_ratio(
            metrics,
            "control_delta_mean",
            baseline["control_delta_mean"],
        )
        - 1.0,
        "joint_acc_regression": _metric_ratio(
            metrics,
            "joint_acc_mean",
            baseline["joint_acc_mean"],
        )
        - 1.0,
    }


def assess_upper_bound_candidate(
    motion: str,
    metrics: Mapping[str, Any],
    mpc: Mapping[str, Any],
    duration_sec: float | None = None,
    control_duration_sec: float | None = None,
) -> dict[str, Any]:
    """Classify a relaxed local/posture upper-bound candidate."""

    baseline = _get_baseline(motion)
    score = _as_float(metrics.get("score"))
    score_delta = score - baseline["score"] if score is not None else float("-inf")
    score_relaxed_passed = _at_least(
        metrics,
        "score",
        baseline["score"] - UPPER_BOUND_SCORE_TOLERANCE,
    )
    max_global_ratio = max(
        _metric_ratio(metrics, metric, baseline[metric])
        for metric in GLOBAL_GUARDRAIL_METRICS
    )
    global_relaxed_passed = max_global_ratio <= UPPER_BOUND_GLOBAL_REGRESSION_MULTIPLIER
    required_metrics_passed = not _missing_required_metrics(metrics)
    required_mpc_passed = not _missing_required_mpc(mpc)
    mpc_guardrail_passed = _mpc_guardrail_passed(metrics, mpc)
    runtime_guardrail_passed = _runtime_guardrail_passed(
        duration_sec,
        control_duration_sec,
    )
    hard_guardrail_passed = mpc_guardrail_passed and runtime_guardrail_passed

    local_improvements = [
        _local_improvement(metrics, metric, baseline[metric])
        for metric in LOCAL_IMPROVEMENT_METRICS
    ]
    local_count_3 = sum(
        int(_local_threshold_improved(metrics, baseline, metric, 0.97))
        for metric in LOCAL_IMPROVEMENT_METRICS
    )
    local_count_5 = sum(
        int(_local_threshold_improved(metrics, baseline, metric, 0.95))
        for metric in LOCAL_IMPROVEMENT_METRICS
    )
    local_count_8 = sum(
        int(_local_threshold_improved(metrics, baseline, metric, 0.92))
        for metric in LOCAL_IMPROVEMENT_METRICS
    )
    local_composite_improvement = sum(local_improvements) / len(local_improvements)

    checks = (
        ("required_metrics", required_metrics_passed),
        ("required_mpc_data", required_mpc_passed),
        ("relaxed_score_guardrail", score_relaxed_passed),
        ("relaxed_global_guardrail", global_relaxed_passed),
        ("mpc_guardrail", mpc_guardrail_passed),
        ("runtime_guardrail", runtime_guardrail_passed),
    )
    failure_labels = [label for label, passed in checks if not passed]
    if failure_labels:
        frontier_class = "invalid"
    elif local_count_5 >= 2 or local_count_3 == 3:
        frontier_class = "local_frontier"
    elif local_count_3 >= 2:
        frontier_class = "near_frontier"
    else:
        frontier_class = "dead_end"

    return {
        "frontier_class": frontier_class,
        "failure_labels": failure_labels,
        "hard_guardrail_passed": hard_guardrail_passed,
        "score_relaxed_passed": score_relaxed_passed,
        "global_relaxed_passed": global_relaxed_passed,
        "mpc_guardrail_passed": mpc_guardrail_passed,
        "runtime_guardrail_passed": runtime_guardrail_passed,
        "local_count_3": local_count_3,
        "local_count_5": local_count_5,
        "local_count_8": local_count_8,
        "local_composite_improvement": local_composite_improvement,
        "score_delta": score_delta,
        "max_global_ratio": max_global_ratio,
        "contact_delta": _metric_delta(
            metrics,
            "contact_mismatch_rate",
            baseline["contact_mismatch_rate"],
        ),
        "control_delta_regression": _metric_ratio(
            metrics,
            "control_delta_mean",
            baseline["control_delta_mean"],
        )
        - 1.0,
        "joint_acc_regression": _metric_ratio(
            metrics,
            "joint_acc_mean",
            baseline["joint_acc_mean"],
        )
        - 1.0,
    }


def assess_pareto_group(
    motion: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify repeated runs for one motion/sample budget."""

    if not rows:
        return _empty_pareto_result("invalid", ["no_repeats"])

    baseline = _get_baseline(motion)
    repeat_count = len(rows)
    ok_count = sum(int(row.get("status") == "ok") for row in rows)
    success_count = sum(
        int(_as_bool(row.get("metric_success")) is True) for row in rows
    )
    accepted_count = sum(
        int(_as_bool(row.get("mpc_accepted")) is True) for row in rows
    )
    fallback_count = sum(
        int(_as_bool(row.get("mpc_used_baseline_fallback")) is True)
        for row in rows
    )
    full_run_count = sum(
        int(_as_float(row.get("metric_num_steps")) == FULL_RUN_STEPS)
        for row in rows
    )
    accepted_windows_count = sum(
        int(_as_float(row.get("mpc_accepted_windows")) == FULL_RUN_ACCEPTED_WINDOWS)
        for row in rows
    )
    fallback_clean_count = sum(
        int(_as_bool(row.get("mpc_used_baseline_fallback")) is False)
        for row in rows
    )

    failure_labels: list[str] = []
    if ok_count != repeat_count:
        failure_labels.append("status")
    if success_count != repeat_count:
        failure_labels.append("metric_success")
    if full_run_count != repeat_count:
        failure_labels.append("full_run")
    if accepted_count != repeat_count:
        failure_labels.append("mpc_accepted")
    if accepted_windows_count != repeat_count:
        failure_labels.append("accepted_windows")
    if fallback_clean_count != repeat_count:
        failure_labels.append("baseline_fallback")
    if not _pareto_required_metrics_passed(rows):
        failure_labels.append("required_metrics")
    if not _pareto_reference_metrics_passed(rows):
        failure_labels.append("reference_metrics")

    score_mean = _mean_prefixed(rows, "metric_score")
    reference_score_mean = _mean_prefixed(rows, "ref_score")
    score_delta_vs_best = (
        score_mean - baseline["score"] if score_mean is not None else float("-inf")
    )
    score_delta_vs_baseline = (
        score_mean - reference_score_mean
        if score_mean is not None and reference_score_mean is not None
        else float("-inf")
    )
    max_global_ratio_vs_best = _mean_max_global_ratio(
        rows,
        prefix="metric_",
        baseline=baseline,
    )
    max_global_ratio_vs_baseline = _mean_max_global_ratio(
        rows,
        prefix="metric_",
        ref_prefix="ref_",
    )
    metrics = {
        f"{metric}_mean": _mean_prefixed(rows, f"metric_{metric}")
        for metric in PARETO_PRIMARY_METRICS
    }
    metrics.update(
        {
            "duration_sec_mean": _mean_prefixed(rows, "duration_sec"),
            "score_delta_vs_best_mean": score_delta_vs_best,
            "score_delta_vs_baseline_mean": score_delta_vs_baseline,
            "max_global_ratio_vs_best_mean": max_global_ratio_vs_best,
            "max_global_ratio_vs_baseline_mean": max_global_ratio_vs_baseline,
        }
    )

    if ok_count == 0:
        pareto_class = "invalid"
    elif failure_labels:
        pareto_class = "unstable"
    elif (
        score_delta_vs_baseline >= -PARETO_BASELINE_SCORE_TOLERANCE
        and max_global_ratio_vs_baseline <= PARETO_BASELINE_GLOBAL_RATIO_LIMIT
    ):
        pareto_class = "baseline_close"
    elif score_delta_vs_best > 0.0 and max_global_ratio_vs_best <= 1.0:
        pareto_class = "promising_budget"
    else:
        pareto_class = "stable_but_low_quality"

    return {
        "pareto_class": pareto_class,
        "failure_labels": failure_labels,
        "repeat_count": repeat_count,
        "ok_count": ok_count,
        "success_count": success_count,
        "accepted_count": accepted_count,
        "fallback_count": fallback_count,
        "full_run_count": full_run_count,
        "accepted_windows_count": accepted_windows_count,
        **metrics,
    }


def _empty_pareto_result(
    pareto_class: str,
    failure_labels: Sequence[str],
) -> dict[str, Any]:
    metrics = {f"{metric}_mean": float("nan") for metric in PARETO_PRIMARY_METRICS}
    metrics.update(
        {
            "duration_sec_mean": float("nan"),
            "score_delta_vs_best_mean": float("-inf"),
            "score_delta_vs_baseline_mean": float("-inf"),
            "max_global_ratio_vs_best_mean": float("inf"),
            "max_global_ratio_vs_baseline_mean": float("inf"),
        }
    )
    return {
        "pareto_class": pareto_class,
        "failure_labels": list(failure_labels),
        "repeat_count": 0,
        "ok_count": 0,
        "success_count": 0,
        "accepted_count": 0,
        "fallback_count": 0,
        "full_run_count": 0,
        "accepted_windows_count": 0,
        **metrics,
    }


def _pareto_required_metrics_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        _as_float(row.get(f"metric_{metric}")) is not None
        for row in rows
        for metric in PARETO_PRIMARY_METRICS
    )


def _pareto_reference_metrics_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    required_ref_keys = ("ref_score",) + tuple(
        f"ref_{metric}" for metric in GLOBAL_GUARDRAIL_METRICS
    )
    return all(
        _as_float(row.get(key)) is not None
        for row in rows
        for key in required_ref_keys
    )


def _mean_prefixed(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    numbers = [
        value
        for value in (_as_float(row.get(key)) for row in rows)
        if value is not None
    ]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _mean_max_global_ratio(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    baseline: Mapping[str, float] | None = None,
    ref_prefix: str | None = None,
) -> float:
    ratios: list[float] = []
    for row in rows:
        row_ratios: list[float] = []
        for metric in GLOBAL_GUARDRAIL_METRICS:
            value = _as_float(row.get(f"{prefix}{metric}"))
            denominator = (
                baseline[metric]
                if baseline is not None
                else _as_float(row.get(f"{ref_prefix}{metric}"))
            )
            if value is None or denominator is None or denominator == 0.0:
                continue
            row_ratios.append(value / float(denominator))
        if row_ratios:
            ratios.append(max(row_ratios))
    if not ratios:
        return float("inf")
    return sum(ratios) / len(ratios)


def _get_candidate(candidate_name: str) -> CandidateDefinition:
    try:
        return CANDIDATE_DEFINITIONS[candidate_name]
    except KeyError as exc:
        raise ValueError(f"Unknown candidate {candidate_name!r}.") from exc


def _get_baseline(motion: str) -> Mapping[str, float]:
    try:
        return BEST_S128_BASELINES[motion]
    except KeyError as exc:
        raise ValueError(f"Unknown motion {motion!r}.") from exc


def _contact_limit(motion: str, baseline: Mapping[str, float]) -> float:
    limit = baseline["contact_mismatch_rate"]
    if motion == "qixing":
        return min(limit, QIXING_CONTACT_TARGET)
    return limit


def _mpc_guardrail_passed(
    metrics: Mapping[str, Any],
    mpc: Mapping[str, Any],
) -> bool:
    fallback = _as_bool(mpc.get("used_baseline_fallback"))
    if fallback is not False:
        return False
    if _as_bool(mpc.get("accepted")) is not True:
        return False

    num_steps = _as_float(metrics.get("num_steps"))
    if num_steps != FULL_RUN_STEPS:
        return False
    return _as_int(mpc.get("accepted_windows")) == FULL_RUN_ACCEPTED_WINDOWS


def _runtime_guardrail_passed(
    duration_sec: float | None,
    control_duration_sec: float | None,
) -> bool:
    if duration_sec is None or control_duration_sec is None:
        return False
    if not math.isfinite(duration_sec) or not math.isfinite(control_duration_sec):
        return False
    if duration_sec <= 0.0:
        return False
    if control_duration_sec <= 0.0:
        return False
    return duration_sec <= control_duration_sec * RUNTIME_REGRESSION_MULTIPLIER


def _at_most(values: Mapping[str, Any], key: str, limit: float) -> bool:
    value = _as_float(values.get(key))
    return value is not None and value <= limit


def _at_least(values: Mapping[str, Any], key: str, limit: float) -> bool:
    value = _as_float(values.get(key))
    return value is not None and value >= limit


def _metric_delta(values: Mapping[str, Any], key: str, baseline: float) -> float:
    value = _as_float(values.get(key))
    if value is None:
        return float("nan")
    return value - baseline


def _metric_ratio(values: Mapping[str, Any], key: str, baseline: float) -> float:
    value = _as_float(values.get(key))
    if value is None or baseline <= 0.0:
        return float("inf")
    return value / baseline


def _local_improvement(values: Mapping[str, Any], key: str, baseline: float) -> float:
    value = _as_float(values.get(key))
    if value is None or baseline <= 0.0:
        return 0.0
    return 1.0 - value / baseline


def _local_threshold_improved(
    values: Mapping[str, Any],
    baseline: Mapping[str, Any],
    key: str,
    multiplier: float,
) -> bool:
    value = _as_float(values.get(key))
    baseline_value = _as_float(baseline.get(key))
    if value is None or baseline_value is None or baseline_value <= 0.0:
        return False
    return value <= baseline_value * multiplier


def _missing_required_metrics(values: Mapping[str, Any]) -> list[str]:
    return [
        metric
        for metric in ASSESSMENT_REQUIRED_METRICS
        if _as_float(values.get(metric)) is None
    ]


def _missing_required_mpc(values: Mapping[str, Any]) -> list[str]:
    return [field for field in ASSESSMENT_REQUIRED_MPC if field not in values]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped.isdigit():
        return None
    return int(stripped)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None
