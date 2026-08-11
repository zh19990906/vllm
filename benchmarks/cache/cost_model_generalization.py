# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from benchmarks.cache.cost_model_calibration import (
    CalibrationDataset,
    DecisionSample,
    Percentile,
    evaluate_profile,
)


@dataclass(frozen=True, slots=True)
class GeneralizationCondition:
    condition_id: str
    model: str
    served_model: str
    concurrency: int
    request_rate: str | float
    tensor_parallel_size: int
    gpu_uuid: str
    environment_artifact: str
    run_directories: dict[str, str]
    dataset: CalibrationDataset
    sample_metadata: tuple[dict[str, Any], ...]
    excluded_samples: tuple[dict[str, Any], ...]


def load_generalization_condition(
    path: Path,
    percentile: str = "p95",
) -> GeneralizationCondition:
    if percentile not in {"p50", "p95", "p99"}:
        raise ValueError(f"unsupported percentile: {percentile}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    if raw.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if raw.get("issue") != 15:
        raise ValueError("issue must be 15")

    condition = raw["condition"]
    requests_per_case = int(condition["requests_per_case"])
    if requests_per_case <= 0:
        raise ValueError("requests_per_case must be positive")
    samples: list[DecisionSample] = []
    sample_metadata: list[dict[str, Any]] = []

    for row in raw.get("samples", []):
        source = row["source"]
        if source not in {"cpu_primary", "secondary:filesystem"}:
            raise ValueError(f"unsupported source: {source}")

        total_external_tokens = row["external_kv_tokens_total"]
        per_request_external_tokens = row["external_kv_tokens_per_request"]

        if (
            type(total_external_tokens) is not int
            or type(per_request_external_tokens) is not int
            or total_external_tokens <= 0
            or per_request_external_tokens <= 0
            or total_external_tokens
            != requests_per_case * per_request_external_tokens
        ):
            raise ValueError(
                "external KV token total must equal "
                "requests_per_case * per-request external KV tokens"
            )

        latency = row["latency_ms"]
        samples.append(
            DecisionSample(
                source=source,
                requested_tokens=int(row["requested_tokens"]),
                external_tokens=per_request_external_tokens,
                actual_recompute_ms=float(
                    latency["recompute"][percentile]
                ),
                actual_restore_ms=float(
                    latency["restore"][percentile]
                ),
            )
        )
        sample_metadata.append(dict(row))

    excluded_samples = tuple(
        dict(item) for item in raw.get("excluded_samples", [])
    )

    dataset = CalibrationDataset(
        percentile=cast(Percentile, percentile),
        source_artifact=str(path),
        requests_per_case=requests_per_case,
        decision_samples=tuple(
            sorted(
                samples,
                key=lambda sample: (
                    sample.requested_tokens,
                    sample.source,
                ),
            )
        ),
        repeat_direction_checks=(),
        excluded_samples=excluded_samples,
    )

    return GeneralizationCondition(
        condition_id=str(condition["id"]),
        model=str(condition["model"]),
        served_model=str(condition["served_model"]),
        concurrency=int(condition["concurrency"]),
        request_rate=condition["request_rate"],
        tensor_parallel_size=int(condition["tensor_parallel_size"]),
        gpu_uuid=str(condition["gpu_uuid"]),
        environment_artifact=str(condition["environment_artifact"]),
        run_directories={
            str(key): str(value)
            for key, value in condition["run_directories"].items()
        },
        dataset=dataset,
        sample_metadata=tuple(sample_metadata),
        excluded_samples=excluded_samples,
    )



DECISION_ACCURACY_MIN = 0.95
PRINCIPAL_MACRO_MAPE_MAX = 15.0
PRINCIPAL_CURVE_MAPE_MAX = 20.0
BOUNDARY_MARGIN_MS = 1.0


def _mape_percent(relative_errors: list[float]) -> float | None:
    if not relative_errors:
        return None
    return 100.0 * sum(relative_errors) / len(relative_errors)


def _build_high_confidence_gate(
    rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    high_confidence = [
        row for row in rows if row.get("confidence") == "high"
    ]

    recompute_by_key: dict[
        tuple[int, int],
        tuple[float, float, float],
    ] = {}
    for row in high_confidence:
        key = (
            int(row["requested_tokens"]),
            int(row["external_tokens"]),
        )
        value = (
            float(row["actual_recompute_ms"]),
            float(row["predicted_recompute_ms"]),
            float(row["recompute_relative_error"]),
        )
        previous = recompute_by_key.get(key)
        if previous is not None and previous != value:
            raise ValueError(
                f"inconsistent recompute scoring for anchor {key}"
            )
        recompute_by_key[key] = value

    recompute_errors = [
        value[2] for value in recompute_by_key.values()
    ]
    cpu_restore_errors = [
        float(row["restore_relative_error"])
        for row in high_confidence
        if row["source"] == "cpu_primary"
    ]
    filesystem_restore_errors = [
        float(row["restore_relative_error"])
        for row in high_confidence
        if row["source"] == "secondary:filesystem"
    ]

    recompute_mape = _mape_percent(recompute_errors)
    cpu_restore_mape = _mape_percent(cpu_restore_errors)
    filesystem_restore_mape = _mape_percent(
        filesystem_restore_errors
    )

    missing_principal_curves: list[str] = []
    if recompute_mape is None:
        missing_principal_curves.append("recompute")
    if cpu_restore_mape is None:
        missing_principal_curves.append("cpu_restore")
    if filesystem_restore_mape is None:
        missing_principal_curves.append("tiered_fs_restore")

    decision_total = len(high_confidence)
    decision_correct = sum(
        bool(row["decision_correct"])
        for row in high_confidence
    )
    decision_accuracy = (
        decision_correct / decision_total
        if decision_total
        else 0.0
    )

    principal_macro_mape = None
    if not missing_principal_curves:
        assert recompute_mape is not None
        assert cpu_restore_mape is not None
        assert filesystem_restore_mape is not None
        principal_macro_mape = (
            recompute_mape
            + cpu_restore_mape
            + filesystem_restore_mape
        ) / 3.0

    clear_margin_wrong_decisions = sum(
        (
            not bool(row["decision_correct"])
            and abs(float(row["actual_margin_ms"]))
            > BOUNDARY_MARGIN_MS
        )
        for row in high_confidence
    )

    failure_reasons: list[str] = []

    if missing_principal_curves:
        classification = "insufficient_evidence"
    else:
        assert principal_macro_mape is not None
        assert recompute_mape is not None
        assert cpu_restore_mape is not None
        assert filesystem_restore_mape is not None

        if decision_accuracy < DECISION_ACCURACY_MIN:
            failure_reasons.append("decision_accuracy")
        if principal_macro_mape > PRINCIPAL_MACRO_MAPE_MAX:
            failure_reasons.append("principal_macro_mape")
        if any(
            value > PRINCIPAL_CURVE_MAPE_MAX
            for value in (
                recompute_mape,
                cpu_restore_mape,
                filesystem_restore_mape,
            )
        ):
            failure_reasons.append("principal_curve_mape")
        if clear_margin_wrong_decisions:
            failure_reasons.append(
                "clear_margin_wrong_decision"
            )

        classification = (
            "fixed_profile_transfer_fail"
            if failure_reasons
            else "fixed_profile_transfer_pass"
        )

    gate = {
        "thresholds": {
            "decision_accuracy_min": DECISION_ACCURACY_MIN,
            "principal_macro_mape_percent_max": (
                PRINCIPAL_MACRO_MAPE_MAX
            ),
            "principal_curve_mape_percent_max": (
                PRINCIPAL_CURVE_MAPE_MAX
            ),
            "boundary_margin_ms": BOUNDARY_MARGIN_MS,
        },
        "high_confidence": {
            "decision_correct": decision_correct,
            "decision_total": decision_total,
            "decision_accuracy": decision_accuracy,
            "recompute_sample_count": len(recompute_errors),
            "cpu_restore_sample_count": len(
                cpu_restore_errors
            ),
            "tiered_fs_restore_sample_count": len(
                filesystem_restore_errors
            ),
            "recompute_mape_percent": recompute_mape,
            "cpu_restore_mape_percent": cpu_restore_mape,
            "tiered_fs_restore_mape_percent": (
                filesystem_restore_mape
            ),
            "principal_macro_mape_percent": (
                principal_macro_mape
            ),
        },
        "missing_principal_curves": missing_principal_curves,
        "clear_margin_wrong_decisions": (
            clear_margin_wrong_decisions
        ),
        "failure_reasons": failure_reasons,
    }
    return classification, gate


def evaluate_frozen_condition(
    condition: GeneralizationCondition,
    profile: dict[str, Any],
    *,
    profile_identity: str,
) -> dict[str, Any]:
    evaluation = evaluate_profile(condition.dataset, profile)
    rows = list(evaluation["samples"])
    classification, gate = _build_high_confidence_gate(rows)

    return {
        "schema_version": 1,
        "issue": 15,
        "mode": "frozen_profile_holdout",
        "condition_id": condition.condition_id,
        "profile_identity": profile_identity,
        "percentile": condition.dataset.percentile,
        "classification": classification,
        "gate": gate,
        "low_confidence_samples": [
            row for row in rows
            if row.get("confidence") != "high"
        ],
        "evaluation": evaluation,
    }



def _diagnose_curve(
    points: list[tuple[float, float]],
) -> dict[str, Any]:
    if not points:
        return {
            "sample_count": 0,
            "raw_mape_percent": None,
            "scale": None,
            "residual_mape_percent": None,
            "classification": "insufficient_evidence",
        }

    for actual, predicted in points:
        if actual <= 0:
            raise ValueError(
                "diagnostic actual latency must be positive"
            )
        if predicted <= 0:
            raise ValueError(
                "diagnostic predicted latency must be positive"
            )

    raw_mape = 100.0 * statistics.fmean(
        abs(predicted - actual) / actual
        for actual, predicted in points
    )
    scale = float(
        statistics.median(
            actual / predicted
            for actual, predicted in points
        )
    )
    residual_mape = 100.0 * statistics.fmean(
        abs(predicted * scale - actual) / actual
        for actual, predicted in points
    )

    if raw_mape <= PRINCIPAL_MACRO_MAPE_MAX:
        classification = "transferable"
    elif residual_mape <= PRINCIPAL_MACRO_MAPE_MAX:
        classification = "environment_specific_scale_candidate"
    else:
        classification = "curve_shape_or_missing_feature"

    return {
        "sample_count": len(points),
        "raw_mape_percent": raw_mape,
        "scale": scale,
        "residual_mape_percent": residual_mape,
        "classification": classification,
    }


def diagnose_curve_scaling(
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    rows = evaluation["evaluation"]["samples"]

    recompute_by_key: dict[
        tuple[int, int],
        tuple[float, float],
    ] = {}
    cpu_restore: list[tuple[float, float]] = []
    filesystem_restore: list[tuple[float, float]] = []

    for row in rows:
        if row.get("confidence") != "high":
            continue

        recompute_key = (
            int(row["requested_tokens"]),
            int(row["external_tokens"]),
        )
        recompute_value = (
            float(row["actual_recompute_ms"]),
            float(row["predicted_recompute_ms"]),
        )
        previous = recompute_by_key.get(recompute_key)
        if previous is not None and previous != recompute_value:
            raise ValueError(
                "inconsistent recompute diagnostics for anchor "
                f"{recompute_key}"
            )
        recompute_by_key[recompute_key] = recompute_value

        restore_value = (
            float(row["actual_restore_ms"]),
            float(row["predicted_restore_ms"]),
        )
        if row["source"] == "cpu_primary":
            cpu_restore.append(restore_value)
        elif row["source"] == "secondary:filesystem":
            filesystem_restore.append(restore_value)

    return {
        "method": "median_actual_over_frozen_prediction_scalar",
        "thresholds": {
            "raw_mape_percent_transferable_max": (
                PRINCIPAL_MACRO_MAPE_MAX
            ),
            "scalar_residual_mape_percent_max": (
                PRINCIPAL_MACRO_MAPE_MAX
            ),
        },
        "curves": {
            "recompute": _diagnose_curve(
                list(recompute_by_key.values())
            ),
            "cpu_restore": _diagnose_curve(cpu_restore),
            "tiered_fs_restore": _diagnose_curve(
                filesystem_restore
            ),
        },
    }
