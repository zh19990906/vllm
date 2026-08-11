# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Percentile = Literal["p50", "p95", "p99"]
Source = Literal["cpu_primary", "secondary:filesystem"]


_REPO_ROOT = Path(__file__).resolve().parents[2]
_COST_MODEL_PATH = _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "cost_model.py"
_SPEC = importlib.util.spec_from_file_location(
    "vllm_cost_model_for_calibration",
    _COST_MODEL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"unable to load cost model: {_COST_MODEL_PATH}")
_COST_MODEL = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_SPEC.name, _COST_MODEL)
_SPEC.loader.exec_module(_COST_MODEL)

LoadProvenance = _COST_MODEL.LoadProvenance
OffloadCostModel = _COST_MODEL.OffloadCostModel


@dataclass(frozen=True, slots=True)
class DecisionSample:
    source: Source
    requested_tokens: int
    external_tokens: int
    actual_recompute_ms: float
    actual_restore_ms: float


@dataclass(frozen=True, slots=True)
class CalibrationDataset:
    percentile: Percentile
    source_artifact: str
    requests_per_case: int
    decision_samples: tuple[DecisionSample, ...]
    repeat_direction_checks: tuple[dict[str, Any], ...]
    excluded_samples: tuple[dict[str, Any], ...]


def load_profile_artifact(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("profile artifact must be a JSON object")

    cost_model = raw.get("cache_cost_model")
    if not isinstance(cost_model, dict):
        raise ValueError("profile artifact requires cache_cost_model")
    if cost_model.get("mode") != "shadow":
        raise ValueError("profile artifact cache_cost_model.mode must be shadow")
    if not isinstance(cost_model.get("profile"), dict):
        raise ValueError("profile artifact requires cache_cost_model.profile mapping")

    return cost_model


def load_issue13_dataset(
    path: Path,
    percentile: str = "p95",
) -> CalibrationDataset:
    if percentile not in {"p50", "p95", "p99"}:
        raise ValueError(f"unsupported percentile: {percentile}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    requests_per_case = int(raw["scope"]["requests_per_case"])
    samples: list[DecisionSample] = []

    def external_tokens(row: dict[str, Any]) -> int:
        total = row["external_kv_tokens"]
        if type(total) is not int or total <= 0:
            raise ValueError("external_kv_tokens must be a positive integer")
        if total % requests_per_case != 0:
            raise ValueError(
                "external_kv_tokens must be divisible by requests_per_case"
            )
        return total // requests_per_case

    for row in raw.get("wide_curve", []):
        tokens = external_tokens(row)
        requested = int(row["requested_tokens"])
        recompute = float(row["recompute_ttft_ms"][percentile])

        samples.append(
            DecisionSample(
                source="cpu_primary",
                requested_tokens=requested,
                external_tokens=tokens,
                actual_recompute_ms=recompute,
                actual_restore_ms=float(row["cpu_restore_ttft_ms"][percentile]),
            )
        )
        samples.append(
            DecisionSample(
                source="secondary:filesystem",
                requested_tokens=requested,
                external_tokens=tokens,
                actual_recompute_ms=recompute,
                actual_restore_ms=float(row["tiered_fs_ttft_ms"][percentile]),
            )
        )

    for row in raw.get("cpu_crossover_points", []):
        samples.append(
            DecisionSample(
                source="cpu_primary",
                requested_tokens=int(row["requested_tokens"]),
                external_tokens=external_tokens(row),
                actual_recompute_ms=float(row["recompute_ttft_ms"][percentile]),
                actual_restore_ms=float(row["cpu_restore_ttft_ms"][percentile]),
            )
        )

    repeat_direction_checks = tuple(
        {
            "requested_tokens": int(requested_tokens),
            "p95_delta_ms": tuple(float(value) for value in repeat["p95_delta_ms"]),
            "all_restore_faster": all(
                float(value) < 0 for value in repeat["p95_delta_ms"]
            ),
        }
        for requested_tokens, repeat in sorted(
            raw.get("boundary_repeats", {}).items(),
            key=lambda item: int(item[0]),
        )
    )

    excluded_samples: list[dict[str, Any]] = []
    if raw.get("invalid_cpu_2g_sweep"):
        excluded_samples.append(
            {
                "reason": "invalid_cpu_restore_provenance",
                "count": len(raw["invalid_cpu_2g_sweep"]),
            }
        )
    if raw.get("workload_generation_failure_208"):
        excluded_samples.append(
            {
                "reason": "workload_generation_failure",
                "count": len(raw["workload_generation_failure_208"]),
            }
        )

    return CalibrationDataset(
        percentile=percentile,
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
        repeat_direction_checks=repeat_direction_checks,
        excluded_samples=tuple(excluded_samples),
    )


def _median_curve(points: list[tuple[int, float]]) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for tokens, value in points:
        grouped[tokens].append(value)
    return {
        tokens: float(statistics.median(grouped[tokens])) for tokens in sorted(grouped)
    }


def derive_calibrated_profile(
    dataset: CalibrationDataset,
    before_profile: Mapping[str, Any],
) -> dict[str, Any]:
    recompute_seen: set[tuple[int, int, float]] = set()
    recompute_points: list[tuple[int, float]] = []
    cpu_points: list[tuple[int, float]] = []
    filesystem_points: list[tuple[int, float]] = []

    for sample in dataset.decision_samples:
        recompute_key = (
            sample.requested_tokens,
            sample.external_tokens,
            sample.actual_recompute_ms,
        )
        if recompute_key not in recompute_seen:
            recompute_seen.add(recompute_key)
            recompute_points.append(
                (sample.external_tokens, sample.actual_recompute_ms)
            )

        if sample.source == "cpu_primary":
            cpu_points.append((sample.external_tokens, sample.actual_restore_ms))
        elif sample.source == "secondary:filesystem":
            filesystem_points.append((sample.external_tokens, sample.actual_restore_ms))

    before_fs = before_profile["profile"]["tiers"]["filesystem"]

    return {
        "mode": "shadow",
        "ewma_alpha": float(before_profile.get("ewma_alpha", 0.2)),
        "sample_scale_min": float(before_profile.get("sample_scale_min", 0.25)),
        "sample_scale_max": float(before_profile.get("sample_scale_max", 4.0)),
        "profile": {
            "recompute_ms": _median_curve(recompute_points),
            "tiers": {
                "cpu_primary": {
                    "restore_ms": _median_curve(cpu_points),
                },
                "filesystem": {
                    "restore_ms": _median_curve(filesystem_points),
                    "promotion_ms": dict(before_fs["promotion_ms"]),
                },
            },
        },
    }


def _mape(errors: list[float]) -> float:
    if not errors:
        raise ValueError("cannot calculate MAPE without samples")
    return 100.0 * sum(errors) / len(errors)


def _evaluate_profile(
    dataset: CalibrationDataset,
    profile: Mapping[str, Any],
    *,
    boundary_margin_ms: float,
) -> dict[str, Any]:
    model = OffloadCostModel.from_extra_config({"cache_cost_model": dict(profile)})
    if model is None:
        raise ValueError("evaluation profile must enable shadow mode")

    rows: list[dict[str, Any]] = []
    for sample in dataset.decision_samples:
        decision = model.shadow_decide(
            LoadProvenance(
                source=sample.source,
                external_tokens=sample.external_tokens,
                secondary_promoted_tokens=(
                    sample.external_tokens
                    if sample.source.startswith("secondary:")
                    else 0
                ),
                sources=(sample.source,),
                confidence="high",
            )
        )
        if decision is None:
            raise ValueError(f"profile cannot score source {sample.source}")

        actual_margin_ms = sample.actual_restore_ms - sample.actual_recompute_ms
        predicted_margin_ms = (
            decision.restore_estimate_ms - decision.recompute_estimate_ms
        )

        rows.append(
            {
                "source": sample.source,
                "requested_tokens": sample.requested_tokens,
                "external_tokens": sample.external_tokens,
                "actual_recompute_ms": sample.actual_recompute_ms,
                "actual_restore_ms": sample.actual_restore_ms,
                "predicted_recompute_ms": decision.recompute_estimate_ms,
                "predicted_restore_ms": decision.restore_estimate_ms,
                "actual_margin_ms": actual_margin_ms,
                "predicted_margin_ms": predicted_margin_ms,
                "actual_preferred": (
                    "restore" if actual_margin_ms < 0 else "recompute"
                ),
                "predicted_preferred": decision.preferred,
                "boundary_sensitive": (abs(actual_margin_ms) <= boundary_margin_ms),
                "recompute_abs_error_ms": abs(
                    decision.recompute_estimate_ms - sample.actual_recompute_ms
                ),
                "recompute_relative_error": abs(
                    decision.recompute_estimate_ms - sample.actual_recompute_ms
                )
                / sample.actual_recompute_ms,
                "restore_abs_error_ms": abs(
                    decision.restore_estimate_ms - sample.actual_restore_ms
                ),
                "restore_relative_error": abs(
                    decision.restore_estimate_ms - sample.actual_restore_ms
                )
                / sample.actual_restore_ms,
                "decision_correct": (
                    decision.preferred
                    == ("restore" if actual_margin_ms < 0 else "recompute")
                ),
                "confidence": decision.confidence,
                "runtime_scale": decision.runtime_scale,
            }
        )

    recompute_by_key: dict[
        tuple[int, int],
        tuple[float, float, float],
    ] = {}
    for row in rows:
        key = (
            row["requested_tokens"],
            row["external_tokens"],
        )
        value = (
            row["actual_recompute_ms"],
            row["predicted_recompute_ms"],
            row["recompute_relative_error"],
        )
        previous = recompute_by_key.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"inconsistent recompute scoring for anchor {key}")
        recompute_by_key[key] = value

    recompute_errors = [value[2] for value in recompute_by_key.values()]
    cpu_restore_errors = [
        row["restore_relative_error"] for row in rows if row["source"] == "cpu_primary"
    ]
    filesystem_restore_errors = [
        row["restore_relative_error"]
        for row in rows
        if row["source"] == "secondary:filesystem"
    ]

    recompute_mape = _mape(recompute_errors)
    cpu_restore_mape = _mape(cpu_restore_errors)
    filesystem_restore_mape = _mape(filesystem_restore_errors)

    return {
        "samples": rows,
        "aggregate": {
            "decision_correct": sum(row["decision_correct"] for row in rows),
            "decision_total": len(rows),
            "decision_accuracy": (
                sum(row["decision_correct"] for row in rows) / len(rows)
            ),
            "recompute_sample_count": len(recompute_errors),
            "cpu_restore_sample_count": len(cpu_restore_errors),
            "tiered_fs_restore_sample_count": len(filesystem_restore_errors),
            "recompute_mape_percent": recompute_mape,
            "cpu_restore_mape_percent": cpu_restore_mape,
            "tiered_fs_restore_mape_percent": filesystem_restore_mape,
            "principal_macro_mape_percent": (
                recompute_mape + cpu_restore_mape + filesystem_restore_mape
            )
            / 3.0,
        },
    }


def evaluate_profile(
    dataset: CalibrationDataset,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    return _evaluate_profile(
        dataset,
        profile,
        boundary_margin_ms=1.0,
    )


def build_calibration_result(
    dataset: CalibrationDataset,
    before_profile: Mapping[str, Any],
    after_profile: Mapping[str, Any],
    *,
    mape_threshold_percent: float = 15.0,
    decision_accuracy_threshold: float = 0.95,
    boundary_margin_ms: float = 1.0,
) -> dict[str, Any]:
    before = _evaluate_profile(
        dataset,
        before_profile,
        boundary_margin_ms=boundary_margin_ms,
    )
    after = _evaluate_profile(
        dataset,
        after_profile,
        boundary_margin_ms=boundary_margin_ms,
    )

    after_aggregate = after["aggregate"]
    accuracy_ok = after_aggregate["decision_accuracy"] >= decision_accuracy_threshold
    mape_ok = after_aggregate["principal_macro_mape_percent"] <= mape_threshold_percent

    return {
        "schema_version": 1,
        "source_artifact": dataset.source_artifact,
        "percentile": dataset.percentile,
        "acceptance_thresholds": {
            "decision_accuracy_min": decision_accuracy_threshold,
            "principal_macro_mape_percent_max": (mape_threshold_percent),
            "boundary_margin_ms": boundary_margin_ms,
        },
        "before_profile": dict(before_profile),
        "calibrated_profile": dict(after_profile),
        "before": before,
        "after": after,
        "repeat_direction_checks": list(dataset.repeat_direction_checks),
        "excluded_samples": list(dataset.excluded_samples),
        "acceptance": {
            "passed": accuracy_ok and mape_ok,
            "decision_accuracy_ok": accuracy_ok,
            "principal_macro_mape_ok": mape_ok,
            "decision_correct": after_aggregate["decision_correct"],
            "decision_total": after_aggregate["decision_total"],
            "decision_accuracy": after_aggregate["decision_accuracy"],
            "principal_macro_mape_percent": after_aggregate[
                "principal_macro_mape_percent"
            ],
        },
    }
