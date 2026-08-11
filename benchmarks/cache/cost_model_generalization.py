# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from benchmarks.cache.cost_model_calibration import (
    CalibrationDataset,
    DecisionSample,
    Percentile,
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
