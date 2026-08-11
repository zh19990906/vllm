# Issue #14 Shadow Cost Model Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the existing shadow-only KV offload cost model against the #13 P95 hardware baseline, reach 14/14 correct anchor decisions and principal P95 macro-MAPE <= 15%, and record reproducible before/after evidence without changing the active execution path.

**Architecture:** Keep `OffloadCostModel` and `CostCurve` semantics unchanged in Phase 1. Add a pure calibration module that turns the checked-in #13 structured artifact into runtime-axis (`external_tokens`) calibration samples, derives median-aggregated P95 curves, evaluates both the preserved #12 baseline profile and the new calibrated profile through the real `OffloadCostModel`, and emits deterministic metrics. A thin CLI drives the evaluator; focused unit tests verify profile construction, decision accounting, error metrics, and EWMA behavior; final validation artifacts record the real #13 before/after result.

**Tech Stack:** Python 3.11, stdlib `argparse` / `dataclasses` / `json` / `statistics`, existing `vllm/v1/kv_offload/cost_model.py`, pytest, Ruff, #13 structured JSON validation artifact.

## Global Constraints

- Primary calibration percentile is **P95 TTFT**; P50 and P99 are diagnostics only.
- Acceptance requires **14/14 anchor decisions correct** on the fixed #13 baseline, which satisfies the agreed `>=95%` threshold.
- Acceptance requires **principal P95 macro-MAPE <= 15%**, where the macro-average is over recompute, CPU-primary restore, and tiered-fs restore curve MAPEs.
- `abs(actual_margin_ms) <= 1.0` marks a boundary-sensitive sample but does not remove it from decision accuracy.
- Runtime calibration axis is **actual external KV tokens per request**, not requested prompt tokens.
- Duplicate `(curve, external_tokens)` calibration points use the **median P95 latency** while every original anchor remains in evaluation.
- `boundary_repeats` are supplemental direction checks only; they do not enter MAPE or the 14-anchor decision denominator because the compact artifact lacks per-repeat absolute TTFT.
- Exclude the 2 GiB CPU sweep from restore calibration because provenance proves recompute; exclude the 208 workload-generation failure from latency scoring.
- Keep `mode: shadow`; do not enable or implement active enforcement.
- Do not change matched-token count, lookup semantics, allocation, transfer jobs, cache contents, scheduler return values, or actual restore/recompute execution.
- Phase 1 must not modify `CostCurve`, `OffloadCostModel`, scheduler, provenance, manager, or metrics behavior.
- Preserve the #12 secondary promotion curve and EWMA settings as inherited calibration metadata because #13 does not contain a clean per-anchor promotion-latency curve.
- Do not introduce requested-token or transfer-byte runtime features in #14 solely to fit this baseline; those remain #15 candidates.
- Do not claim physical NVMe provenance; use `tiered-fs` / filesystem terminology.
- Do not run Pod full `pre-commit --all-files`; run focused local checks and leave repository-wide pre-commit to GitHub Actions.
- If the real Phase 1 gate unexpectedly misses either acceptance threshold, stop after emitting residual evidence. Do not silently tune the formula; return to the approved design's evidence-triggered Phase 2 decision.

---

## File Structure

- Create `benchmarks/cache/profiles/issue12-shadow-cost-baseline.json`
    - Explicit, machine-specific preservation of the #12 test/design profile used for before-calibration scoring.
    - Contains provenance metadata plus the exact `cache_cost_model` mapping; no runtime defaults are introduced.
- Create `benchmarks/cache/cost_model_calibration.py`
    - Pure artifact parsing, eligibility/exclusion rules, runtime-axis sample construction, median profile derivation, real `OffloadCostModel` evaluation, aggregate metrics, repeat-direction checks, and deterministic result construction.
- Create `benchmarks/cache/evaluate_cost_model.py`
    - Thin CLI around the pure calibration module; JSON output and `--check` exit status.
- Create `benchmarks/cache/tests/test_cost_model_calibration.py`
    - Synthetic-artifact TDD coverage for parsing, pairing, duplicate aggregation, exclusions, metrics, decisions, baseline-vs-calibrated behavior, and deterministic output.
- Modify `tests/v1/kv_offload/test_cost_model.py`
    - Add calibrated-profile decision regression coverage and deterministic EWMA convergence/stability tests only; Phase 1 does not change production cost-model code.
- Create `docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json`
    - Machine-readable real #13 before/after calibration result.
- Create `docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.md`
    - Human-readable metrics, per-anchor decisions, error-source explanation, exclusions, commands, EWMA verification, limitations, and shadow-only statement.

---

### Task 1: Preserve the #12 Before-Calibration Profile Explicitly

**Files:**

- Create: `benchmarks/cache/profiles/issue12-shadow-cost-baseline.json`
- Test: `benchmarks/cache/tests/test_cost_model_calibration.py`

**Interfaces:**

- Consumes: #12 profile values already recorded in `tests/v1/kv_offload/test_cost_model.py` and `docs/superpowers/specs/2026-08-07-kv-offload-shadow-cost-model-design.md`.
- Produces: `load_profile_artifact(path: Path) -> dict[str, Any]` contract consumed by Tasks 2-5; the JSON's `cache_cost_model` value is directly acceptable to `OffloadCostModel.from_extra_config()` when wrapped as `{"cache_cost_model": ...}` only if needed by the helper.

- [ ] **Step 1: Write the baseline profile artifact**

Create `benchmarks/cache/profiles/issue12-shadow-cost-baseline.json` with exactly this shape and values:

```json
{
  "schema_version": 1,
  "name": "issue12-shadow-cost-baseline",
  "provenance": {
    "issue": 12,
    "profile_role": "before_calibration",
    "source_test": "tests/v1/kv_offload/test_cost_model.py",
    "source_design": "docs/superpowers/specs/2026-08-07-kv-offload-shadow-cost-model-design.md",
    "note": "Benchmark profile evidence only; not a production default."
  },
  "cache_cost_model": {
    "mode": "shadow",
    "ewma_alpha": 0.2,
    "sample_scale_min": 0.25,
    "sample_scale_max": 4.0,
    "profile": {
      "recompute_ms": {
        "256": 26.414,
        "512": 44.961,
        "1024": 81.705,
        "2048": 152.461,
        "4096": 308.424
      },
      "tiers": {
        "cpu_primary": {
          "restore_ms": {
            "1024": 24.49
          }
        },
        "filesystem": {
          "restore_ms": {
            "256": 31.119,
            "512": 56.979,
            "1024": 108.132,
            "2048": 244.266,
            "4096": 651.127
          },
          "promotion_ms": {
            "256": 13.916,
            "512": 35.23,
            "1024": 81.458,
            "2048": 171.505,
            "4096": 498.874
          }
        }
      }
    }
  }
}
```

- [ ] **Step 2: Write RED profile-artifact parsing tests**

Create `benchmarks/cache/tests/test_cost_model_calibration.py` with a direct import of the pure calibration module that Task 2 will create, and add:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.cache.cost_model_calibration import load_profile_artifact


def test_load_profile_artifact_returns_shadow_mapping(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "fixture",
                "provenance": {"profile_role": "before_calibration"},
                "cache_cost_model": {
                    "mode": "shadow",
                    "profile": {
                        "recompute_ms": {"128": 10.0},
                        "tiers": {
                            "cpu_primary": {"restore_ms": {"128": 12.0}}
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_profile_artifact(path)

    assert profile["mode"] == "shadow"
    assert profile["profile"]["recompute_ms"] == {"128": 10.0}


def test_load_profile_artifact_rejects_missing_cost_model(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="cache_cost_model"):
        load_profile_artifact(path)
```

- [ ] **Step 3: Run RED and confirm the module is absent**

Run:

```bash
python -m pytest -q \
  benchmarks/cache/tests/test_cost_model_calibration.py::test_load_profile_artifact_returns_shadow_mapping \
  benchmarks/cache/tests/test_cost_model_calibration.py::test_load_profile_artifact_rejects_missing_cost_model
```

Expected: FAIL during import because `benchmarks.cache.cost_model_calibration` does not exist yet.

- [ ] **Step 4: Commit the explicit baseline artifact and RED tests together**

```bash
git add \
  benchmarks/cache/profiles/issue12-shadow-cost-baseline.json \
  benchmarks/cache/tests/test_cost_model_calibration.py
git commit -m "test: define issue 14 calibration baseline"
```

This RED commit is intentional; Task 2 supplies the implementation immediately afterward.

---

### Task 2: Build the Pure #13 Calibration Dataset and Derive the P95 Profile

**Files:**

- Create: `benchmarks/cache/cost_model_calibration.py`
- Modify: `benchmarks/cache/tests/test_cost_model_calibration.py`

**Interfaces:**

- Consumes: `load_profile_artifact(path: Path) -> dict[str, Any]` and the #13 artifact schema (`scope.requests_per_case`, `wide_curve`, `cpu_crossover_points`, `boundary_repeats`, invalid/failure sections).
- Produces:

```python
@dataclass(frozen=True, slots=True)
class DecisionSample:
    source: Literal["cpu_primary", "secondary:filesystem"]
    requested_tokens: int
    external_tokens: int
    actual_recompute_ms: float
    actual_restore_ms: float

@dataclass(frozen=True, slots=True)
class CalibrationDataset:
    percentile: Literal["p50", "p95", "p99"]
    source_artifact: str
    requests_per_case: int
    decision_samples: tuple[DecisionSample, ...]
    repeat_direction_checks: tuple[dict[str, Any], ...]
    excluded_samples: tuple[dict[str, Any], ...]


def load_profile_artifact(path: Path) -> dict[str, Any]: ...
def load_issue13_dataset(path: Path, percentile: str = "p95") -> CalibrationDataset: ...
def derive_calibrated_profile(
    dataset: CalibrationDataset,
    before_profile: Mapping[str, Any],
) -> dict[str, Any]: ...
```

The returned calibrated profile is a `cache_cost_model` mapping with `mode="shadow"`, inherited EWMA/clamp settings, derived `profile.recompute_ms`, derived CPU/filesystem `restore_ms`, and inherited filesystem `promotion_ms`.

- [ ] **Step 1: Add a compact synthetic #13 artifact fixture**

Append to `benchmarks/cache/tests/test_cost_model_calibration.py`:

```python
@pytest.fixture
def issue13_artifact(tmp_path: Path) -> Path:
    artifact = {
        "schema_version": 1,
        "issue": 13,
        "scope": {"requests_per_case": 8},
        "wide_curve": [
            {
                "requested_tokens": 256,
                "recompute_ttft_ms": {"p50": 20.0, "p95": 25.0, "p99": 27.0},
                "cpu_restore_ttft_ms": {"p50": 18.0, "p95": 21.0, "p99": 22.0},
                "tiered_fs_ttft_ms": {"p50": 28.0, "p95": 31.0, "p99": 33.0},
                "external_kv_tokens": 8 * 232,
            },
            {
                "requested_tokens": 512,
                "recompute_ttft_ms": {"p50": 40.0, "p95": 45.0, "p99": 47.0},
                "cpu_restore_ttft_ms": {"p50": 19.0, "p95": 23.0, "p99": 24.0},
                "tiered_fs_ttft_ms": {"p50": 50.0, "p95": 59.0, "p99": 61.0},
                "external_kv_tokens": 8 * 512,
            },
        ],
        "cpu_crossover_points": [
            {
                "requested_tokens": 192,
                "recompute_ttft_ms": {"p50": 18.0, "p95": 22.0, "p99": 23.0},
                "cpu_restore_ttft_ms": {"p50": 19.0, "p95": 21.8, "p99": 22.8},
                "external_kv_tokens": 8 * 168,
            },
            {
                "requested_tokens": 216,
                "recompute_ttft_ms": {"p50": 23.0, "p95": 24.8, "p99": 25.3},
                "cpu_restore_ttft_ms": {"p50": 20.0, "p95": 22.1, "p99": 22.9},
                "external_kv_tokens": 8 * 192,
            },
            {
                "requested_tokens": 224,
                "recompute_ttft_ms": {"p50": 23.2, "p95": 25.2, "p99": 26.0},
                "cpu_restore_ttft_ms": {"p50": 19.9, "p95": 22.3, "p99": 23.1},
                "external_kv_tokens": 8 * 192,
            },
        ],
        "boundary_repeats": {
            "192": {"p95_delta_ms": [-0.3, -0.2, -0.1]},
            "216": {"p95_delta_ms": [-3.2, -3.0, -2.3]},
        },
        "invalid_cpu_2g_sweep": [
            {"requested_tokens": 256, "classification": "recompute_not_cpu_restore"}
        ],
        "workload_generation_failure_208": [
            {
                "cache_mode": "cpu-offload",
                "status": "benchmark_error",
                "stage": "workload",
                "type": "WorkloadGenerationError",
            }
        ],
    }
    path = tmp_path / "issue13.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path
```

- [ ] **Step 2: Add RED dataset-construction tests**

```python
from benchmarks.cache.cost_model_calibration import (
    derive_calibrated_profile,
    load_issue13_dataset,
)


def test_dataset_uses_p95_and_external_tokens(issue13_artifact: Path) -> None:
    dataset = load_issue13_dataset(issue13_artifact, percentile="p95")

    assert dataset.percentile == "p95"
    assert dataset.requests_per_case == 8
    assert len(dataset.decision_samples) == 7  # 2 wide CPU + 2 wide FS + 3 short CPU

    cpu_256 = next(
        sample
        for sample in dataset.decision_samples
        if sample.source == "cpu_primary" and sample.requested_tokens == 256
    )
    assert cpu_256.external_tokens == 232
    assert cpu_256.actual_recompute_ms == pytest.approx(25.0)
    assert cpu_256.actual_restore_ms == pytest.approx(21.0)


def test_dataset_retains_repeat_directions_and_exclusions(
    issue13_artifact: Path,
) -> None:
    dataset = load_issue13_dataset(issue13_artifact)

    assert [item["requested_tokens"] for item in dataset.repeat_direction_checks] == [
        192,
        216,
    ]
    assert all(item["all_restore_faster"] for item in dataset.repeat_direction_checks)
    assert {item["reason"] for item in dataset.excluded_samples} == {
        "invalid_cpu_restore_provenance",
        "workload_generation_failure",
    }
```

- [ ] **Step 3: Add RED duplicate-median and promotion-inheritance tests**

```python
def test_derive_profile_aggregates_duplicate_external_tokens_by_median(
    issue13_artifact: Path,
) -> None:
    dataset = load_issue13_dataset(issue13_artifact)
    before = {
        "mode": "shadow",
        "ewma_alpha": 0.2,
        "sample_scale_min": 0.25,
        "sample_scale_max": 4.0,
        "profile": {
            "recompute_ms": {256: 26.0},
            "tiers": {
                "cpu_primary": {"restore_ms": {256: 20.0}},
                "filesystem": {
                    "restore_ms": {256: 30.0},
                    "promotion_ms": {256: 12.0},
                },
            },
        },
    }

    calibrated = derive_calibrated_profile(dataset, before)

    assert calibrated["profile"]["recompute_ms"][192] == pytest.approx(25.0)
    assert calibrated["profile"]["tiers"]["cpu_primary"]["restore_ms"][192] == pytest.approx(22.2)
    assert calibrated["profile"]["tiers"]["filesystem"]["promotion_ms"] == {
        256: 12.0
    }
```

The duplicate values above are medians of the 216/224 P95 samples at the shared 192-external-token point: recompute `(24.8 + 25.2) / 2 = 25.0`, CPU restore `(22.1 + 22.3) / 2 = 22.2`.

- [ ] **Step 4: Implement profile-artifact validation and dataset types**

Create `benchmarks/cache/cost_model_calibration.py` beginning with:

```python
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Percentile = Literal["p50", "p95", "p99"]
Source = Literal["cpu_primary", "secondary:filesystem"]


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


def _load_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def load_profile_artifact(path: Path) -> dict[str, Any]:
    raw = _load_json_object(path)
    profile = raw.get("cache_cost_model")
    if not isinstance(profile, dict):
        raise ValueError("profile artifact requires cache_cost_model mapping")
    if profile.get("mode") != "shadow":
        raise ValueError("profile artifact cache_cost_model.mode must be shadow")
    if not isinstance(profile.get("profile"), dict):
        raise ValueError("profile artifact requires cache_cost_model.profile")
    return profile
```

- [ ] **Step 5: Implement strict external-token conversion and sample extraction**

Add helpers with explicit validation:

```python
def _external_tokens_per_request(total: object, requests_per_case: int) -> int:
    if type(total) is not int or total <= 0:
        raise ValueError("external_kv_tokens must be a positive integer")
    if total % requests_per_case:
        raise ValueError(
            "external_kv_tokens must divide evenly by scope.requests_per_case"
        )
    return total // requests_per_case


def _latency(row: Mapping[str, Any], field: str, percentile: Percentile) -> float:
    values = row.get(field)
    if not isinstance(values, Mapping):
        raise ValueError(f"missing {field}")
    value = values.get(percentile)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"invalid {field}.{percentile}")
    return float(value)
```

`load_issue13_dataset()` must:

1. accept only `p50`, `p95`, `p99`;
2. require positive integer `scope.requests_per_case`;
3. create **two** decision samples per `wide_curve` row: `cpu_primary` and `secondary:filesystem` using the same paired recompute latency and external-token coordinate;
4. create one `cpu_primary` decision sample per `cpu_crossover_points` row;
5. sort decision samples by `(requested_tokens, source)` for deterministic output;
6. convert each `boundary_repeats.<requested>.p95_delta_ms` list into:

```python
{
    "requested_tokens": requested,
    "delta_ms": [float, ...],
    "all_restore_faster": all(delta < 0 for delta in values),
    "all_recompute_faster": all(delta > 0 for delta in values),
}
```

1. add one exclusion record for the invalid CPU sweep and one for the 208 workload failure with reasons exactly `invalid_cpu_restore_provenance` and `workload_generation_failure`.

- [ ] **Step 6: Implement median profile derivation**

Add:

```python
def _median_curve(points: list[tuple[int, float]]) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for tokens, value in points:
        grouped[tokens].append(value)
    return {
        tokens: float(statistics.median(grouped[tokens]))
        for tokens in sorted(grouped)
    }
```

`derive_calibrated_profile()` must:

- build recompute points from unique `(requested_tokens, external_tokens, actual_recompute_ms)` anchors so wide CPU/FS pairing does not double-count recompute;
- build CPU restore points from `source == "cpu_primary"`;
- build filesystem restore points from `source == "secondary:filesystem"`;
- carry `mode`, `ewma_alpha`, `sample_scale_min`, and `sample_scale_max` from `before_profile`;
- carry only the existing filesystem `promotion_ms` curve from `before_profile`; do not synthesize a new promotion curve from #13;
- return integer token keys in Python; JSON serialization may stringify them naturally.

Core assembly:

```python
return {
    "mode": "shadow",
    "ewma_alpha": float(before_profile.get("ewma_alpha", 0.2)),
    "sample_scale_min": float(before_profile.get("sample_scale_min", 0.25)),
    "sample_scale_max": float(before_profile.get("sample_scale_max", 4.0)),
    "profile": {
        "recompute_ms": _median_curve(recompute_points),
        "tiers": {
            "cpu_primary": {"restore_ms": _median_curve(cpu_points)},
            "filesystem": {
                "restore_ms": _median_curve(filesystem_points),
                "promotion_ms": dict(before_fs["promotion_ms"]),
            },
        },
    },
}
```

- [ ] **Step 7: Run focused tests**

```bash
python -m pytest -q benchmarks/cache/tests/test_cost_model_calibration.py
```

Expected: all Task 1-2 tests PASS.

- [ ] **Step 8: Run local syntax/style checks for the new pure module**

```bash
python -m compileall -q benchmarks/cache/cost_model_calibration.py
python -m ruff check \
  benchmarks/cache/cost_model_calibration.py \
  benchmarks/cache/tests/test_cost_model_calibration.py
python -m ruff format --check \
  benchmarks/cache/cost_model_calibration.py \
  benchmarks/cache/tests/test_cost_model_calibration.py
```

Expected: all exit 0.

- [ ] **Step 9: Commit the pure dataset/profile implementation**

```bash
git add \
  benchmarks/cache/cost_model_calibration.py \
  benchmarks/cache/tests/test_cost_model_calibration.py
git commit -m "feat: derive cache cost calibration profile"
```

---

### Task 3: Evaluate Before/After Profiles Through the Real Cost Model and Add the CLI

**Files:**

- Modify: `benchmarks/cache/cost_model_calibration.py`
- Create: `benchmarks/cache/evaluate_cost_model.py`
- Modify: `benchmarks/cache/tests/test_cost_model_calibration.py`

**Interfaces:**

- Consumes: `CalibrationDataset`, `load_profile_artifact()`, `derive_calibrated_profile()` from Task 2 and the existing `OffloadCostModel.from_extra_config()` / `LoadProvenance` behavior.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class SampleEvaluation: ...


def evaluate_profile(
    dataset: CalibrationDataset,
    profile: Mapping[str, Any],
) -> dict[str, Any]: ...


def build_calibration_result(
    dataset: CalibrationDataset,
    before_profile: Mapping[str, Any],
    after_profile: Mapping[str, Any],
    *,
    mape_threshold_percent: float = 15.0,
    decision_accuracy_threshold: float = 0.95,
    boundary_margin_ms: float = 1.0,
) -> dict[str, Any]: ...
```

CLI:

```text
python benchmarks/cache/evaluate_cost_model.py \
  --input <issue13.json> \
  --before-profile <issue12-profile.json> \
  --percentile p95 \
  --output <result.json> \
  [--check]
```

- [ ] **Step 1: Add a pure-module loader for the real cost model implementation**

To keep benchmark unit tests independent of importing the full `vllm` package, mirror the existing `tests/v1/kv_offload/test_cost_model.py` technique inside the calibration module:

```python
import importlib.util
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COST_MODEL_PATH = _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "cost_model.py"
_SPEC = importlib.util.spec_from_file_location(
    "vllm_cost_model_for_calibration", _COST_MODEL_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"unable to load cost model: {_COST_MODEL_PATH}")
_COST_MODEL = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_SPEC.name, _COST_MODEL)
_SPEC.loader.exec_module(_COST_MODEL)

LoadProvenance = _COST_MODEL.LoadProvenance
OffloadCostModel = _COST_MODEL.OffloadCostModel
```

This is evaluation tooling only; no production import path changes are made.

- [ ] **Step 2: Add RED per-sample evaluation tests**

Append:

```python
from benchmarks.cache.cost_model_calibration import evaluate_profile


def test_evaluate_profile_scores_costs_decisions_and_boundary(
    issue13_artifact: Path,
) -> None:
    dataset = load_issue13_dataset(issue13_artifact)
    before = {
        "mode": "shadow",
        "profile": {
            "recompute_ms": {168: 22.0, 192: 25.0, 232: 25.0, 512: 45.0},
            "tiers": {
                "cpu_primary": {
                    "restore_ms": {168: 21.8, 192: 22.2, 232: 21.0, 512: 23.0}
                },
                "filesystem": {"restore_ms": {232: 31.0, 512: 59.0}},
            },
        },
    }

    result = evaluate_profile(dataset, before)

    sample_192 = next(
        row
        for row in result["samples"]
        if row["source"] == "cpu_primary" and row["requested_tokens"] == 192
    )
    assert sample_192["actual_preferred"] == "restore"
    assert sample_192["predicted_preferred"] == "restore"
    assert sample_192["boundary_sensitive"] is True
    assert sample_192["actual_margin_ms"] == pytest.approx(-0.2)
    assert result["aggregate"]["decision_total"] == 7
```

- [ ] **Step 3: Add RED unique-recompute and macro-MAPE tests**

```python
def test_recompute_mape_counts_unique_requested_anchors_once(
    issue13_artifact: Path,
) -> None:
    dataset = load_issue13_dataset(issue13_artifact)
    profile = derive_calibrated_profile(
        dataset,
        {
            "mode": "shadow",
            "profile": {
                "recompute_ms": {256: 1.0},
                "tiers": {
                    "cpu_primary": {"restore_ms": {256: 1.0}},
                    "filesystem": {
                        "restore_ms": {256: 1.0},
                        "promotion_ms": {256: 1.0},
                    },
                },
            },
        },
    )
    result = evaluate_profile(dataset, profile)

    # Requested anchors are 192, 216, 224, 256, 512: five recompute errors,
    # even though wide anchors create both CPU and filesystem decisions.
    assert result["aggregate"]["recompute_sample_count"] == 5
    assert result["aggregate"]["cpu_restore_sample_count"] == 5
    assert result["aggregate"]["tiered_fs_restore_sample_count"] == 2
    expected = (
        result["aggregate"]["recompute_mape_percent"]
        + result["aggregate"]["cpu_restore_mape_percent"]
        + result["aggregate"]["tiered_fs_restore_mape_percent"]
    ) / 3.0
    assert result["aggregate"]["principal_macro_mape_percent"] == pytest.approx(
        expected
    )
```

- [ ] **Step 4: Implement sample scoring using the actual `OffloadCostModel`**

For each `DecisionSample`, call:

```python
model = OffloadCostModel.from_extra_config({"cache_cost_model": dict(profile)})
if model is None:
    raise ValueError("evaluation profile must enable shadow mode")

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
```

For each row compute:

```python
actual_margin = sample.actual_restore_ms - sample.actual_recompute_ms
predicted_margin = decision.restore_estimate_ms - decision.recompute_estimate_ms
actual_preferred = "restore" if actual_margin < 0 else "recompute"
predicted_preferred = decision.preferred
```

Store actual/predicted costs, absolute errors, relative errors, margins, decision correctness, confidence, runtime scale, and `boundary_sensitive = abs(actual_margin) <= 1.0`.

- [ ] **Step 5: Implement aggregate metrics without recompute double-counting**

Use a unique key `(requested_tokens, external_tokens)` for recompute scoring. For two decision rows sharing the same paired recompute sample, assert the actual and predicted recompute values match; count them once.

Use:

```python
def _mape(errors: list[float]) -> float:
    if not errors:
        raise ValueError("cannot calculate MAPE without samples")
    return 100.0 * sum(errors) / len(errors)
```

Aggregate fields must include:

```text
recompute_sample_count
cpu_restore_sample_count
tiered_fs_restore_sample_count
recompute_mape_percent
cpu_restore_mape_percent
tiered_fs_restore_mape_percent
principal_macro_mape_percent
decision_correct
decision_total
decision_accuracy
```

- [ ] **Step 6: Add RED before/after acceptance tests**

Use a deliberately bad one-point CPU before profile and the derived profile:

```python
from benchmarks.cache.cost_model_calibration import build_calibration_result


def test_build_result_reports_before_after_and_acceptance(
    issue13_artifact: Path,
) -> None:
    dataset = load_issue13_dataset(issue13_artifact)
    before = {
        "mode": "shadow",
        "ewma_alpha": 0.2,
        "sample_scale_min": 0.25,
        "sample_scale_max": 4.0,
        "profile": {
            "recompute_ms": {256: 25.0, 512: 45.0},
            "tiers": {
                "cpu_primary": {"restore_ms": {512: 23.0}},
                "filesystem": {
                    "restore_ms": {256: 31.0, 512: 59.0},
                    "promotion_ms": {256: 10.0, 512: 20.0},
                },
            },
        },
    }
    after = derive_calibrated_profile(dataset, before)

    result = build_calibration_result(
        dataset,
        before,
        after,
        mape_threshold_percent=15.0,
        decision_accuracy_threshold=0.95,
    )

    assert result["before"]["aggregate"]["decision_accuracy"] < 1.0
    assert result["after"]["aggregate"]["decision_accuracy"] == 1.0
    assert result["acceptance"]["passed"] is True
    assert result["acceptance"]["decision_total"] == 7
```

- [ ] **Step 7: Implement calibration result and supplemental repeat checks**

`build_calibration_result()` returns deterministic plain JSON-compatible data with:

```python
{
    "schema_version": 1,
    "source_artifact": dataset.source_artifact,
    "percentile": dataset.percentile,
    "acceptance_thresholds": {
        "decision_accuracy_min": 0.95,
        "principal_macro_mape_percent_max": 15.0,
        "boundary_margin_ms": 1.0,
    },
    "before_profile": before_profile,
    "calibrated_profile": after_profile,
    "before": evaluate_profile(dataset, before_profile),
    "after": evaluate_profile(dataset, after_profile),
    "repeat_direction_checks": list(dataset.repeat_direction_checks),
    "excluded_samples": list(dataset.excluded_samples),
    "acceptance": {...},
}
```

Acceptance uses only `after`:

```python
accuracy_ok = after_agg["decision_accuracy"] >= decision_accuracy_threshold
mape_ok = (
    after_agg["principal_macro_mape_percent"] <= mape_threshold_percent
)
passed = accuracy_ok and mape_ok
```

Do not relax acceptance for boundary-sensitive samples.

- [ ] **Step 8: Add RED CLI tests**

Append tests that import `benchmarks.cache.evaluate_cost_model` only after the file exists in the next step:

```python
def test_cli_writes_deterministic_json_and_check_status(
    issue13_artifact: Path, tmp_path: Path
) -> None:
    from benchmarks.cache import evaluate_cost_model

    before_path = tmp_path / "before.json"
    before_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache_cost_model": {
                    "mode": "shadow",
                    "profile": {
                        "recompute_ms": {"256": 25.0, "512": 45.0},
                        "tiers": {
                            "cpu_primary": {"restore_ms": {"512": 23.0}},
                            "filesystem": {
                                "restore_ms": {"256": 31.0, "512": 59.0},
                                "promotion_ms": {"256": 10.0, "512": 20.0},
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    rc = evaluate_cost_model.main(
        [
            "--input",
            str(issue13_artifact),
            "--before-profile",
            str(before_path),
            "--percentile",
            "p95",
            "--output",
            str(output),
            "--check",
        ]
    )

    assert rc == 0
    first = output.read_text(encoding="utf-8")
    assert json.loads(first)["acceptance"]["passed"] is True

    rc2 = evaluate_cost_model.main(
        [
            "--input",
            str(issue13_artifact),
            "--before-profile",
            str(before_path),
            "--percentile",
            "p95",
            "--output",
            str(output),
        ]
    )
    assert rc2 == 0
    assert output.read_text(encoding="utf-8") == first
```

- [ ] **Step 9: Implement the thin CLI**

Create `benchmarks/cache/evaluate_cost_model.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.cache.cost_model_calibration import (
    build_calibration_result,
    derive_calibrated_profile,
    load_issue13_dataset,
    load_profile_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate KV offload shadow cost calibration against #13 data"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--before-profile", type=Path, required=True)
    parser.add_argument(
        "--percentile", choices=("p50", "p95", "p99"), default="p95"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_issue13_dataset(args.input, percentile=args.percentile)
    before = load_profile_artifact(args.before_profile)
    after = derive_calibrated_profile(dataset, before)
    result = build_calibration_result(dataset, before, after)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    after_agg = result["after"]["aggregate"]
    print(
        "after: "
        f"decision={after_agg['decision_correct']}/{after_agg['decision_total']} "
        f"accuracy={after_agg['decision_accuracy']:.3f} "
        f"macro_mape={after_agg['principal_macro_mape_percent']:.3f}% "
        f"passed={result['acceptance']['passed']}"
    )
    if args.check and not result["acceptance"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Keep stdout to one compact summary line; detailed data belong in the output JSON.

- [ ] **Step 10: Run Task 3 focused tests**

```bash
python -m pytest -q benchmarks/cache/tests/test_cost_model_calibration.py
```

Expected: PASS.

- [ ] **Step 11: Run compile/Ruff checks**

```bash
python -m compileall -q \
  benchmarks/cache/cost_model_calibration.py \
  benchmarks/cache/evaluate_cost_model.py
python -m ruff check \
  benchmarks/cache/cost_model_calibration.py \
  benchmarks/cache/evaluate_cost_model.py \
  benchmarks/cache/tests/test_cost_model_calibration.py
python -m ruff format --check \
  benchmarks/cache/cost_model_calibration.py \
  benchmarks/cache/evaluate_cost_model.py \
  benchmarks/cache/tests/test_cost_model_calibration.py
```

Expected: all exit 0.

- [ ] **Step 12: Commit evaluator + CLI**

```bash
git add \
  benchmarks/cache/cost_model_calibration.py \
  benchmarks/cache/evaluate_cost_model.py \
  benchmarks/cache/tests/test_cost_model_calibration.py
git commit -m "feat: evaluate cache cost calibration"
```

---

### Task 4: Add Calibrated Decision Regressions and EWMA Convergence Tests

**Files:**

- Modify: `tests/v1/kv_offload/test_cost_model.py`

**Interfaces:**

- Consumes: unchanged `CostCurve`, `OffloadCostModel`, `LoadProvenance`, and existing #12 EWMA behavior.
- Produces: regression evidence that Phase 1 needs no runtime formula change and that EWMA converges/isolation/clamp semantics remain correct.

- [ ] **Step 1: Add a calibrated P95 test profile on the runtime external-token axis**

Append a test-only profile using representative #13-derived P95 points. Keep this profile local to tests; do not make it a runtime default:

```python
CALIBRATED_P95_PROFILE = {
    "cache_cost_model": {
        "mode": "shadow",
        "ewma_alpha": 0.2,
        "sample_scale_min": 0.25,
        "sample_scale_max": 4.0,
        "profile": {
            "recompute_ms": {
                104: 19.660,
                168: 22.186,
                192: 25.082,
                232: 26.663,
                512: 44.813,
                1024: 81.258,
                2016: 152.433,
                4088: 309.140,
            },
            "tiers": {
                "cpu_primary": {
                    "restore_ms": {
                        104: 21.220,
                        168: 21.830,
                        192: 22.212,
                        232: 21.872,
                        512: 23.057,
                        1024: 24.687,
                        2016: 29.173,
                        4088: 35.213,
                    }
                },
                "filesystem": {
                    "restore_ms": {
                        232: 36.007,
                        512: 59.159,
                        1024: 101.799,
                        2016: 320.793,
                        4088: 648.235,
                    },
                    "promotion_ms": PROFILE["cache_cost_model"]["profile"][
                        "tiers"
                    ]["filesystem"]["promotion_ms"],
                },
            },
        },
    }
}
```

The `192` values are the medians of requested 216 and 224 anchors sharing 192 external tokens.

- [ ] **Step 2: Add calibrated decision regression tests**

```python
def _calibrated_model() -> Any:
    model = OffloadCostModel.from_extra_config(CALIBRATED_P95_PROFILE)
    assert model is not None
    return model


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (104, "recompute"),
        (168, "restore"),
        (192, "restore"),
        (232, "restore"),
        (512, "restore"),
        (1024, "restore"),
        (2016, "restore"),
        (4088, "restore"),
    ],
)
def test_calibrated_cpu_p95_decisions(tokens: int, expected: str) -> None:
    decision = _calibrated_model().shadow_decide(
        LoadProvenance(
            source="cpu_primary",
            external_tokens=tokens,
            secondary_promoted_tokens=0,
            sources=("cpu_primary",),
            confidence="high",
        )
    )
    assert decision is not None
    assert decision.preferred == expected


@pytest.mark.parametrize("tokens", [232, 512, 1024, 2016, 4088])
def test_calibrated_filesystem_p95_still_prefers_recompute(tokens: int) -> None:
    decision = _calibrated_model().shadow_decide(
        LoadProvenance(
            source="secondary:filesystem",
            external_tokens=tokens,
            secondary_promoted_tokens=tokens,
            sources=("secondary:filesystem",),
            confidence="high",
        )
    )
    assert decision is not None
    assert decision.preferred == "recompute"
```

- [ ] **Step 3: Add EWMA stationary-convergence RED tests**

Use the existing #12 profile so seeded promotion curves remain unchanged:

```python
def test_ewma_converges_monotonically_toward_stationary_scale() -> None:
    model = _profile_model()
    observed_ms = 81.458 * 2.0
    scales = []

    for _ in range(5):
        observation = model.observe_secondary_promotion(
            "filesystem", 1024, observed_ms
        )
        assert observation is not None
        scales.append(observation.runtime_scale)

    assert scales == sorted(scales)
    assert all(scale < 2.0 for scale in scales)
    assert scales == pytest.approx([1.2, 1.36, 1.488, 1.5904, 1.67232])


def test_ewma_stable_observations_have_diminishing_updates() -> None:
    model = _profile_model()
    observed_ms = 81.458 * 2.0
    scales = []

    for _ in range(5):
        observation = model.observe_secondary_promotion(
            "filesystem", 1024, observed_ms
        )
        assert observation is not None
        scales.append(observation.runtime_scale)

    increments = [right - left for left, right in zip(scales, scales[1:])]
    assert increments[0] > increments[1] > increments[2] > increments[3] > 0
```

Existing tests already cover bucket isolation, clamp behavior, and no updates for CPU/no-promotion tiers; do not duplicate them unnecessarily.

- [ ] **Step 4: Run the pure cost-model test file**

```bash
python -m pytest -q tests/v1/kv_offload/test_cost_model.py
```

Expected: PASS without modifying `vllm/v1/kv_offload/cost_model.py`.

- [ ] **Step 5: Run targeted static checks**

```bash
python -m ruff check tests/v1/kv_offload/test_cost_model.py
python -m ruff format --check tests/v1/kv_offload/test_cost_model.py
git --no-pager diff --check
```

Expected: all exit 0.

- [ ] **Step 6: Commit regression/EWMA tests**

```bash
git add tests/v1/kv_offload/test_cost_model.py
git commit -m "test: validate calibrated shadow cost decisions"
```

---

### Task 5: Run the Real #13 Phase 1 Gate and Record #14 Validation Evidence

**Files:**

- Create: `docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json`
- Create: `docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.md`
- Modify only if evidence requires: none in Phase 1; production runtime files remain untouched.

**Interfaces:**

- Consumes:
    - `docs/engineering/validation/2026-08-10-issue13-restore-recompute-crossover.json`
    - `benchmarks/cache/profiles/issue12-shadow-cost-baseline.json`
    - `benchmarks/cache/evaluate_cost_model.py`
- Produces: final machine-readable and human-readable #14 baseline calibration evidence.

- [ ] **Step 1: Execute the real P95 evaluator in check mode**

Run from repository root:

```bash
python benchmarks/cache/evaluate_cost_model.py \
  --input docs/engineering/validation/2026-08-10-issue13-restore-recompute-crossover.json \
  --before-profile benchmarks/cache/profiles/issue12-shadow-cost-baseline.json \
  --percentile p95 \
  --output /tmp/issue14-calibration.json \
  --check
```

Expected compact stdout shape:

```text
after: decision=14/14 accuracy=1.000 macro_mape=<value <= 15.000>% passed=True
```

Expected exit code: `0`.

**Hard gate:** if exit code is nonzero, or decision correctness is not 14/14, or macro-MAPE exceeds 15%, do not edit `CostCurve` or `OffloadCostModel` ad hoc. Preserve `/tmp/issue14-calibration.json`, summarize residuals by source/token, and stop implementation for an explicit Phase 2 design review.

- [ ] **Step 2: Verify expected before-calibration failure mode from the real result**

Run a compact extraction only:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path('/tmp/issue14-calibration.json')
d = json.loads(p.read_text())
for name in ('before', 'after'):
    a = d[name]['aggregate']
    print(
        name,
        f"decision={a['decision_correct']}/{a['decision_total']}",
        f"accuracy={a['decision_accuracy']:.3f}",
        f"recompute_mape={a['recompute_mape_percent']:.3f}%",
        f"cpu_mape={a['cpu_restore_mape_percent']:.3f}%",
        f"fs_mape={a['tiered_fs_restore_mape_percent']:.3f}%",
        f"macro_mape={a['principal_macro_mape_percent']:.3f}%",
    )

wrong = [
    row for row in d['before']['samples'] if not row['decision_correct']
]
print('before_wrong=', [
    (row['source'], row['requested_tokens'], row['external_tokens'])
    for row in wrong
])
PY
```

Expected qualitative result:

- before profile fails at least the short CPU side because the single 1024 CPU sample proportionally extrapolates below range;
- after profile reaches 14/14;
- after profile principal macro-MAPE <= 15%.

Do not hard-code an expected before MAPE value in tests; it is evidence output, not an acceptance threshold.

- [ ] **Step 3: Copy the machine-readable result into the validation path**

```bash
cp /tmp/issue14-calibration.json \
  docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json
python -m json.tool \
  docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json \
  >/dev/null
```

Expected: JSON validation exits 0.

- [ ] **Step 4: Write the Markdown validation report from the machine-readable result**

Create `docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.md` with these exact sections:

```markdown
# Issue #14: shadow cost model calibration validation

## Executive conclusion
## Source data and environment
## Acceptance criteria
## Before-calibration baseline
## Calibrated external-token P95 profile
## After-calibration metrics
## Per-anchor decision evidence
## Boundary repeat direction checks
## Error-source analysis
## EWMA validation
## Exclusions and provenance caveats
## Commands and checks
## Completion against #14
## Limitations and handoff
```

The report must state explicitly:

- primary metric is P95;
- decision acceptance is 14/14 on this baseline;
- principal macro-MAPE acceptance is <=15%;
- the old #12 CPU curve had one 1024 sample and proportional low-confidence extrapolation;
- calibration uses external tokens, not requested tokens;
- 216/224 collapse to 192 external tokens and use median profile aggregation while both anchors remain scored;
- the 2 GiB CPU sweep is excluded as recompute, not restore;
- 208 is excluded as workload-generation failure;
- filesystem is lower-tier/tiered-fs on container overlay, not proven physical NVMe;
- filesystem promotion EWMA seed curve is inherited from #12 because #13 does not provide a clean per-anchor promotion curve;
- no runtime formula or active execution path was changed in Phase 1;
- #15 owns model/concurrency/hardware generalization; #16 owns active enforcement.

Populate all before/after metrics and per-anchor rows by reading the JSON result; do not retype benchmark values from chat history.

- [ ] **Step 5: Run all focused #14 tests**

```bash
python -m pytest -q \
  benchmarks/cache/tests/test_cost_model_calibration.py \
  tests/v1/kv_offload/test_cost_model.py
```

Expected: PASS.

- [ ] **Step 6: Run focused cache-suite regression tests that exercise nearby benchmark code**

```bash
python -m pytest -q \
  benchmarks/cache/tests/test_run_suite.py \
  benchmarks/cache/tests/test_config.py
```

Expected: PASS except for any already-known unrelated fixture defect explicitly tracked separately; do not modify unrelated #21 work inside #14.

If the known #21 newline fixture still fails on the current baseline, record it as a pre-existing independent failure and continue only after confirming the new #14 tests and changed-file checks are green.

- [ ] **Step 7: Run compile, Ruff, JSON, and diff checks**

```bash
python -m compileall -q \
  benchmarks/cache/cost_model_calibration.py \
  benchmarks/cache/evaluate_cost_model.py
python -m ruff check \
  benchmarks/cache/cost_model_calibration.py \
  benchmarks/cache/evaluate_cost_model.py \
  benchmarks/cache/tests/test_cost_model_calibration.py \
  tests/v1/kv_offload/test_cost_model.py
python -m ruff format --check \
  benchmarks/cache/cost_model_calibration.py \
  benchmarks/cache/evaluate_cost_model.py \
  benchmarks/cache/tests/test_cost_model_calibration.py \
  tests/v1/kv_offload/test_cost_model.py
python -m json.tool \
  benchmarks/cache/profiles/issue12-shadow-cost-baseline.json \
  >/dev/null
python -m json.tool \
  docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json \
  >/dev/null
git --no-pager diff --check
```

Expected: all #14-owned checks exit 0.

- [ ] **Step 8: Verify Phase 1 did not change runtime implementation files**

Run:

```bash
git --no-pager diff --name-only main...HEAD
```

Expected changed-file set contains only:

```text
benchmarks/cache/profiles/issue12-shadow-cost-baseline.json
benchmarks/cache/cost_model_calibration.py
benchmarks/cache/evaluate_cost_model.py
benchmarks/cache/tests/test_cost_model_calibration.py
tests/v1/kv_offload/test_cost_model.py
docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json
docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.md
docs/superpowers/specs/2026-08-10-issue14-shadow-cost-model-calibration-design.md
docs/superpowers/plans/2026-08-10-issue14-shadow-cost-model-calibration.md
```

In particular, there must be no changes to:

```text
vllm/v1/kv_offload/cost_model.py
vllm/v1/kv_offload/tiering/manager.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
```

- [ ] **Step 9: Commit final validation artifacts**

```bash
git add \
  docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json \
  docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.md
git commit -m "docs: record issue 14 cost calibration"
```

- [ ] **Step 10: Fresh completion verification before any PR/Issue completion claim**

Run the real evaluator and focused tests again after the final commit:

```bash
python benchmarks/cache/evaluate_cost_model.py \
  --input docs/engineering/validation/2026-08-10-issue13-restore-recompute-crossover.json \
  --before-profile benchmarks/cache/profiles/issue12-shadow-cost-baseline.json \
  --percentile p95 \
  --output /tmp/issue14-final-check.json \
  --check
python -m pytest -q \
  benchmarks/cache/tests/test_cost_model_calibration.py \
  tests/v1/kv_offload/test_cost_model.py
git --no-pager diff --check
```

Expected:

```text
Evaluator: passed=True, decision=14/14, macro_mape <= 15%
Tests: PASS
Diff check: exit 0
```

Only after this fresh evidence may the implementation be described as complete or proposed for merge.

---

## Plan Self-Review

### Spec coverage

- P95 target and explicit 14/14 / <=15% thresholds: Tasks 3 and 5.
- External-token runtime axis: Task 2.
- Duplicate median aggregation with raw-anchor scoring: Tasks 2 and 3.
- Invalid 2 GiB CPU and 208 failure exclusions: Task 2.
- Boundary repeats supplemental only: Tasks 2, 3, and 5.
- Before/after comparison: Tasks 1, 3, and 5.
- Preserve source-tier distinction: Tasks 2-4.
- Preserve secondary promotion EWMA and inherited promotion curve: Tasks 1, 2, and 4.
- EWMA convergence/isolation/clamp validation: Task 4 plus existing tests.
- Shadow-only runtime invariant: Global Constraints and Task 5 changed-file verification.
- Validation Markdown + JSON: Task 5.
- #15/#16 scope boundary: Task 5 report requirements.
- Evidence-triggered Phase 2 stop condition: Global Constraints and Task 5 hard gate.

### Placeholder scan

The plan contains no `TBD`, `TODO`, or unspecified implementation steps. The only conditional branch is the approved Phase 1 acceptance hard gate: failure explicitly stops implementation for Phase 2 design review rather than authorizing unplanned runtime changes.

### Type/interface consistency

- `DecisionSample`, `CalibrationDataset`, `load_profile_artifact`, `load_issue13_dataset`, `derive_calibrated_profile`, `evaluate_profile`, and `build_calibration_result` are introduced once and consumed consistently by later tasks.
- Profile mappings use the existing `cache_cost_model` schema accepted by `OffloadCostModel.from_extra_config()`.
- Source names match existing runtime semantics: `cpu_primary` and `secondary:filesystem`.
- P95 external-token coordinates match #13 semantics; requested tokens remain provenance only.
