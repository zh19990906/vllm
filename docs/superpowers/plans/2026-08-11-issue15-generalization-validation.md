# Issue #15 Generalization Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce fixed-profile holdout evidence for the Issue #14 KV restore/recompute cost model under one materially contended 7B load condition and one Qwen2.5-14B model-scale condition, then classify each curve as transferable, scale-sensitive, or shape/missing-feature limited without changing active runtime behavior.

**Architecture:** Keep the Issue #14 calibration path unchanged. Add a separate pure generalization module and CLI that consume a frozen Issue #14 profile plus a condition dataset. Add a dataset builder that converts existing cache-benchmark run artifacts into the stable Issue #15 measurement schema with explicit restore provenance. Hardware execution remains shadow-only and uses the existing `benchmarks/cache/run_suite.py` workload/metrics pipeline.

**Tech Stack:** Python 3.11, stdlib `dataclasses`/`json`/`statistics`, existing vLLM `OffloadCostModel`, existing cache benchmark YAML/Pydantic configuration, stdlib `unittest` for Pod-local TDD, GitHub repository-wide pre-commit as authoritative CI.

## Global Constraints

- GitHub is the authoritative remote and write path; the Pod is only for build, focused tests, benchmark, and hardware validation.
- Do not run `git push` from the Pod.
- Do not use `git clean -fd`, `reset --hard`, or checkout operations that can remove the three intentional untracked local YAML files in `/code/vllm`.
- Do not install pytest merely for this work. New focused tests must be runnable with stdlib `unittest` on the Pod; repository CI may also collect them through pytest.
- Issue #15 remains shadow-only. Do not modify scheduler enforcement, matched-token behavior, transfer scheduling, cache contents, or active restore/recompute selection.
- Primary percentile is P95.
- Primary evidence is evaluation of the frozen Issue #14 calibrated profile. New-condition data must be saved before any diagnostic scaling.
- Requested prompt tokens are workload anchors only. Evaluation uses runtime external KV tokens per request.
- Formal requested-token anchors are exactly `128, 192, 256, 1024, 4096` unless a material deviation is first recorded on Issue #15.
- C-load probes concurrency exactly `2 -> 4 -> 8` against a same-session C1 sentinel at 1024 requested tokens and stops at the first materially contended load.
- Material contention means one principal P95 path changes by at least 20%, another by at least 10%, and required restore provenance remains valid.
- Fixed-profile transfer requires valid high-confidence evidence for all three principal curves, decision accuracy >= 95%, principal macro-MAPE <= 15%, no principal curve MAPE > 20%, and no wrong high-confidence decision with absolute actual margin > 1 ms.
- If any principal curve lacks valid high-confidence formal evidence, classify the condition `insufficient_evidence`; do not call it transfer pass.
- Low-confidence samples remain evidence but do not expand Issue #16 eligibility.
- Diagnostic scale is exactly one multiplicative scalar per curve: median of `actual_latency / frozen_predicted_latency`.
- Do not call container-local filesystem evidence physical NVMe unless provenance proves it.
- Do not start Issue #16 active enforcement while #15 evidence is incomplete.
- Any material experiment interpretation change must be commented on Issue #15 before measurements use the changed condition.

---

## File Structure

Implementation should keep responsibilities separated as follows:

- `benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json`
  - immutable frozen-profile artifact extracted byte-for-value from the checked-in Issue #14 calibration result;
  - contains profile provenance and a top-level `cache_cost_model` mapping accepted by existing `load_profile_artifact()`.
- `benchmarks/cache/cost_model_generalization.py`
  - pure Issue #15 dataset loading, frozen-profile evaluation, fixed transfer gate, low-confidence partitioning, and one-scalar failure diagnostics;
  - no benchmark process execution and no active runtime changes.
- `benchmarks/cache/evaluate_cost_model_generalization.py`
  - deterministic CLI wrapper around the pure module;
  - never derives or mutates a profile from the new-condition dataset.
- `benchmarks/cache/build_generalization_dataset.py`
  - converts completed `run_suite.py` artifacts into the Issue #15 condition schema;
  - validates workload identity and restore provenance before accepting samples.
- `benchmarks/cache/tests/test_cost_model_generalization.py`
  - stdlib `unittest` coverage for frozen evaluation, high/low-confidence gate semantics, insufficient evidence, and scale/shape classification.
- `benchmarks/cache/tests/test_build_generalization_dataset.py`
  - stdlib `unittest` coverage for run pairing, workload SHA fairness, external-token normalization, CPU transfer evidence, filesystem lookup evidence, and invalid-sample recording.
- `benchmarks/cache/configs/issue15-7b-load-sentinel-cpu.yaml`
  - 1024-token, C1/C2/C4/C8 CPU-primary provenance sentinel template with 8 GiB CPU tier.
- `benchmarks/cache/configs/issue15-7b-load-sentinel-fs.yaml`
  - 1024-token, C1/C2/C4/C8 lower-tier filesystem sentinel template with 2 GiB CPU tier.
- `benchmarks/cache/configs/issue15-14b-formal-cpu.yaml`
  - 14B/C1 formal-anchor CPU-primary template, initially 8 GiB CPU tier; provenance preflight decides whether a capacity adjustment is required before formal measurement.
- `benchmarks/cache/configs/issue15-14b-formal-fs.yaml`
  - 14B/C1 formal-anchor lower-tier filesystem template with 2 GiB CPU tier.
- `docs/engineering/validation/2026-08-11-issue15-generalization-validation.json`
  - checked-in machine-readable final results and classifications.
- `docs/engineering/validation/2026-08-11-issue15-generalization-validation.md`
  - concise interpretation, failure boundaries, transferable/environment-specific parameters, and Issue #16 eligibility map.
- `docs/engineering/handoffs/2026-08-11-issue16-active-decision-handoff.md`
  - only created when #15 evidence is complete; states where active-decision design may and may not proceed.

Do not modify `vllm/v1/kv_offload/cost_model.py` unless a test proves the pure evaluator cannot reuse its existing public API. The expected path is no runtime-file change.

---

### Task 1: Create an isolated Issue #15 Pod worktree without touching preserved local YAML

**Files:**
- No repository file changes.
- Worktree target: `/code/vllm-worktrees/issue15-generalization-validation`

**Interfaces:**
- Consumes: live GitHub `main` SHA re-verified immediately before execution.
- Produces: isolated Pod worktree based on the mirror commit that exactly matches live GitHub `main`.

- [ ] **Step 1: Invoke the worktree skill before touching the Pod checkout**

Read `superpowers:using-git-worktrees` and follow its isolation/safety checks. Do not reuse the current `feature/cache-eviction-restore-benchmark` checkout as the implementation worktree.

- [ ] **Step 2: Re-verify live GitHub main and record the expected SHA**

At execution time, query GitHub again. Do not assume the design-time `b43b3d83048f55f00f85e3e7d230d1b98c25b4f9` is still current.

- [ ] **Step 3: On the Pod, verify the preserved files before any fetch/worktree operation**

Run a single compact block equivalent to:

```bash
cd /code/vllm || exit 1
printf 'branch='; git branch --show-current
printf 'head='; git rev-parse HEAD
for f in \
  benchmarks/cache/configs/local-pressure-cpu-hit.yaml \
  benchmarks/cache/configs/local-pressure.yaml \
  benchmarks/cache/configs/local-smoke.yaml
do
  test -f "$f" && printf 'PRESENT %s\n' "$f" || printf 'MISSING %s\n' "$f"
done
git status --short --branch
```

Expected: all three local YAML files remain present and untracked.

- [ ] **Step 4: Fetch the Pod mirror without changing the current worktree**

Use the configured Pod remote only. Fetching is allowed; pushing is not.

```bash
cd /code/vllm || exit 1
git fetch origin main
printf 'origin/main='; git rev-parse origin/main
```

Compare `origin/main` to the live GitHub SHA from Step 2. If they differ, stop execution and record the mirror mismatch; do not build a worktree from stale code.

- [ ] **Step 5: Create the isolated worktree from the verified mirror SHA**

Use a fresh local branch name and never checkout the old `/code/vllm` worktree:

```bash
cd /code/vllm || exit 1
mkdir -p /code/vllm-worktrees
git worktree add -b local/issue15-generalization-validation \
  /code/vllm-worktrees/issue15-generalization-validation origin/main
cd /code/vllm-worktrees/issue15-generalization-validation || exit 1
printf 'branch='; git branch --show-current
printf 'head='; git rev-parse HEAD
git status --short --branch
```

Expected: clean new worktree at the verified `origin/main` SHA.

---

### Task 2: Freeze the Issue #14 calibrated profile as a first-class artifact

**Files:**
- Create: `benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json`
- Test: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:**
- Consumes: `docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json["calibrated_profile"]` and existing `load_profile_artifact(path)`.
- Produces: a profile artifact whose `cache_cost_model` mapping is exactly equal to the checked-in Issue #14 `calibrated_profile` mapping.

- [ ] **Step 1: Write the failing frozen-profile fidelity test**

Create a stdlib test module beginning with:

```python
import json
import unittest
from pathlib import Path

from benchmarks.cache.cost_model_calibration import load_profile_artifact


_REPO_ROOT = Path(__file__).resolve().parents[3]


class FrozenProfileArtifactTests(unittest.TestCase):
    def test_issue14_frozen_profile_matches_calibration_result(self) -> None:
        result = json.loads(
            (_REPO_ROOT / "docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json")
            .read_text(encoding="utf-8")
        )
        artifact_path = (
            _REPO_ROOT / "benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json"
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact["cache_cost_model"], result["calibrated_profile"])
        self.assertEqual(load_profile_artifact(artifact_path), result["calibrated_profile"])
        self.assertEqual(artifact["provenance"]["issue"], 14)
        self.assertEqual(artifact["provenance"]["profile_role"], "frozen_holdout_seed")
```

- [ ] **Step 2: Run the test to verify RED without pytest**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
```

Expected: ERROR because `issue14-shadow-cost-calibrated.json` does not exist.

- [ ] **Step 3: Create the frozen artifact by exact extraction, not manual retyping**

Use a one-shot Python command in the worktree to read the calibration result and write:

```json
{
  "schema_version": 1,
  "name": "issue14-shadow-cost-calibrated",
  "provenance": {
    "issue": 14,
    "profile_role": "frozen_holdout_seed",
    "source_artifact": "docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json",
    "source_field": "calibrated_profile",
    "percentile": "p95",
    "note": "Frozen Issue #14 shadow profile for Issue #15 holdout evaluation; not a production default."
  },
  "cache_cost_model": { ...exact calibrated_profile mapping... }
}
```

The script must use `json.dumps(..., indent=2, sort_keys=True) + "\n"` and must copy the mapping from the source JSON programmatically.

- [ ] **Step 4: Run the focused test to verify GREEN**

Run the same `unittest discover` command. Expected: PASS for the fidelity test.

- [ ] **Step 5: Commit the artifact and initial test locally**

```bash
git add benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "test: freeze issue 14 cost profile for holdout"
```

---

### Task 3: Add a neutral Issue #15 condition dataset loader

**Files:**
- Create: `benchmarks/cache/cost_model_generalization.py`
- Modify: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:**
- Consumes: Issue #15 condition JSON with condition metadata plus per-source rows.
- Produces:
  - `GeneralizationCondition`
  - `load_generalization_condition(path: Path, percentile: str = "p95") -> GeneralizationCondition`
  - an embedded existing `CalibrationDataset` compatible with the existing pure cost evaluator.

- [ ] **Step 1: Write failing tests for runtime-token normalization and metadata retention**

Add tests using `tempfile.TemporaryDirectory()` and an input shaped exactly like:

```python
payload = {
    "schema_version": 1,
    "issue": 15,
    "condition": {
        "id": "c-model",
        "model": "/mnt/model/Qwen2.5-14B-Instruct",
        "served_model": "qwen2.5-14b",
        "concurrency": 1,
        "request_rate": "inf",
        "requests_per_case": 8,
        "tensor_parallel_size": 1,
        "gpu_uuid": "GPU-test",
        "environment_artifact": "/code/results/cache/run/environment.json",
        "run_directories": {
            "recompute": "/code/results/cache/recompute",
            "cpu_primary": "/code/results/cache/cpu",
            "secondary_filesystem": "/code/results/cache/fs",
        },
    },
    "samples": [
        {
            "source": "cpu_primary",
            "requested_tokens": 256,
            "external_kv_tokens_total": 1856,
            "external_tokens_per_request": 232,
            "recompute_ttft_ms": {"p95": 30.0},
            "restore_ttft_ms": {"p95": 24.0},
            "transfer_evidence": {
                "cpu_to_gpu_transfers": 8,
                "cpu_to_gpu_bytes": 1000,
                "tiered_fs_async_lookups": 0,
            },
            "workload": {"measure_sha256": "a", "populate_sha256": "b"},
        },
        {
            "source": "secondary:filesystem",
            "requested_tokens": 256,
            "external_kv_tokens_total": 1856,
            "external_tokens_per_request": 232,
            "recompute_ttft_ms": {"p95": 30.0},
            "restore_ttft_ms": {"p95": 40.0},
            "transfer_evidence": {
                "cpu_to_gpu_transfers": 8,
                "cpu_to_gpu_bytes": 1000,
                "tiered_fs_async_lookups": 8,
            },
            "workload": {"measure_sha256": "a", "populate_sha256": "b"},
        },
    ],
    "excluded_samples": [],
}
```

Assertions must verify:

```python
condition = load_generalization_condition(path)
self.assertEqual(condition.condition_id, "c-model")
self.assertEqual(condition.model, "/mnt/model/Qwen2.5-14B-Instruct")
self.assertEqual(condition.concurrency, 1)
self.assertEqual(condition.dataset.requests_per_case, 8)
self.assertEqual(condition.dataset.decision_samples[0].external_tokens, 232)
```

Also add a rejection test where `external_kv_tokens_total != requests_per_case * external_tokens_per_request`.

- [ ] **Step 2: Run focused tests and verify RED**

Expected: ImportError because `cost_model_generalization.py` and `load_generalization_condition` do not exist.

- [ ] **Step 3: Implement the minimal dataclass and loader**

Use these public definitions:

```python
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
    ...
```

The loader must:

1. require `schema_version == 1` and `issue == 15`;
2. require P95/P50/P99 only, matching the existing calibration loader choices;
3. validate positive integer `requests_per_case`, `external_kv_tokens_total`, and `external_tokens_per_request`;
4. require exact divisibility/equality between total and per-request external tokens;
5. accept only `cpu_primary` and `secondary:filesystem` sources;
6. convert each valid row to existing `DecisionSample` using `external_tokens_per_request`;
7. build an existing `CalibrationDataset` with empty repeat checks and the condition's excluded-sample records;
8. preserve the original per-row transfer/workload metadata separately for reporting.

- [ ] **Step 4: Run focused tests and verify GREEN**

Use stdlib unittest only.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "feat: load cost model generalization conditions"
```

---

### Task 4: Implement frozen-profile evaluation and the pre-registered transfer gate

**Files:**
- Modify: `benchmarks/cache/cost_model_generalization.py`
- Modify: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:**
- Consumes: `GeneralizationCondition`, a supplied frozen `cache_cost_model` mapping.
- Produces:
  - `evaluate_frozen_condition(condition, profile, *, profile_identity) -> dict[str, Any]`
  - result schema with separate `high_confidence`, `low_confidence`, aggregate errors, and fixed transfer classification.

- [ ] **Step 1: Write RED tests proving the supplied profile is evaluated without derivation**

Create a synthetic condition where the supplied profile intentionally predicts a wrong CPU decision. Assert the returned result preserves that wrong prediction rather than fitting it away:

```python
result = evaluate_frozen_condition(
    condition,
    frozen_profile,
    profile_identity="fixture",
)
self.assertEqual(result["profile_identity"], "fixture")
self.assertEqual(result["evaluation_mode"], "frozen_profile_holdout")
self.assertFalse(result["samples"][0]["decision_correct"])
```

Also assert the function never returns a `calibrated_profile` key.

- [ ] **Step 2: Write RED gate tests for all pre-registered conditions**

Cover these exact cases:

1. all three principal curves have high-confidence evidence, accuracy 1.0, macro <= 15%, each curve <= 20% -> `fixed_profile_transfer_pass`;
2. one principal curve missing high-confidence evidence -> `insufficient_evidence`;
3. macro <= 15% but CPU restore MAPE > 20% -> `fixed_profile_transfer_fail`;
4. wrong high-confidence decision with `abs(actual_margin_ms) > 1.0` -> `fixed_profile_transfer_fail`;
5. only wrong decision is boundary-sensitive with `abs(actual_margin_ms) <= 1.0`, but all error gates otherwise pass -> gate is decided by accuracy threshold, not by silently dropping the row;
6. low-confidence rows are reported separately and do not create principal-curve evidence for the pass gate.

- [ ] **Step 3: Run focused tests and verify RED**

Expected: missing `evaluate_frozen_condition` / classification helpers.

- [ ] **Step 4: Implement using the existing pure evaluator**

Call the existing `evaluate_profile(condition.dataset, profile)` exactly once. Do not call `derive_calibrated_profile()`.

Add constants:

```python
DECISION_ACCURACY_MIN = 0.95
PRINCIPAL_MACRO_MAPE_MAX = 15.0
PRINCIPAL_CURVE_MAPE_MAX = 20.0
BOUNDARY_MARGIN_MS = 1.0
```

Partition rows by `row["confidence"]`. Recompute the high-confidence aggregate rather than reusing the all-row aggregate, because the primary gate is explicitly high-confidence-only. Deduplicate recompute errors by `(requested_tokens, external_tokens)` exactly as the existing evaluator does.

Return both:

- all-sample raw evaluation for evidence;
- high-confidence gate aggregate;
- low-confidence rows and their descriptive error summary;
- `classification` in `{fixed_profile_transfer_pass, fixed_profile_transfer_fail, insufficient_evidence}`;
- frozen threshold values in the JSON output.

- [ ] **Step 5: Run focused tests and verify GREEN**

- [ ] **Step 6: Commit**

```bash
git add benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "feat: evaluate frozen cost profiles on holdout data"
```

---

### Task 5: Add one-scalar transfer/scale/shape diagnostics without altering the primary verdict

**Files:**
- Modify: `benchmarks/cache/cost_model_generalization.py`
- Modify: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:**
- Consumes: saved frozen evaluation rows.
- Produces: `diagnose_curve_scaling(evaluation: Mapping[str, Any]) -> dict[str, Any]`.

- [ ] **Step 1: Write RED tests for three diagnostic classes**

Synthetic row sets must cover:

```text
transferable:
  raw relative errors average <= 15%

environment_specific_scale_candidate:
  actual/predicted ratios are approximately constant at 1.5x,
  raw MAPE > 15%, scalar-residual MAPE <= 15%

curve_shape_or_missing_feature:
  ratios vary materially with token count, for example 1.0x, 1.5x, 2.0x,
  one-scalar residual MAPE remains > 15%
```

- [ ] **Step 2: Run tests and verify RED**

- [ ] **Step 3: Implement exact scalar semantics**

For each principal curve:

```python
scale = statistics.median(
    row["actual_ms"] / row["predicted_ms"] for row in curve_rows
)
residual_mape = 100.0 * statistics.fmean(
    abs((row["predicted_ms"] * scale) - row["actual_ms"]) / row["actual_ms"]
    for row in curve_rows
)
```

Use the same three logical curves as the primary macro metric:

- recompute;
- CPU-primary restore;
- secondary-filesystem restore.

Classification rules are exact:

```text
raw MAPE <= 15% -> transferable
raw MAPE > 15% and residual MAPE <= 15% -> environment_specific_scale_candidate
residual MAPE > 15% -> curve_shape_or_missing_feature
```

Store the scalar and residual error only under a `diagnostics` section. Do not modify `classification` from Task 4 and do not replace predicted values in primary samples.

- [ ] **Step 4: Run tests and verify GREEN**

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "feat: classify cost model scale and shape drift"
```

---

### Task 6: Add a deterministic frozen-profile generalization CLI

**Files:**
- Create: `benchmarks/cache/evaluate_cost_model_generalization.py`
- Modify: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:**
- Consumes: `--input` Issue #15 condition JSON and `--profile` frozen profile artifact.
- Produces: deterministic result JSON and compact one-line summary.

- [ ] **Step 1: Write RED CLI tests**

Add a unittest that calls `main([...])` twice with the same fixture and asserts byte-identical output.

The CLI contract is exactly:

```text
--input PATH      required
--profile PATH    required
--percentile      p50|p95|p99, default p95
--output PATH     required
--diagnose        optional; adds one-scalar diagnostic section after primary evaluation
--check           optional; exits 1 unless classification is fixed_profile_transfer_pass
```

Do not expose CLI flags that change the frozen acceptance thresholds.

Summary format:

```text
holdout: condition=<id> decision=<correct>/<total> accuracy=<x.xxx> macro_mape=<x.xxx>% classification=<classification>
```

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Implement the CLI**

Load the profile only with existing `load_profile_artifact()`. Load the condition with `load_generalization_condition()`. Call `evaluate_frozen_condition()`. If `--diagnose` is present, attach `diagnose_curve_scaling(result)` after the primary result is already built.

Write deterministic JSON with sorted keys, two-space indentation, and a trailing newline.

- [ ] **Step 4: Run focused tests and verify GREEN**

- [ ] **Step 5: Verify the old Issue #14 CLI still has unchanged semantics**

Use direct Python invocation on the checked-in #13 artifact and confirm the existing evaluator still prints the known `after:` summary and produces the original acceptance behavior. This is a compatibility proof, not a new hardware run.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/cache/evaluate_cost_model_generalization.py \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "feat: add frozen profile generalization evaluator"
```

---

### Task 7: Build Issue #15 condition datasets from existing benchmark run artifacts

**Files:**
- Create: `benchmarks/cache/build_generalization_dataset.py`
- Create: `benchmarks/cache/tests/test_build_generalization_dataset.py`

**Interfaces:**
- Consumes three `run_suite.py` run directories for one condition:
  - recompute/no-cache source;
  - CPU-primary source;
  - secondary-filesystem source.
- Produces a schema-version-1 Issue #15 condition JSON accepted by `load_generalization_condition()`.

- [ ] **Step 1: Write synthetic run fixtures with stdlib unittest**

The test must create temporary run directories containing:

- `manifest.json` with model/config/requests-per-case;
- `environment.json`;
- `scenario-results.jsonl` with completed `eviction-restore` records;
- each raw case `metadata.json` containing measure/populate SHA256 values.

Use one requested-token anchor and all three cache modes. The no-cache and restore records must have identical workload SHA values.

- [ ] **Step 2: Write RED tests for accepted provenance**

The builder must accept:

- no-cache recompute with completed native result;
- CPU-primary restore only when external KV tokens are positive and CPU-to-GPU transfer count/bytes are positive;
- filesystem restore only when the same external/CPU-to-GPU evidence is present and the tiering async lookup count is positive.

The tests should use normalized Prometheus delta keys matching the runtime evidence pattern, for example keys whose base names end in:

```text
prompt_tokens_by_source{source="external_kv_transfer"}
kv_offload_size_count{transfer_type="CPU_to_GPU"}
kv_offload_total_bytes{transfer_type="CPU_to_GPU"}
```

and use normalized selected cache evidence for the tiering lookup count when available.

- [ ] **Step 3: Write RED rejection/exclusion tests**

Cover:

1. CPU configured mode but zero external tokens -> excluded with `invalid_cpu_restore_provenance`;
2. CPU external tokens positive but zero CPU-to-GPU transfers -> excluded;
3. filesystem restore with no async lookup evidence -> excluded with `invalid_secondary_restore_provenance`;
4. workload measure/populate SHA mismatch between recompute and restore -> raise a hard fairness error, do not emit a comparable sample;
5. model, concurrency, requested-anchor, or requests-per-case mismatch across paired runs -> hard error;
6. incomplete benchmark record -> excluded with its status/reason.

- [ ] **Step 4: Run tests and verify RED**

- [ ] **Step 5: Implement the builder with these public helpers**

```python
def build_condition_dataset(
    *,
    condition_id: str,
    recompute_run: Path,
    cpu_run: Path,
    filesystem_run: Path,
    percentile: str = "p95",
) -> dict[str, Any]:
    ...


def main(argv: Sequence[str] | None = None) -> int:
    ...
```

CLI contract:

```text
--condition-id ID
--recompute-run PATH
--cpu-run PATH
--filesystem-run PATH
--percentile p50|p95|p99 (default p95)
--output PATH
```

The builder must infer model identity, concurrency, request rate, requests-per-case, TP size, environment artifact, and run paths from manifests/records instead of accepting them as unverified CLI labels.

- [ ] **Step 6: Emit both total and per-request external tokens**

For each accepted restore row write:

```json
"external_kv_tokens_total": 1856,
"external_tokens_per_request": 232
```

Reject totals that are not divisible by requests-per-case.

- [ ] **Step 7: Run tests and verify GREEN**

- [ ] **Step 8: Commit**

```bash
git add benchmarks/cache/build_generalization_dataset.py \
  benchmarks/cache/tests/test_build_generalization_dataset.py
git commit -m "feat: build generalization datasets from cache runs"
```

---

### Task 8: Add pre-registered Issue #15 benchmark configuration templates

**Files:**
- Create: `benchmarks/cache/configs/issue15-7b-load-sentinel-cpu.yaml`
- Create: `benchmarks/cache/configs/issue15-7b-load-sentinel-fs.yaml`
- Create: `benchmarks/cache/configs/issue15-14b-formal-cpu.yaml`
- Create: `benchmarks/cache/configs/issue15-14b-formal-fs.yaml`

**Interfaces:**
- Consumes: existing strict `SuiteConfig`, existing `run_suite.py`, existing `eviction-restore` workload.
- Produces: reproducible pre-registered config templates; formal C-load configs are derived outside Git after concurrency selection.

- [ ] **Step 1: Create the 7B CPU sentinel config from the checked-in crossover config**

Exact intentional changes from `local-crossover.yaml`:

```yaml
cache:
  cpu_bytes_to_use: 8589934592
  filesystem:
    enabled: false
workload:
  prompt_tokens: [1024]
  concurrency: [1, 2, 4, 8]
  shared_prefix_ratios: [0.0]
```

Keep model path, TP=1, GPU KV bytes, pressure fill token budget, requests-per-case, request rate, seed, output tokens, and token-length tolerance unchanged.

- [ ] **Step 2: Create the 7B filesystem sentinel config**

Use CPU bytes `2147483648`, filesystem enabled, prompt `[1024]`, concurrency `[1,2,4,8]`, and otherwise the same control settings.

- [ ] **Step 3: Create the 14B formal configs**

Use:

```yaml
model:
  id: /mnt/model/Qwen2.5-14B-Instruct
  served_name: qwen2.5-14b
workload:
  tokenizer: /mnt/model/Qwen2.5-14B-Instruct
  prompt_tokens: [128, 192, 256, 1024, 4096]
  concurrency: [1]
  shared_prefix_ratios: [0.0]
```

CPU template starts with 8 GiB CPU tier and filesystem disabled. Filesystem template uses 2 GiB CPU tier and filesystem enabled. All other controls match the 7B control where feasible.

- [ ] **Step 4: Validate all four configs with the real loader**

Run a focused Python assertion loop:

```python
from pathlib import Path
from benchmarks.cache.config import load_suite_config

paths = sorted(Path("benchmarks/cache/configs").glob("issue15-*.yaml"))
assert len(paths) == 4
for path in paths:
    config = load_suite_config(path)
    assert config.parallelism.tensor_parallel_size == 1
    assert config.workload.request_rate == ["inf"]
print("validated", len(paths), "issue15 configs")
```

- [ ] **Step 5: Dry-run all four configs before any GPU measurement**

Use `run_suite.py --dry-run`; redirect full output to files and retain only compact case-count summaries. Verify generated cases include the intended `eviction-restore` cases.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/cache/configs/issue15-*.yaml
git commit -m "bench: add issue 15 validation configs"
```

---

### Task 9: Verify implementation locally before hardware execution

**Files:**
- No new files unless fixes are required by the focused verification.

**Interfaces:**
- Consumes: Tasks 2-8 implementation.
- Produces: a clean local implementation proof before any expensive run.

- [ ] **Step 1: Run both new unittest modules**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_build_generalization_dataset.py' -v
```

Expected: all tests PASS without pytest installed.

- [ ] **Step 2: Run the frozen evaluator against the existing control as a compatibility characterization**

First build a small Issue #15-format fixture from checked-in #13 structured values or a dedicated test fixture, then evaluate using `issue14-shadow-cost-calibrated.json`. This is not a hardware rerun. The decision/error behavior must match the #14 calibrated baseline within the same sample set.

- [ ] **Step 3: Run compile and static focused checks**

```bash
python -m compileall -q \
  benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/evaluate_cost_model_generalization.py \
  benchmarks/cache/build_generalization_dataset.py \
  benchmarks/cache/tests/test_cost_model_generalization.py \
  benchmarks/cache/tests/test_build_generalization_dataset.py
ruff check \
  benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/evaluate_cost_model_generalization.py \
  benchmarks/cache/build_generalization_dataset.py \
  benchmarks/cache/tests/test_cost_model_generalization.py \
  benchmarks/cache/tests/test_build_generalization_dataset.py
ruff format --check \
  benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/evaluate_cost_model_generalization.py \
  benchmarks/cache/build_generalization_dataset.py \
  benchmarks/cache/tests/test_cost_model_generalization.py \
  benchmarks/cache/tests/test_build_generalization_dataset.py
git diff --check
```

If `ruff` is unavailable on the Pod, record that fact and defer repository-wide/static authority to GitHub Actions rather than installing unrelated tooling.

- [ ] **Step 4: Verify the runtime scope invariant**

```bash
git diff --name-only origin/main...HEAD | sort
```

Expected: no file under `vllm/v1/kv_offload/`, scheduler, attention, block manager, or other active inference path.

---

### Task 10: Run Phase 0 provenance, drift, and 14B feasibility preflight

**Files:**
- Raw outputs only under `/code/results/cache`; no checked-in result yet.

**Interfaces:**
- Consumes: verified implementation/configs.
- Produces: same-session provenance, 7B/C1 sentinel, and 14B load/workload feasibility evidence.

- [ ] **Step 1: Capture locale-independent environment evidence**

Use existing `collect_environment_evidence()` through a dry or minimal run and independently preserve compact provenance with `lscpu --json`, `numactl --hardware`, `nvidia-smi` UUID/topology, `findmnt -T /tmp/vllm-kv-cache -o TARGET,SOURCE,FSTYPE,OPTIONS`, memory snapshot, Python/vLLM version, and worktree HEAD.

Do not infer physical NVMe from an overlay filesystem.

- [ ] **Step 2: Verify GPU0 UUID matches the control GPU**

Expected GPU0 UUID: `GPU-5516e45d-3e50-69ef-f0f2-8ecff465beea`. If device ordering or UUID differs, record it before continuing and do not silently call the condition same-hardware control.

- [ ] **Step 3: Run only the 7B/C1 1024 sentinel cases needed for recompute, CPU-primary, and filesystem paths**

Use `--case-id` to avoid all unrelated generated cases. Determine selected case IDs from a dry-run `scenarios.json` by filtering:

```text
workload_kind == eviction-restore
prompt_tokens == 1024
concurrency == 1
cache_mode in {no-cache, cpu-offload, tiered-fs}
```

CPU-primary comes from the 8 GiB CPU sentinel config; tiered-fs comes from the 2 GiB filesystem sentinel config. Preserve workload SHA equality.

- [ ] **Step 4: Compare sentinel against archived control only as a drift diagnostic**

Record P95 recompute/CPU/filesystem latencies and provenance. Do not add sentinel data to the generalization accuracy aggregate.

- [ ] **Step 5: Preflight 14B load and deterministic workload generation**

Run dry/workload generation for all five formal anchors before expensive benchmark execution. Do not reseed failed anchors.

- [ ] **Step 6: Preflight 14B CPU-primary capacity provenance at 1024**

Run one 14B/C1 1024 CPU-primary case with the 8 GiB CPU template. Accept it only if external KV tokens and CPU-to-GPU transfer evidence are positive.

If the intended CPU restore does not occur because the victim is evicted from CPU, preserve the failed provenance and add a material-deviation comment to Issue #15 before changing CPU capacity. The comment must state the observed failure, the exact old/new CPU capacity, and that capacity is being changed only to preserve the `cpu_primary` source tier rather than as a generalization axis.

- [ ] **Step 7: Preflight 14B filesystem provenance at 1024**

Require positive external KV tokens, positive CPU-to-GPU transfer count/bytes, and positive async tiering lookup evidence. If it does not reach the filesystem tier, preserve the evidence and record a material deviation before changing pressure/capacity.

---

### Task 11: Select the minimum materially contended C-load condition

**Files:**
- Raw sentinel outputs under `/code/results/cache`.
- A small selection JSON outside Git until final report generation.

**Interfaces:**
- Consumes: same-session C1 sentinel plus C2/C4/C8 candidate sentinel runs at requested 1024.
- Produces: exactly one selected C-load concurrency or a recorded `no_material_contention_through_c8` outcome.

- [ ] **Step 1: Run C2 sentinel only**

Measure recompute, CPU-primary, and filesystem paths with the same 1024 workload identity and tier-specific capacities as C1.

- [ ] **Step 2: Calculate P95 relative changes**

For each principal path:

```text
relative_change = abs(candidate_p95 - c1_p95) / c1_p95
```

Accept C2 only if one path is >= 0.20 and another path is >= 0.10 and restore provenance remains valid.

- [ ] **Step 3: If C2 does not qualify, run C4 and apply the same rule**

- [ ] **Step 4: If C4 does not qualify, run C8 and apply the same rule**

- [ ] **Step 5: Stop immediately at the first qualifying concurrency**

Do not run larger candidates after selection.

- [ ] **Step 6: If C8 still does not qualify, stop expansion**

Record `no_material_contention_through_c8`. Do not silently try C16/C32. Add an Issue #15 decision-log comment because this outcome changes the formal C-load execution path and may lead to `insufficient_evidence` for the load axis.

- [ ] **Step 7: Create local formal C-load configs only after selection**

Programmatically copy the 7B CPU/filesystem sentinel YAML into `/code/results/cache/issue15-configs/` and replace:

```yaml
workload:
  prompt_tokens: [128, 192, 256, 1024, 4096]
  concurrency: [<selected concurrency as an integer written by the script>]
```

Do not commit the derived local config. Record its SHA256 and selected concurrency in the final structured artifact.

---

### Task 12: Run the two formal new conditions and build immutable measurement artifacts

**Files:**
- Raw runs: `/code/results/cache/...`
- Intermediate condition JSONs: `/code/results/cache/issue15-structured/c-load.json`, `/code/results/cache/issue15-structured/c-model.json`

**Interfaces:**
- Consumes: selected C-load configs and 14B formal configs.
- Produces: two structured condition datasets before any diagnostic fitting.

- [ ] **Step 1: Run formal C-load recompute/CPU/filesystem cases only**

Use `--case-id` filtering so only `eviction-restore` cases for the five formal anchors and selected concurrency execute. Do not run cold/warm/shared/mixed/restart cases.

- [ ] **Step 2: Build `c-load.json` immediately after raw runs**

Use:

```bash
python benchmarks/cache/build_generalization_dataset.py \
  --condition-id c-load \
  --recompute-run "$RECOMPUTE_RUN" \
  --cpu-run "$CPU_RUN" \
  --filesystem-run "$FILESYSTEM_RUN" \
  --percentile p95 \
  --output /code/results/cache/issue15-structured/c-load.json
```

In execution, set the three shell variables from the exact run directories just created; print them before invoking the builder and record them in the Issue #15 work log.

- [ ] **Step 3: Validate `c-load.json` before any evaluator run**

Use Python assertions to require condition ID, five requested anchors where valid, positive per-request external tokens for accepted restore samples, and explicit exclusions for invalid points.

- [ ] **Step 4: Run formal C-model recompute/CPU/filesystem cases only**

Use Qwen2.5-14B-Instruct, GPU0, TP1, C1, and the five formal anchors. Apply only provenance-preserving capacity adjustments that were pre-recorded in Issue #15 during Task 10.

- [ ] **Step 5: Build and validate `c-model.json` before evaluation**

Use the same builder and validation rules.

- [ ] **Step 6: Do not rerun a failed point merely to improve the aggregate**

Retry only operationally retryable failures with preserved first-failure evidence. Deterministic workload-generation failure remains an exclusion, not a reason to reseed.

---

### Task 13: Evaluate frozen generalization, then diagnose failures without erasing them

**Files:**
- Intermediate evaluations under `/code/results/cache/issue15-structured/`.

**Interfaces:**
- Consumes: frozen profile artifact plus `c-load.json` / `c-model.json`.
- Produces: primary holdout verdicts, optional diagnostics, and boundary-refinement triggers.

- [ ] **Step 1: Evaluate C-load with the frozen profile and save primary evidence**

```bash
python benchmarks/cache/evaluate_cost_model_generalization.py \
  --input /code/results/cache/issue15-structured/c-load.json \
  --profile benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json \
  --percentile p95 \
  --output /code/results/cache/issue15-structured/c-load-evaluation.json
```

Do not pass `--diagnose` on the first invocation.

- [ ] **Step 2: Evaluate C-model the same way**

Save `c-model-evaluation.json` before diagnostics.

- [ ] **Step 3: Preserve checksums of both primary evaluation files**

Compute SHA256 before any diagnostic command.

- [ ] **Step 4: Run diagnostics only for conditions/curves that fail raw transfer**

Use `--diagnose` to a different output filename such as `c-model-diagnostic.json`. Never overwrite primary evaluation files.

- [ ] **Step 5: Trigger local boundary refinement only when the approved rule fires**

Refine only if:

- P95 absolute actual margin <= 1 ms;
- adjacent formal anchors change actual preferred-path sign; or
- frozen profile makes a clearly non-boundary wrong decision.

Choose local anchors between the implicated neighboring points, preserve deterministic generation, and stop when the boundary/failure region is bracketed.

- [ ] **Step 6: Add an Issue #15 comment at the first meaningful failure boundary or stop decision**

Record:

- condition and source tier;
- anchor/external-token region;
- raw prediction error and decision error;
- whether one scalar repairs the curve;
- whether a new input is implicated;
- whether the matrix stops or a narrowly scoped extra measurement is justified.

---

### Task 14: Produce final structured results, validation report, and Issue #16 eligibility handoff

**Files:**
- Create: `docs/engineering/validation/2026-08-11-issue15-generalization-validation.json`
- Create: `docs/engineering/validation/2026-08-11-issue15-generalization-validation.md`
- Create: `docs/engineering/handoffs/2026-08-11-issue16-active-decision-handoff.md`
- Modify: `docs/engineering/CURRENT_STATE.md`
- Modify: `docs/engineering/README.md` if its validation/handoff index requires the new records.

**Interfaces:**
- Consumes: immutable primary evaluations, diagnostics, raw run provenance, and Issue #15 decision-log entries.
- Produces: closeable Issue #15 evidence and bounded Issue #16 input.

- [ ] **Step 1: Build the checked-in JSON from immutable intermediate files**

The final JSON must include:

- schema version and issue number;
- design spec and plan paths;
- live-base provenance used for the experiment worktree;
- C0 archived control summary;
- C-load selection trace for C2/C4/C8 until stop;
- C-load and C-model condition metadata;
- all accepted and excluded samples;
- primary frozen-profile evaluations and their SHA256 values;
- low-confidence evidence separately;
- diagnostics separately;
- parameter classification for recompute, CPU-primary restore, and secondary filesystem restore;
- failure boundaries;
- candidate required features/inputs;
- Issue #16 eligibility map.

- [ ] **Step 2: Write the report to answer every Issue #15 research question explicitly**

The Markdown report must answer:

1. whether the external-token recompute curve transfers;
2. whether CPU-primary restore transfers;
3. whether machine/memory/PCIe/NUMA scaling is implicated by available evidence;
4. whether secondary tier is path-specific;
5. whether concurrency changes global scale or curve shape/boundary;
6. which parameters can remain fixed;
7. which parameters need online calibration;
8. which low-cardinality runtime observations are sufficient candidates for calibration;
9. where confidence/systematic-error boundaries lie;
10. whether a new input/feature should be designed before broader sweeps.

Negative results count as valid completion evidence if they are preserved and interpreted.

- [ ] **Step 3: Write the Issue #16 handoff as an eligibility map, not a global success claim**

Each row/entry must state at least:

```text
model/load region
source tier
external-token region observed
fixed-profile status
confidence status
active-design eligibility: eligible | ineligible | needs-online-calibration
reason
```

Do not authorize active behavior in low-confidence extrapolation or failed regions.

- [ ] **Step 4: Update current engineering state only to the evidence actually established**

Do not mark Issue #15 completed merely because code is ready. The branch may state that the PR closes #15 only after the required evidence exists.

- [ ] **Step 5: Run final local verification**

Use:

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_build_generalization_dataset.py' -v
python -m compileall -q benchmarks/cache
python - <<'PY'
import json
from pathlib import Path
for path in [
    Path('docs/engineering/validation/2026-08-11-issue15-generalization-validation.json'),
    Path('benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json'),
]:
    json.loads(path.read_text(encoding='utf-8'))
    print('json-ok', path)
PY
git diff --check
```

Run targeted Ruff checks if available. Do not install pytest just for local proof.

- [ ] **Step 6: Verify no forbidden active-runtime path changed**

Compare changed filenames against the verified base. Any unexpected file under `vllm/v1/kv_offload/`, scheduler, attention, cache manager, transfer execution, or CUDA paths is a stop condition requiring explicit review.

- [ ] **Step 7: Commit final evidence locally**

Use a commit message such as:

```bash
git add benchmarks/cache docs/engineering docs/superpowers
git commit -m "validate: record issue 15 cost model generalization"
```

Do not push from the Pod.

---

### Task 15: Deliver through GitHub, run authoritative CI, and keep merge gated by explicit authorization

**Files:**
- GitHub branch/PR metadata plus the exact verified file contents from the Pod worktree.

**Interfaces:**
- Consumes: verified local commits/content and final artifact checksums.
- Produces: Draft PR against current `main`, CI evidence, and Issue #15 final decision-log comment.

- [ ] **Step 1: Transfer verified content through the GitHub-authoritative path**

Use the established content-addressed handoff pattern from Issue #14 if direct filesystem transfer is needed. Verify each delivered file's Git blob/content hash against the Pod version before opening the PR.

- [ ] **Step 2: Open the feature PR as Draft**

The PR body should include `Closes #15` only when all close criteria are present. Draft state does not authorize merge.

Summarize:

- frozen-profile-first semantics;
- selected C-load concurrency or no-contention outcome;
- 14B model-scale evidence;
- fixed-profile accuracy/error comparison;
- transfer/scale/shape classification;
- failure boundaries and Issue #16 eligibility;
- exact scope boundary: no active enforcement.

- [ ] **Step 3: Run GitHub Actions and treat repository-wide pre-commit as authoritative**

Do not claim CI ran pytest unless the actual workflow does so. Report exact checks/runs that exist.

- [ ] **Step 4: Fix CI only within Issue #15 scope and re-verify semantics**

Any CI repair that changes experimental interpretation is a material change and must first be logged on Issue #15. Formatting/SPDX/type hygiene that does not alter interpretation does not require issue noise.

- [ ] **Step 5: Add the final Issue #15 decision-log comment**

Link the Draft/ready PR, structured result path, validation report, Issue #16 handoff, final classifications, and any ineligible regions.

- [ ] **Step 6: Do not merge without fresh explicit user authorization**

Even with green CI and an approved result, stop before merge and request explicit authorization for this PR merge. Previous authorization to design, implement, comment, create a PR, or fix CI is not merge authorization.

---

## Plan Self-Review Checklist

Before execution begins, verify these mappings:

- frozen Issue #14 profile identity -> Task 2;
- runtime external tokens vs requested anchors -> Tasks 3 and 7;
- fixed-profile holdout without derive/recalibrate -> Tasks 4 and 6;
- high-confidence-only gate and insufficient evidence -> Task 4;
- one-scalar diagnostic without overwriting primary evidence -> Task 5 and Task 13;
- workload fairness and actual restore provenance -> Task 7;
- minimum C2/C4/C8 contention selection -> Task 11;
- separate CPU-primary vs filesystem capacities/tier paths -> Tasks 8, 10, 12;
- model-scale axis Qwen2.5-14B/C1 -> Tasks 8, 10, 12;
- five formal anchors -> Tasks 8 and 12;
- failure-boundary-only refinement -> Task 13;
- structured results, validation report, transferable/environment-specific classification -> Task 14;
- bounded Issue #16 eligibility -> Task 14;
- Issue comment decision journal -> Tasks 10, 11, 13, 15;
- no active enforcement and explicit merge authorization -> Global Constraints and Task 15.
