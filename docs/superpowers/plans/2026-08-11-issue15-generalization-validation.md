# Issue #15 Generalization Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce fixed-profile holdout evidence for the Issue #14 KV restore/recompute cost model under one materially contended 7B load condition and one Qwen2.5-14B model-scale condition, then classify each principal curve as transferable, environment-scale-sensitive, or shape/missing-feature limited without changing active runtime behavior.

**Architecture:** Keep the Issue #14 calibration CLI unchanged. Add a separate pure generalization module and CLI that evaluate a supplied frozen Issue #14 profile against a structured condition dataset. Add a dataset builder that converts existing `run_suite.py` artifacts into the Issue #15 schema with explicit workload fairness and restore provenance. Hardware execution remains shadow-only and uses the existing cache benchmark stack.

**Tech Stack:** Python 3.11, stdlib `dataclasses`/`json`/`statistics`/`unittest`, existing `benchmarks.cache.cost_model_calibration` primitives, existing `vllm.v1.kv_offload.cost_model.OffloadCostModel`, existing cache benchmark YAML/Pydantic pipeline, GitHub repository-wide pre-commit as authoritative CI.

## Global Constraints

- GitHub is the authoritative remote and write path; the Pod is for build, focused tests, benchmark, and hardware validation.
- Never run `git push` from the Pod.
- Never use `git clean -fd`, `reset --hard`, or a checkout operation that can remove the three intentional untracked local YAML files under `/code/vllm/benchmarks/cache/configs/`.
- Do not install pytest for this work. New focused tests must run with stdlib `unittest`; GitHub CI may also collect them through pytest if the workflow does so.
- Issue #15 remains shadow-only. Do not change scheduler enforcement, matched-token behavior, transfer scheduling, cache contents, or active restore/recompute selection.
- Primary percentile is P95.
- Primary evidence is the frozen Issue #14 calibrated profile evaluated on new-condition data saved before any diagnostic scaling.
- Requested prompt tokens are workload anchors only. Evaluation uses runtime external KV tokens per request.
- Formal requested-token anchors are exactly `128, 192, 256, 1024, 4096` unless a material deviation is first recorded on Issue #15.
- C-load probes concurrency exactly `2 -> 4 -> 8` against a same-session C1 sentinel at 1024 requested tokens and stops at the first materially contended load.
- Material contention means one principal P95 path changes by at least 20%, another principal path changes by at least 10%, and required restore provenance remains valid.
- Fixed-profile transfer requires valid high-confidence evidence for all three principal curves, decision accuracy >= 95%, principal macro-MAPE <= 15%, no principal curve MAPE > 20%, and no wrong high-confidence decision with absolute actual margin > 1 ms.
- If any principal curve lacks valid high-confidence formal evidence, classify the condition `insufficient_evidence`.
- Low-confidence samples remain evidence but never expand Issue #16 eligibility.
- Diagnostic scale is one multiplicative scalar per curve: median of `actual_latency / frozen_predicted_latency`.
- Container-local filesystem evidence remains `filesystem` / `tiered-fs` unless physical device provenance proves a stronger claim.
- Material experiment-interpretation changes must be commented on Issue #15 before measurements use the changed condition.
- A green PR never implies merge authorization; request fresh explicit authorization before merge.

---

## File Structure

- Create `benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json`: immutable profile copied programmatically from the checked-in Issue #14 `calibrated_profile` field.
- Create `benchmarks/cache/cost_model_generalization.py`: condition loader, frozen evaluation, high-confidence gate, low-confidence partition, one-scalar diagnostics.
- Create `benchmarks/cache/evaluate_cost_model_generalization.py`: deterministic frozen-profile CLI; never derives a new profile.
- Create `benchmarks/cache/build_generalization_dataset.py`: converts run-suite artifacts into the condition JSON and validates workload/restore provenance.
- Create `benchmarks/cache/tests/test_cost_model_generalization.py`: stdlib tests for profile fidelity, loading, frozen evaluation, gate semantics, diagnostics, and CLI determinism.
- Create `benchmarks/cache/tests/test_build_generalization_dataset.py`: stdlib tests for run pairing and provenance validation.
- Create four config templates:
  - `benchmarks/cache/configs/issue15-7b-load-sentinel-cpu.yaml`
  - `benchmarks/cache/configs/issue15-7b-load-sentinel-fs.yaml`
  - `benchmarks/cache/configs/issue15-14b-formal-cpu.yaml`
  - `benchmarks/cache/configs/issue15-14b-formal-fs.yaml`
- Create final evidence:
  - `docs/engineering/validation/2026-08-11-issue15-generalization-validation.json`
  - `docs/engineering/validation/2026-08-11-issue15-generalization-validation.md`
  - `docs/engineering/handoffs/2026-08-11-issue16-active-decision-handoff.md`
- Update `docs/engineering/CURRENT_STATE.md` and, if needed for indexing, `docs/engineering/README.md` only after complete evidence exists.

Expected implementation scope has no change under `vllm/v1/kv_offload/` or any active scheduler/inference path.

---

### Task 1: Create a safe isolated Pod worktree

**Files:** No repository changes.

**Interfaces:**
- Consumes: live GitHub `main` SHA re-verified at execution time.
- Produces: `/code/vllm-worktrees/issue15-generalization-validation` based on a Pod mirror SHA exactly matching live GitHub `main`.

- [ ] **Step 1: Invoke `superpowers:using-git-worktrees` before Pod repository operations.**

- [ ] **Step 2: Re-read live GitHub `main` and record its SHA.**

Do not reuse the design-time SHA as an assumption.

- [ ] **Step 3: Verify the three preserved local YAML files and current checkout without modifying it.**

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

Expected: all three files are present and untracked.

- [ ] **Step 4: Fetch only the Pod mirror ref; do not checkout or reset the old worktree.**

```bash
cd /code/vllm || exit 1
git fetch origin main
printf 'origin/main='; git rev-parse origin/main
```

Compare the printed SHA to live GitHub. If they differ, stop execution and record a mirror-sync mismatch.

- [ ] **Step 5: Create the fresh worktree from verified `origin/main`.**

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

Expected: clean isolated worktree at the verified SHA.

---

### Task 2: Freeze the Issue #14 calibrated profile as a first-class artifact

**Files:**
- Create: `benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json`
- Create: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:**
- Consumes: `docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json["calibrated_profile"]`.
- Produces: a top-level `cache_cost_model` mapping accepted by existing `load_profile_artifact()` and exactly equal to the source calibration result.

- [ ] **Step 1: Write the RED fidelity test.**

```python
import json
import unittest
from pathlib import Path

from benchmarks.cache.cost_model_calibration import load_profile_artifact

_REPO_ROOT = Path(__file__).resolve().parents[3]


class FrozenProfileArtifactTests(unittest.TestCase):
    def test_issue14_frozen_profile_matches_calibration_result(self) -> None:
        source_path = _REPO_ROOT / (
            "docs/engineering/validation/"
            "2026-08-10-issue14-shadow-cost-model-calibration.json"
        )
        profile_path = _REPO_ROOT / (
            "benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json"
        )
        source = json.loads(source_path.read_text(encoding="utf-8"))
        artifact = json.loads(profile_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["cache_cost_model"], source["calibrated_profile"])
        self.assertEqual(load_profile_artifact(profile_path), source["calibrated_profile"])
        self.assertEqual(artifact["provenance"]["issue"], 14)
        self.assertEqual(artifact["provenance"]["profile_role"], "frozen_holdout_seed")
```

- [ ] **Step 2: Run RED without pytest.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
```

Expected: error because the frozen profile file does not exist.

- [ ] **Step 3: Generate the artifact programmatically.**

Run this exact repository-local script:

```bash
python - <<'PY'
import json
from pathlib import Path

source_path = Path(
    'docs/engineering/validation/2026-08-10-issue14-shadow-cost-model-calibration.json'
)
out_path = Path('benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json')
source = json.loads(source_path.read_text(encoding='utf-8'))
payload = {
    'schema_version': 1,
    'name': 'issue14-shadow-cost-calibrated',
    'provenance': {
        'issue': 14,
        'profile_role': 'frozen_holdout_seed',
        'source_artifact': str(source_path),
        'source_field': 'calibrated_profile',
        'percentile': 'p95',
        'note': (
            'Frozen Issue #14 shadow profile for Issue #15 holdout evaluation; '
            'not a production default.'
        ),
    },
    'cache_cost_model': source['calibrated_profile'],
}
out_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
print(out_path)
PY
```

- [ ] **Step 4: Run GREEN and commit locally.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
git add benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "test: freeze issue 14 cost profile for holdout"
```

---

### Task 3: Add a neutral Issue #15 condition loader

**Files:**
- Create: `benchmarks/cache/cost_model_generalization.py`
- Modify: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:**
- Produces `GeneralizationCondition` and `load_generalization_condition(path: Path, percentile: str = "p95")`.
- Reuses existing `DecisionSample` and `CalibrationDataset` rather than changing #14 calibration semantics.

- [ ] **Step 1: Add a RED loader test with a complete two-source fixture.**

The fixture must contain condition metadata plus two rows at requested 256, one `cpu_primary` and one `secondary:filesystem`, with:

```python
fixture = {
    'schema_version': 1,
    'issue': 15,
    'condition': {
        'id': 'c-model',
        'model': '/mnt/model/Qwen2.5-14B-Instruct',
        'served_model': 'qwen2.5-14b',
        'concurrency': 1,
        'request_rate': 'inf',
        'requests_per_case': 8,
        'tensor_parallel_size': 1,
        'gpu_uuid': 'GPU-test',
        'environment_artifact': '/code/results/cache/run/environment.json',
        'run_directories': {
            'recompute': '/code/results/cache/recompute',
            'cpu_primary': '/code/results/cache/cpu',
            'secondary_filesystem': '/code/results/cache/fs',
        },
    },
    'samples': [
        {
            'source': 'cpu_primary',
            'requested_tokens': 256,
            'external_kv_tokens_total': 1856,
            'external_tokens_per_request': 232,
            'recompute_ttft_ms': {'p95': 30.0},
            'restore_ttft_ms': {'p95': 24.0},
            'transfer_evidence': {
                'cpu_to_gpu_transfers': 8,
                'cpu_to_gpu_bytes': 1000,
                'tiered_fs_async_lookups': 0,
            },
            'workload': {'measure_sha256': 'a', 'populate_sha256': 'b'},
        },
        {
            'source': 'secondary:filesystem',
            'requested_tokens': 256,
            'external_kv_tokens_total': 1856,
            'external_tokens_per_request': 232,
            'recompute_ttft_ms': {'p95': 30.0},
            'restore_ttft_ms': {'p95': 40.0},
            'transfer_evidence': {
                'cpu_to_gpu_transfers': 8,
                'cpu_to_gpu_bytes': 1000,
                'tiered_fs_async_lookups': 8,
            },
            'workload': {'measure_sha256': 'a', 'populate_sha256': 'b'},
        },
    ],
    'excluded_samples': [],
}
```

Assert model/concurrency metadata and that the internal decision samples use `external_tokens == 232`. Add a second test that changes total external tokens to 1857 and expects `ValueError`.

- [ ] **Step 2: Run RED.**

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the dataclass and loader using the exact validation flow below.**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from benchmarks.cache.cost_model_calibration import CalibrationDataset, DecisionSample


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
    percentile: str = 'p95',
) -> GeneralizationCondition:
    if percentile not in {'p50', 'p95', 'p99'}:
        raise ValueError(f'unsupported percentile: {percentile}')
    raw = json.loads(path.read_text(encoding='utf-8'))
    if raw.get('schema_version') != 1 or raw.get('issue') != 15:
        raise ValueError('generalization artifact must be schema 1 for issue 15')
    meta = raw['condition']
    requests = meta['requests_per_case']
    if type(requests) is not int or requests <= 0:
        raise ValueError('requests_per_case must be a positive integer')
    decision_samples = []
    sample_metadata = []
    for row in raw.get('samples', []):
        source = row['source']
        if source not in {'cpu_primary', 'secondary:filesystem'}:
            raise ValueError(f'unsupported source: {source}')
        total = row['external_kv_tokens_total']
        per_request = row['external_tokens_per_request']
        if type(total) is not int or total <= 0:
            raise ValueError('external_kv_tokens_total must be a positive integer')
        if type(per_request) is not int or per_request <= 0:
            raise ValueError('external_tokens_per_request must be a positive integer')
        if total != requests * per_request:
            raise ValueError('external token total/per-request mismatch')
        decision_samples.append(
            DecisionSample(
                source=source,
                requested_tokens=int(row['requested_tokens']),
                external_tokens=per_request,
                actual_recompute_ms=float(row['recompute_ttft_ms'][percentile]),
                actual_restore_ms=float(row['restore_ttft_ms'][percentile]),
            )
        )
        sample_metadata.append(dict(row))
    dataset = CalibrationDataset(
        percentile=percentile,
        source_artifact=str(path),
        requests_per_case=requests,
        decision_samples=tuple(
            sorted(decision_samples, key=lambda s: (s.requested_tokens, s.source))
        ),
        repeat_direction_checks=(),
        excluded_samples=tuple(raw.get('excluded_samples', [])),
    )
    return GeneralizationCondition(
        condition_id=str(meta['id']),
        model=str(meta['model']),
        served_model=str(meta['served_model']),
        concurrency=int(meta['concurrency']),
        request_rate=meta['request_rate'],
        tensor_parallel_size=int(meta['tensor_parallel_size']),
        gpu_uuid=str(meta['gpu_uuid']),
        environment_artifact=str(meta['environment_artifact']),
        run_directories={str(k): str(v) for k, v in meta['run_directories'].items()},
        dataset=dataset,
        sample_metadata=tuple(sample_metadata),
        excluded_samples=tuple(raw.get('excluded_samples', [])),
    )
```

- [ ] **Step 4: Run GREEN and commit.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
git add benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "feat: load cost model generalization conditions"
```

---

### Task 4: Implement frozen-profile evaluation and the transfer gate

**Files:**
- Modify: `benchmarks/cache/cost_model_generalization.py`
- Modify: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:**
- Produces `evaluate_frozen_condition(condition, profile, *, profile_identity)`.
- Calls existing `evaluate_profile()` only; never calls `derive_calibrated_profile()`.

- [ ] **Step 1: Add RED tests for a deliberately wrong supplied profile.**

Assert:

```python
result = evaluate_frozen_condition(
    condition,
    frozen_profile,
    profile_identity='fixture-profile',
)
self.assertEqual(result['evaluation_mode'], 'frozen_profile_holdout')
self.assertEqual(result['profile_identity'], 'fixture-profile')
self.assertNotIn('calibrated_profile', result)
self.assertFalse(result['samples'][0]['decision_correct'])
```

- [ ] **Step 2: Add RED gate tests for six cases.**

Cover:

1. all three curves represented by high-confidence rows and all gates pass -> `fixed_profile_transfer_pass`;
2. any curve has zero high-confidence evidence -> `insufficient_evidence`;
3. macro <= 15% but a principal curve > 20% -> `fixed_profile_transfer_fail`;
4. wrong high-confidence decision with absolute actual margin > 1 ms -> fail;
5. boundary-sensitive wrong row remains counted in accuracy rather than dropped;
6. low-confidence rows are reported but do not satisfy the evidence-presence gate.

- [ ] **Step 3: Run RED.**

- [ ] **Step 4: Implement fixed constants and high-confidence aggregation.**

```python
DECISION_ACCURACY_MIN = 0.95
PRINCIPAL_MACRO_MAPE_MAX = 15.0
PRINCIPAL_CURVE_MAPE_MAX = 20.0
BOUNDARY_MARGIN_MS = 1.0
```

Call:

```python
raw = evaluate_profile(condition.dataset, profile)
```

Partition `raw['samples']` by confidence. For high-confidence rows, deduplicate recompute errors by `(requested_tokens, external_tokens)`, compute separate recompute/CPU/filesystem MAPE values, then equal-weight macro-MAPE. Determine evidence presence from non-zero sample counts for each principal curve.

Use this exact verdict order:

```text
if any principal high-confidence sample count is zero:
    insufficient_evidence
elif decision_accuracy < 0.95:
    fixed_profile_transfer_fail
elif principal_macro_mape_percent > 15.0:
    fixed_profile_transfer_fail
elif any principal curve MAPE > 20.0:
    fixed_profile_transfer_fail
elif any wrong high-confidence row has abs(actual_margin_ms) > 1.0:
    fixed_profile_transfer_fail
else:
    fixed_profile_transfer_pass
```

Return immutable raw sample predictions plus threshold values, high-confidence aggregate, low-confidence sample list, and verdict.

- [ ] **Step 5: Run GREEN and commit.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
git add benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "feat: evaluate frozen cost profiles on holdout data"
```

---

### Task 5: Add one-scalar scale/shape diagnostics

**Files:**
- Modify: `benchmarks/cache/cost_model_generalization.py`
- Modify: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:** Produces `diagnose_curve_scaling(evaluation)` without modifying the primary verdict or predictions.

- [ ] **Step 1: Add RED synthetic tests for all three diagnostic classes.**

Use one curve with raw MAPE <= 15%, one with constant 1.5x actual/predicted ratio, and one with ratios `1.0, 1.5, 2.0` that remain >15% residual after one scalar.

- [ ] **Step 2: Implement exact diagnostic math.**

For each principal curve:

```python
scale = statistics.median(actual / predicted for actual, predicted in points)
residual_mape = 100.0 * statistics.fmean(
    abs(predicted * scale - actual) / actual
    for actual, predicted in points
)
```

Classify:

```text
raw MAPE <= 15% -> transferable
raw MAPE > 15% and residual MAPE <= 15% -> environment_specific_scale_candidate
residual MAPE > 15% -> curve_shape_or_missing_feature
```

Store diagnostics under a separate `diagnostics` mapping only.

- [ ] **Step 3: Run GREEN and commit.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
git add benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "feat: classify cost model scale and shape drift"
```

---

### Task 6: Add the deterministic frozen-profile CLI

**Files:**
- Create: `benchmarks/cache/evaluate_cost_model_generalization.py`
- Modify: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:**
- Required args: `--input`, `--profile`, `--output`.
- Optional args: `--percentile p50|p95|p99`, default P95; `--diagnose`; `--check`.
- No CLI option may alter frozen acceptance thresholds.

- [ ] **Step 1: Add a RED deterministic CLI test.**

Call `main(args)` twice with the same fixture and assert identical output bytes. Assert summary starts with `holdout: condition=c-model` and `--check` exits 1 for a non-pass verdict.

- [ ] **Step 2: Implement the CLI flow exactly.**

```python
condition = load_generalization_condition(args.input, percentile=args.percentile)
profile = load_profile_artifact(args.profile)
result = evaluate_frozen_condition(
    condition,
    profile,
    profile_identity=str(args.profile),
)
if args.diagnose:
    result['diagnostics'] = diagnose_curve_scaling(result)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(result, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
```

Print one compact line containing condition, decision correct/total, accuracy, macro-MAPE, and classification. `--check` returns 0 only for `fixed_profile_transfer_pass`.

- [ ] **Step 3: Run GREEN and characterize the old #14 CLI unchanged.**

Do not modify `benchmarks/cache/evaluate_cost_model.py`. Run the existing checked-in #13 evaluator offline and confirm it retains its current `after:` behavior.

- [ ] **Step 4: Commit.**

```bash
git add benchmarks/cache/evaluate_cost_model_generalization.py \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "feat: add frozen profile generalization evaluator"
```

---

### Task 7: Build structured condition datasets from run-suite artifacts

**Files:**
- Create: `benchmarks/cache/build_generalization_dataset.py`
- Create: `benchmarks/cache/tests/test_build_generalization_dataset.py`

**Interfaces:**
- CLI args: `--condition-id`, `--recompute-run`, `--cpu-run`, `--filesystem-run`, `--percentile`, `--output`.
- Model, concurrency, request rate, requests-per-case, TP size, environment path, and run paths are inferred from manifests/records, not trusted from labels.

- [ ] **Step 1: Add synthetic run fixtures using only stdlib tempfile/json.**

Each fixture run contains `manifest.json`, `environment.json`, `scenario-results.jsonl`, and raw `metadata.json`. Use completed `eviction-restore` records at one anchor for no-cache, CPU-offload, and tiered-fs. Give paired records identical measure/populate SHA256 values.

- [ ] **Step 2: Add RED acceptance/rejection tests.**

Accept CPU-primary only with positive external KV tokens and positive CPU-to-GPU transfer count/bytes. Accept filesystem only with those plus positive async tiering lookup evidence. Hard-fail workload SHA mismatch or condition metadata mismatch. Record configured-mode/no-transfer, incomplete benchmark, and missing tier evidence as explicit exclusions.

- [ ] **Step 3: Implement record selection.**

Load each `scenario-results.jsonl` and select records satisfying:

```python
record['workload_kind'] == 'eviction-restore'
```

Then pair by `(prompt_tokens, concurrency, request_rate)` and expected cache mode. Read each record's `workload_metadata` JSON for measure/populate SHA values. Require identical workload SHA values across recompute and restore rows before comparing latency.

- [ ] **Step 4: Implement Prometheus evidence lookup.**

Read `record['normalized']['prometheus']['delta']`. Match metric base names ending with the runtime families used by #13 evidence. Sum positive values for external-KV token samples labeled `source="external_kv_transfer"`, CPU-to-GPU transfer count, and CPU-to-GPU total bytes. For filesystem rows also require the normalized tiering async lookup count/sum evidence to be positive.

- [ ] **Step 5: Emit exact external-token normalization.**

For an accepted restore row:

```python
if external_total % requests_per_case != 0:
    raise ValueError('external KV tokens not divisible by requests_per_case')
external_per_request = external_total // requests_per_case
```

Write both total and per-request values. Pair the same no-cache P95 recompute latency with each source row.

- [ ] **Step 6: Emit condition metadata and exclusions.**

Output schema uses:

```text
schema_version = 1
issue = 15
condition.id = supplied condition id
condition model/concurrency/request-rate/TP/GPU/environment/run paths = inferred evidence
samples = accepted CPU/filesystem rows
excluded_samples = explicit invalid records with reason
```

GPU UUID must be extracted from environment evidence. If unavailable, builder fails rather than inventing provenance.

- [ ] **Step 7: Run GREEN and commit.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_build_generalization_dataset.py' -v
git add benchmarks/cache/build_generalization_dataset.py \
  benchmarks/cache/tests/test_build_generalization_dataset.py
git commit -m "feat: build generalization datasets from cache runs"
```

---

### Task 8: Add the four pre-registered benchmark config templates

**Files:** the four config paths listed in File Structure.

**Interfaces:** Existing strict `load_suite_config()` and `run_suite.py` only; no benchmark-core change.

- [ ] **Step 1: Create 7B CPU sentinel config from `local-crossover.yaml`.**

Keep control model, TP1, GPU KV 2 GiB, pressure fill 65536, requests-per-case 8, request rate inf, seed 1, output 1, tolerance 2. Set CPU tier to 8 GiB, filesystem disabled, prompt list `[1024]`, concurrency `[1, 2, 4, 8]`, shared-prefix ratios `[0.0]`.

- [ ] **Step 2: Create 7B filesystem sentinel config.**

Same controls, CPU tier 2 GiB, filesystem enabled, prompt `[1024]`, concurrency `[1, 2, 4, 8]`.

- [ ] **Step 3: Create 14B CPU and filesystem formal configs.**

Use model/tokenizer `/mnt/model/Qwen2.5-14B-Instruct`, served name `qwen2.5-14b`, TP1, C1, formal prompts `[128, 192, 256, 1024, 4096]`, shared-prefix ratios `[0.0]`. CPU config starts at 8 GiB with filesystem disabled; filesystem config uses 2 GiB with filesystem enabled. Preserve other control settings.

- [ ] **Step 4: Validate all four configs through the real loader.**

```bash
python - <<'PY'
from pathlib import Path
from benchmarks.cache.config import load_suite_config
paths = sorted(Path('benchmarks/cache/configs').glob('issue15-*.yaml'))
assert len(paths) == 4, paths
for path in paths:
    cfg = load_suite_config(path)
    assert cfg.parallelism.tensor_parallel_size == 1
    assert cfg.workload.request_rate == ['inf']
    print('config-ok', path)
PY
```

- [ ] **Step 5: Dry-run all four configs before hardware work.**

Redirect verbose output to files and inspect generated `scenarios.json` for intended eviction-restore cases only. Do not start expensive runs yet.

- [ ] **Step 6: Commit.**

```bash
git add benchmarks/cache/configs/issue15-*.yaml
git commit -m "bench: add issue 15 validation configs"
```

---

### Task 9: Complete local implementation verification before GPU measurement

**Files:** no new files unless verification fixes are required.

- [ ] **Step 1: Run both stdlib test modules.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_build_generalization_dataset.py' -v
```

- [ ] **Step 2: Run focused compile/static checks.**

```bash
python -m compileall -q \
  benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/evaluate_cost_model_generalization.py \
  benchmarks/cache/build_generalization_dataset.py \
  benchmarks/cache/tests/test_cost_model_generalization.py \
  benchmarks/cache/tests/test_build_generalization_dataset.py
git diff --check
```

If Ruff is installed, run `ruff check` and `ruff format --check` on the five Python paths above. If Ruff is absent, record that and rely on GitHub Actions; do not install unrelated tooling merely for convenience.

- [ ] **Step 3: Verify changed-file scope.**

```bash
git diff --name-only origin/main...HEAD | sort
```

Stop if an active runtime/scheduler/inference file appears unexpectedly.

---

### Task 10: Run Phase 0 provenance and feasibility preflight

**Files:** raw results only under `/code/results/cache`.

- [ ] **Step 1: Capture locale-independent environment evidence.**

Use existing `collect_environment_evidence()` plus compact commands for `lscpu --json`, `numactl --hardware`, `nvidia-smi` UUID/topology, memory, `findmnt -T /tmp/vllm-kv-cache`, Python/vLLM version, worktree HEAD/status.

- [ ] **Step 2: Verify control GPU identity.**

GPU0 must match `GPU-5516e45d-3e50-69ef-f0f2-8ecff465beea` to claim same-GPU control. Record any difference before continuing.

- [ ] **Step 3: Run only the 7B/C1 1024 sentinel paths.**

Use dry-run `scenarios.json` to select `eviction-restore`, prompt 1024, concurrency 1 case IDs. CPU-primary comes from the 8 GiB CPU config; filesystem comes from the 2 GiB filesystem config; no-cache recompute may come from either run after workload SHA equality is verified.

- [ ] **Step 4: Treat the C1 sentinel only as drift/provenance evidence.**

Do not put it into the formal generalization aggregate.

- [ ] **Step 5: Preflight 14B deterministic workload generation for all five anchors.**

Do not reseed a failed anchor.

- [ ] **Step 6: Preflight 14B CPU-primary at requested 1024 with 8 GiB CPU.**

Require positive external KV tokens and positive CPU-to-GPU transfer count/bytes. If it recomputes because CPU capacity is insufficient, preserve that failure and post a material-deviation comment before changing capacity. The comment must state old/new capacity and that the adjustment exists only to preserve `cpu_primary` source semantics.

- [ ] **Step 7: Preflight 14B filesystem at requested 1024.**

Require external KV, CPU-to-GPU transfer, and async lower-tier lookup evidence. If the intended tier is not reached, preserve evidence and comment before changing pressure/capacity.

---

### Task 11: Select the minimum materially contended C-load

**Files:** raw sentinel runs plus `/code/results/cache/issue15-selection.json` outside Git.

- [ ] **Step 1: Run the 1024 sentinel at C2.**

Measure recompute, CPU-primary, and filesystem paths with identical workload identity.

- [ ] **Step 2: Compute relative P95 changes against C1.**

```python
relative_change = abs(candidate_p95 - c1_p95) / c1_p95
```

C2 qualifies only when one principal path is >= 0.20, another is >= 0.10, and restore provenance remains valid.

- [ ] **Step 3: If C2 does not qualify, run C4 and apply the identical rule.**

- [ ] **Step 4: If C4 does not qualify, run C8 and apply the identical rule.**

- [ ] **Step 5: Stop at the first qualifying concurrency and write selection JSON.**

Use:

```json
{
  "schema_version": 1,
  "selected_concurrency": 4,
  "selection_reason": "first candidate satisfying pre-registered contention gate"
}
```

The numeric example is replaced by the script using the actually selected candidate; no human edits the selection file after measurement.

- [ ] **Step 6: If C8 still does not qualify, stop expansion.**

Write `selected_concurrency` as JSON null and reason `no_material_contention_through_c8`; add an Issue #15 decision-log comment. Do not run C16/C32.

- [ ] **Step 7: Generate formal 7B configs outside Git from selection JSON.**

A Python script reads `issue15-selection.json`, errors if concurrency is null, copies the two sentinel YAML mappings, replaces prompt list with `[128, 192, 256, 1024, 4096]`, replaces concurrency with a one-element list containing the selected integer, and writes the two configs under `/code/results/cache/issue15-configs/`. Record SHA256 values.

---

### Task 12: Run formal C-load and C-model measurements and build condition datasets

**Files:** raw runs plus `/code/results/cache/issue15-structured/c-load.json` and `c-model.json`.

- [ ] **Step 1: Run only formal C-load eviction-restore cases for the five anchors.**

Use `--case-id` selections from dry-run `scenarios.json`; do not execute unrelated workload kinds.

- [ ] **Step 2: Immediately build C-load structured data before evaluation.**

Set shell variables to the exact run directories just created, print them, then run:

```bash
python benchmarks/cache/build_generalization_dataset.py \
  --condition-id c-load \
  --recompute-run "$RECOMPUTE_RUN" \
  --cpu-run "$CPU_RUN" \
  --filesystem-run "$FILESYSTEM_RUN" \
  --percentile p95 \
  --output /code/results/cache/issue15-structured/c-load.json
```

- [ ] **Step 3: Validate C-load JSON with `load_generalization_condition()`.**

Require condition ID `c-load`, correct selected concurrency, explicit accepted/excluded rows, and positive per-request external tokens for accepted restores.

- [ ] **Step 4: Run the 14B/C1 formal five-anchor cases.**

Use only provenance-preserving capacity adjustments already logged during Task 10.

- [ ] **Step 5: Build and validate `c-model.json` before any evaluator run.**

Use the same builder and loader checks.

- [ ] **Step 6: Preserve first failures.**

Retry only operationally retryable failures. Never reseed deterministic workload-generation failures or rerun merely to improve aggregate metrics.

---

### Task 13: Evaluate frozen profiles, diagnose failures, and refine only local boundaries

**Files:** immutable intermediate evaluations under `/code/results/cache/issue15-structured/`.

- [ ] **Step 1: Evaluate C-load without diagnostics.**

```bash
python benchmarks/cache/evaluate_cost_model_generalization.py \
  --input /code/results/cache/issue15-structured/c-load.json \
  --profile benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json \
  --percentile p95 \
  --output /code/results/cache/issue15-structured/c-load-evaluation.json
```

- [ ] **Step 2: Evaluate C-model without diagnostics.**

Write `/code/results/cache/issue15-structured/c-model-evaluation.json`.

- [ ] **Step 3: Compute SHA256 for both primary evaluation files before any diagnostic run.**

- [ ] **Step 4: Run `--diagnose` only to separate output files for failed curves/conditions.**

Never overwrite primary evaluation JSON.

- [ ] **Step 5: Refine a boundary only when an approved trigger fires.**

Triggers are: P95 absolute actual margin <= 1 ms; adjacent formal anchors flip actual preferred path; or a clearly non-boundary wrong frozen prediction. Add only local anchors between implicated neighbors and stop once a reliable bracket/failure region exists.

- [ ] **Step 6: Add an Issue #15 comment at the first meaningful failure boundary or stop/expand decision.**

Record condition/tier, requested and external-token region, raw error/decision, scalar diagnostic result, implicated missing input, and whether the matrix stops.

---

### Task 14: Produce final evidence and Issue #16 eligibility handoff

**Files:**
- Create: `docs/engineering/validation/2026-08-11-issue15-generalization-validation.json`
- Create: `docs/engineering/validation/2026-08-11-issue15-generalization-validation.md`
- Create: `docs/engineering/handoffs/2026-08-11-issue16-active-decision-handoff.md`
- Modify: `docs/engineering/CURRENT_STATE.md`
- Modify: `docs/engineering/README.md` only if needed for index completeness.

- [ ] **Step 1: Build final JSON from immutable primary evaluations and diagnostics.**

Include design/plan paths, worktree base provenance, archived C0 summary, C-load selection trace, both new-condition metadata, accepted/excluded samples, primary evaluation SHA256 values, high/low-confidence partitions, diagnostics, parameter classifications, failure boundaries, candidate required features, and eligibility map.

- [ ] **Step 2: Write the validation report to answer all ten #15 research questions.**

Explicitly answer recompute transfer, CPU restore transfer, machine/memory/PCIe/NUMA implications, secondary path specificity, concurrency scale-vs-shape behavior, fixed parameters, online-calibrated parameters, low-cardinality runtime observations, low-confidence/systematic-error regions, and whether a new feature should precede a wider sweep.

- [ ] **Step 3: Write Issue #16 handoff as a bounded eligibility map.**

Each entry states model/load region, source tier, observed external-token region, fixed-profile verdict, confidence, active-design eligibility (`eligible`, `ineligible`, or `needs_online_calibration`), and reason. Low-confidence extrapolation and failed regions are never eligible.

- [ ] **Step 4: Update current-state docs only to facts supported by evidence.**

Do not describe #15 as completed until the close criteria are actually met.

- [ ] **Step 5: Run final local verification.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_build_generalization_dataset.py' -v
python -m compileall -q benchmarks/cache
python - <<'PY'
import json
from pathlib import Path
paths = [
    Path('benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json'),
    Path('docs/engineering/validation/2026-08-11-issue15-generalization-validation.json'),
]
for path in paths:
    json.loads(path.read_text(encoding='utf-8'))
    print('json-ok', path)
PY
git diff --check
git diff --name-only origin/main...HEAD | sort
```

Run targeted Ruff checks if available; do not install pytest.

- [ ] **Step 6: Commit final local evidence but do not push.**

```bash
git add benchmarks/cache docs/engineering docs/superpowers
git commit -m "validate: record issue 15 cost model generalization"
```

---

### Task 15: Deliver through GitHub, run authoritative CI, and preserve merge gate

**Files:** GitHub branch/PR metadata plus exact verified contents from the Pod worktree.

- [ ] **Step 1: Transfer verified content through the GitHub-authoritative path.**

Use the established content-addressed handoff pattern if needed. Verify delivered file content hashes against the Pod versions.

- [ ] **Step 2: Open the feature PR as Draft.**

Use `Closes #15` only if all close criteria are present. Summarize frozen-profile semantics, selected load, 14B evidence, error/decision comparison, transfer/scale/shape classification, failure boundaries, Issue #16 eligibility, and the no-active-enforcement scope boundary.

- [ ] **Step 3: Run GitHub Actions and report only checks that actually ran.**

Repository-wide pre-commit is authoritative. Do not claim pytest ran unless the workflow shows it.

- [ ] **Step 4: Fix CI within scope and re-verify semantics.**

Any CI repair that changes experiment interpretation must first be logged on Issue #15. Pure formatting/SPDX/type hygiene does not need issue-level noise.

- [ ] **Step 5: Add final Issue #15 decision-log comment.**

Link PR, structured JSON, report, handoff, final classifications, and ineligible regions.

- [ ] **Step 6: Stop before merge and request fresh explicit authorization.**

No prior design/implementation/PR/CI authorization counts as merge authorization.

---

## Self-Review Coverage Map

- Frozen Issue #14 profile identity: Task 2.
- Runtime external tokens versus requested anchors: Tasks 3 and 7.
- Frozen holdout without derive/recalibrate: Tasks 4 and 6.
- High-confidence gate and `insufficient_evidence`: Task 4.
- One-scalar diagnostic without rewriting primary evidence: Tasks 5 and 13.
- Workload fairness and actual restore provenance: Task 7.
- Pre-registered config/tier capacities: Tasks 8 and 10.
- Minimum C2/C4/C8 contention selection: Task 11.
- Qwen2.5-14B/C1 model-scale axis: Tasks 8, 10, and 12.
- Five formal anchors: Tasks 8 and 12.
- Boundary-only refinement: Task 13.
- Machine-readable results and validation report: Task 14.
- Transferable/environment-specific/missing-feature classification: Tasks 5, 13, and 14.
- Bounded Issue #16 eligibility: Task 14.
- Issue comment decision journal: Tasks 10, 11, 13, and 15.
- No active enforcement and explicit merge authorization: Global Constraints and Task 15.
