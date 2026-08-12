# Issue #15 Generalization Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce fixed-profile holdout evidence for the Issue #14 KV restore/recompute cost model under one materially contended Qwen2.5-7B load condition and one Qwen2.5-14B model-scale condition, then classify each principal curve as transferable, environment-scale-sensitive, or shape/missing-feature limited without changing active runtime behavior.

**Architecture:** Preserve the Issue #14 calibration path unchanged. Add an Issue #15-only pure generalization module and CLI that evaluate a supplied frozen profile against a structured condition dataset. Add a dataset builder that converts existing cache-benchmark run artifacts into the stable Issue #15 schema with explicit workload fairness, external-token normalization, tier provenance, and GPU identity. Hardware execution remains shadow-only and uses the existing `benchmarks/cache/run_suite.py` stack.

**Tech Stack:** Python 3.11, stdlib `dataclasses`/`json`/`statistics`/`unittest`, existing `benchmarks.cache.cost_model_calibration` primitives, existing `OffloadCostModel`, existing cache benchmark YAML/Pydantic pipeline, GitHub repository-wide pre-commit as authoritative CI.

## Global Constraints

- GitHub is the authoritative remote and write path; the Pod is for build, focused tests, benchmark, and hardware validation.
- Never run `git push` from the Pod.
- Never use `git clean -fd`, `reset --hard`, or a checkout operation that can remove the three intentional untracked local YAML files under `/code/vllm/benchmarks/cache/configs/`.
- Do not install pytest for this work. New focused tests must run with stdlib `unittest`; GitHub CI may also collect them through pytest if the workflow actually does so.
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

Create:

- `benchmarks/cache/profiles/issue14-shadow-cost-calibrated.json` — immutable profile copied programmatically from the checked-in Issue #14 `calibrated_profile` field.
- `benchmarks/cache/cost_model_generalization.py` — condition loader, frozen evaluation, high-confidence gate, low-confidence partition, one-scalar diagnostics.
- `benchmarks/cache/evaluate_cost_model_generalization.py` — deterministic frozen-profile CLI; never derives a new profile.
- `benchmarks/cache/build_generalization_dataset.py` — converts run-suite artifacts into condition JSON and validates workload/tier/GPU provenance.
- `benchmarks/cache/tests/test_cost_model_generalization.py` — stdlib tests for profile fidelity, loading, frozen evaluation, gate semantics, diagnostics, CLI determinism.
- `benchmarks/cache/tests/test_build_generalization_dataset.py` — stdlib tests for run pairing and provenance validation.
- `benchmarks/cache/tests/test_issue15_environment_provenance.py` — stdlib test that the benchmark environment capture records GPU index and UUID.
- `benchmarks/cache/configs/issue15-7b-load-sentinel-cpu.yaml`
- `benchmarks/cache/configs/issue15-7b-load-sentinel-fs.yaml`
- `benchmarks/cache/configs/issue15-14b-formal-cpu.yaml`
- `benchmarks/cache/configs/issue15-14b-formal-fs.yaml`
- `docs/engineering/validation/2026-08-11-issue15-generalization-validation.json`
- `docs/engineering/validation/2026-08-11-issue15-generalization-validation.md`
- `docs/engineering/handoffs/2026-08-11-issue16-active-decision-handoff.md`

Modify:

- `benchmarks/cache/metrics.py` — only expand benchmark environment provenance from `name,memory.total,driver_version` to `index,uuid,name,memory.total,driver_version`.
- `docs/engineering/CURRENT_STATE.md` after complete evidence exists.
- `docs/engineering/README.md` only if the validation/handoff index requires new entries.

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

### Task 3: Make benchmark GPU provenance sufficient for Issue #15

**Files:**

- Modify: `benchmarks/cache/metrics.py`
- Create: `benchmarks/cache/tests/test_issue15_environment_provenance.py`

**Interfaces:**

- Consumes: existing `_ENVIRONMENT_COMMANDS` and `collect_environment_evidence()`.
- Produces: `environment.json` GPU inventory lines containing physical index and UUID as well as name/memory/driver.

- [ ] **Step 1: Write the RED stdlib test.**

```python
import unittest

from benchmarks.cache.metrics import _ENVIRONMENT_COMMANDS


class Issue15EnvironmentProvenanceTests(unittest.TestCase):
    def test_gpu_inventory_captures_index_and_uuid(self) -> None:
        command = _ENVIRONMENT_COMMANDS['gpu_inventory']
        self.assertIn(
            '--query-gpu=index,uuid,name,memory.total,driver_version',
            command,
        )
```

- [ ] **Step 2: Run RED.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_issue15_environment_provenance.py' -v
```

Expected: FAIL because the current query omits index and UUID.

- [ ] **Step 3: Make the minimal provenance-only change.**

Change only the query string in `_ENVIRONMENT_COMMANDS['gpu_inventory']` to:

```text
--query-gpu=index,uuid,name,memory.total,driver_version
```

Do not modify resource sampling or runtime metrics.

- [ ] **Step 4: Run GREEN and commit.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_issue15_environment_provenance.py' -v
git add benchmarks/cache/metrics.py \
  benchmarks/cache/tests/test_issue15_environment_provenance.py
git commit -m "bench: capture GPU UUID in cache provenance"
```

---

### Task 4: Add a neutral Issue #15 condition loader

**Files:**

- Create: `benchmarks/cache/cost_model_generalization.py`
- Modify: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:**

- Produces `GeneralizationCondition` and `load_generalization_condition(path: Path, percentile: str = "p95")`.
- Reuses existing `DecisionSample` and `CalibrationDataset` rather than changing #14 calibration semantics.

- [ ] **Step 1: Add a RED loader test with a complete two-source fixture.**

Use condition metadata for `c-model`, Qwen2.5-14B, concurrency 1, requests-per-case 8, TP1, GPU UUID `GPU-test`, and run paths. Add one `cpu_primary` and one `secondary:filesystem` row at requested 256. Each row has total external tokens 1856, per-request external tokens 232, P95 recompute 30 ms, source-specific restore latency, transfer evidence, and identical workload SHA fields. Assert the internal decision samples use `external_tokens == 232`. Add a second test that changes total external tokens to 1857 and expects `ValueError`.

- [ ] **Step 2: Run RED.**

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the dataclass and loader with the following validation flow.**

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
```

`load_generalization_condition()` must require schema 1 / issue 15, positive requests-per-case, sources only in `{cpu_primary, secondary:filesystem}`, positive total/per-request external tokens, exact `total == requests_per_case * per_request`, supported percentile in `{p50,p95,p99}`, and sorted `DecisionSample` output. It builds `CalibrationDataset` with empty repeat checks and preserves sample metadata/exclusions separately.

- [ ] **Step 4: Run GREEN and commit.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
git add benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "feat: load cost model generalization conditions"
```

---

### Task 5: Implement frozen-profile evaluation, gate semantics, and scale diagnostics

**Files:**

- Modify: `benchmarks/cache/cost_model_generalization.py`
- Modify: `benchmarks/cache/tests/test_cost_model_generalization.py`

**Interfaces:**

- Produces `evaluate_frozen_condition(condition, profile, *, profile_identity)`.
- Produces `diagnose_curve_scaling(evaluation)`.
- Calls existing `evaluate_profile()` only; never calls `derive_calibrated_profile()`.

- [ ] **Step 1: Add RED tests proving supplied-profile behavior.**

Use a deliberately wrong supplied profile and assert the wrong decision remains wrong, result mode is `frozen_profile_holdout`, profile identity is preserved, and no `calibrated_profile` key exists.

- [ ] **Step 2: Add RED tests for the pre-registered gate.**

Cover:

1. all three principal curves have high-confidence evidence and all gates pass -> `fixed_profile_transfer_pass`;
2. any principal curve has zero high-confidence evidence -> `insufficient_evidence`;
3. macro <= 15% but one curve > 20% -> `fixed_profile_transfer_fail`;
4. wrong high-confidence decision with absolute actual margin > 1 ms -> fail;
5. boundary-sensitive wrong row remains counted in decision accuracy;
6. low-confidence rows are reported but do not satisfy evidence-presence requirements.

- [ ] **Step 3: Implement constants and high-confidence aggregation.**

```python
DECISION_ACCURACY_MIN = 0.95
PRINCIPAL_MACRO_MAPE_MAX = 15.0
PRINCIPAL_CURVE_MAPE_MAX = 20.0
BOUNDARY_MARGIN_MS = 1.0
```

Call `evaluate_profile(condition.dataset, profile)` once. Recompute the gate aggregate from high-confidence rows only. Deduplicate recompute errors by `(requested_tokens, external_tokens)` exactly as #14 does. Equal-weight the recompute, CPU restore, and filesystem restore MAPE values.

Verdict order is exact:

```text
missing principal high-confidence evidence -> insufficient_evidence
accuracy < 0.95 -> fixed_profile_transfer_fail
macro MAPE > 15 -> fixed_profile_transfer_fail
any principal curve MAPE > 20 -> fixed_profile_transfer_fail
any wrong high-confidence decision with abs(actual margin) > 1 ms -> fixed_profile_transfer_fail
otherwise -> fixed_profile_transfer_pass
```

- [ ] **Step 4: Add and implement one-scalar diagnostics.**

For each principal curve:

```python
scale = statistics.median(actual / predicted for actual, predicted in points)
residual_mape = 100.0 * statistics.fmean(
    abs(predicted * scale - actual) / actual
    for actual, predicted in points
)
```

Classify raw MAPE <=15% as `transferable`; raw >15% with residual <=15% as `environment_specific_scale_candidate`; residual >15% as `curve_shape_or_missing_feature`. Diagnostics never replace primary predictions or verdict.

- [ ] **Step 5: Run GREEN and commit.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
git add benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/tests/test_cost_model_generalization.py
git commit -m "feat: evaluate and diagnose frozen cost profiles"
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

Call `main(args)` twice with identical fixture inputs and assert byte-identical JSON. Assert `--check` exits 1 for any verdict except `fixed_profile_transfer_pass`.

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

Print one compact summary with condition, high-confidence decision correct/total, accuracy, macro-MAPE, and classification.

- [ ] **Step 3: Run GREEN and characterize #14 CLI unchanged.**

Do not modify `benchmarks/cache/evaluate_cost_model.py`. Run the existing checked-in #13 evaluator offline and confirm its current `after:` behavior remains.

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
- Model, concurrency, request rate, requests-per-case, TP size, selected GPU index, GPU UUID, environment path, and run paths are inferred from manifests/environment/records.

- [ ] **Step 1: Add synthetic run fixtures with stdlib tempfile/json.**

Each fixture run contains `manifest.json`, `environment.json`, `scenario-results.jsonl`, and raw `metadata.json`. Manifest server env contains `CUDA_VISIBLE_DEVICES: "0"`. Environment GPU inventory contains two CSV lines with physical indexes and UUIDs. Use completed `eviction-restore` records for no-cache, CPU-offload, and tiered-fs with identical paired workload SHA values.

- [ ] **Step 2: Add RED acceptance/rejection tests.**

Accept CPU-primary only with positive external KV tokens and positive CPU-to-GPU transfer count/bytes. Accept filesystem only with those plus positive async tiering lookup evidence. Hard-fail workload SHA mismatch, model/concurrency/request-rate/requests-per-case mismatch, missing explicit CUDA visibility, invalid GPU index, or environment inventory lacking the selected GPU UUID. Record configured-mode/no-transfer and incomplete benchmark rows as explicit exclusions.

- [ ] **Step 3: Implement selected-GPU provenance.**

Read `manifest['config']['server']['env']['CUDA_VISIBLE_DEVICES']`; require exactly one numeric index for #15. Parse `environment['gpu_inventory']['stdout']` CSV rows emitted by Task 3, locate the matching physical index, and store its UUID. Never infer GPU identity from line order.

- [ ] **Step 4: Implement workload pairing.**

Select only `record['workload_kind'] == 'eviction-restore'`, pair by `(prompt_tokens, concurrency, request_rate)`, and require expected cache modes. Read each `workload_metadata` JSON and require identical measure/populate SHA values before comparing latencies.

- [ ] **Step 5: Implement transfer evidence extraction.**

Read `record['normalized']['prometheus']['delta']`. Sum positive values for external KV token samples labeled `source="external_kv_transfer"`, CPU-to-GPU transfer count, and CPU-to-GPU total bytes. For filesystem rows also require positive async tiering lookup count/sum evidence from normalized cache/Prometheus data.

- [ ] **Step 6: Normalize external tokens and emit schema 1 / issue 15.**

```python
if external_total % requests_per_case != 0:
    raise ValueError('external KV tokens not divisible by requests_per_case')
external_per_request = external_total // requests_per_case
```

Write both total and per-request values, pair the same no-cache recompute latency with each accepted source row, preserve workload hashes and transfer evidence, and emit explicit exclusions.

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

**Interfaces:** Existing strict `load_suite_config()` and `run_suite.py`; no benchmark-core change.

- [ ] **Step 1: Add explicit GPU pinning to every #15 config.**

Under `server.env`, preserve existing environment values and add:

```yaml
CUDA_VISIBLE_DEVICES: "0"
```

This value is part of manifest provenance and must match the selected physical index parsed from `environment.json`.

- [ ] **Step 2: Create 7B CPU sentinel config from `local-crossover.yaml`.**

Keep control model, TP1, GPU KV 2 GiB, pressure fill 65536, requests-per-case 8, request rate inf, seed 1, output 1, tolerance 2. Set CPU tier 8 GiB, filesystem disabled, prompt list `[1024]`, concurrency `[1, 2, 4, 8]`, shared-prefix ratios `[0.0]`.

- [ ] **Step 3: Create 7B filesystem sentinel config.**

Same controls, CPU tier 2 GiB, filesystem enabled, prompt `[1024]`, concurrency `[1, 2, 4, 8]`.

- [ ] **Step 4: Create 14B CPU and filesystem formal configs.**

Use model/tokenizer `/mnt/model/Qwen2.5-14B-Instruct`, served name `qwen2.5-14b`, TP1, C1, formal prompts `[128, 192, 256, 1024, 4096]`, shared-prefix ratios `[0.0]`. CPU config starts at 8 GiB with filesystem disabled; filesystem config uses 2 GiB with filesystem enabled. Preserve other control settings.

- [ ] **Step 5: Validate all four configs through the real loader.**

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
    assert cfg.server.env['CUDA_VISIBLE_DEVICES'] == '0'
    print('config-ok', path)
PY
```

- [ ] **Step 6: Dry-run all four configs before hardware work.**

Redirect verbose output to files and inspect generated `scenarios.json` for intended `eviction-restore` cases. Do not start expensive runs yet.

- [ ] **Step 7: Commit.**

```bash
git add benchmarks/cache/configs/issue15-*.yaml
git commit -m "bench: add issue 15 validation configs"
```

---

### Task 9: Complete local implementation verification before GPU measurement

**Files:** no new files unless verification fixes are required.

- [ ] **Step 1: Run all three new stdlib test modules.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_issue15_environment_provenance.py' -v
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_cost_model_generalization.py' -v
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_build_generalization_dataset.py' -v
```

- [ ] **Step 2: Run focused compile/static checks.**

```bash
python -m compileall -q \
  benchmarks/cache/metrics.py \
  benchmarks/cache/cost_model_generalization.py \
  benchmarks/cache/evaluate_cost_model_generalization.py \
  benchmarks/cache/build_generalization_dataset.py \
  benchmarks/cache/tests/test_issue15_environment_provenance.py \
  benchmarks/cache/tests/test_cost_model_generalization.py \
  benchmarks/cache/tests/test_build_generalization_dataset.py
git diff --check
```

If Ruff is installed, run `ruff check` and `ruff format --check` on these seven Python paths. If Ruff is absent, record it and rely on GitHub Actions; do not install unrelated tooling for convenience.

- [ ] **Step 3: Verify changed-file scope.**

```bash
git diff --name-only origin/main...HEAD | sort
```

Stop if an active runtime/scheduler/inference file appears unexpectedly.

---

### Task 10: Run Phase 0 provenance, workload, and tier feasibility preflight

**Files:** raw results only under `/code/results/cache`.

- [ ] **Step 1: Capture environment provenance through the updated benchmark collector.**

Require `environment.json` to show GPU index/UUID, GPU topology, `lscpu --json`, NUMA topology, Python/vLLM version, git HEAD/status. Independently record memory and `findmnt -T /tmp/vllm-kv-cache -o TARGET,SOURCE,FSTYPE,OPTIONS` in the experiment notes. Do not infer physical NVMe from overlay storage.

- [ ] **Step 2: Verify selected GPU identity.**

Manifest must contain `CUDA_VISIBLE_DEVICES="0"`; environment inventory physical index 0 must map to UUID `GPU-5516e45d-3e50-69ef-f0f2-8ecff465beea` to claim same-GPU control. Record any difference before continuing.

- [ ] **Step 3: Preflight deterministic 14B workload generation without a server.**

Use the 14B filesystem config and this exact pattern:

```bash
python - <<'PY'
from pathlib import Path
from benchmarks.cache.config import create_owned_directory, load_suite_config
from benchmarks.cache.run_suite import load_tokenizer
from benchmarks.cache.scenarios import CacheMode, build_execution_cases
from benchmarks.cache.workload import generate_workload

cfg = load_suite_config(Path('benchmarks/cache/configs/issue15-14b-formal-fs.yaml'))
run_dir = create_owned_directory(
    Path('/code/results/cache/issue15-workload-preflight'),
    cfg.results.root_dir,
)
(run_dir / 'raw').mkdir(exist_ok=True)
tokenizer = load_tokenizer(cfg)
cases = build_execution_cases(cfg, run_dir)
targets = [
    case for case in cases
    if case.cache_mode is CacheMode.NO_CACHE
    and case.workload_kind == 'eviction-restore'
    and case.concurrency == 1
]
assert sorted(case.prompt_tokens for case in targets) == [128, 192, 256, 1024, 4096]
for case in targets:
    artifacts = generate_workload(case, cfg, tokenizer)
    print('workload-ok', case.prompt_tokens, artifacts.metadata_path)
PY
```

Do not reseed a failed anchor.

- [ ] **Step 4: Run only the 7B/C1 1024 sentinel paths.**

Use dry-run `scenarios.json` to select `eviction-restore`, prompt 1024, concurrency 1 case IDs. CPU-primary comes from 8 GiB CPU config; filesystem comes from 2 GiB filesystem config; no-cache recompute is paired only after workload SHA equality is verified. Sentinel data are drift/provenance evidence only, not formal aggregate input.

- [ ] **Step 5: Preflight 14B CPU-primary at requested 1024 with 8 GiB CPU.**

Require positive external KV tokens and positive CPU-to-GPU transfer count/bytes. If it recomputes because CPU capacity is insufficient, preserve that failure and post a material-deviation Issue #15 comment before changing capacity; state old/new capacity and that the change exists only to preserve `cpu_primary` source semantics.

- [ ] **Step 6: Preflight 14B filesystem at requested 1024.**

Require external KV, CPU-to-GPU transfer, and positive async lower-tier lookup evidence. If the intended tier is not reached, preserve evidence and comment before changing pressure/capacity.

---

### Task 11: Select the minimum materially contended C-load

**Files:** raw sentinel runs plus `/code/results/cache/issue15-selection.json` outside Git.

- [ ] **Step 1: Run requested-1024 sentinel at C2.**

Measure recompute, CPU-primary, and filesystem paths with identical workload identity.

- [ ] **Step 2: Compute relative P95 changes against C1.**

```python
relative_change = abs(candidate_p95 - c1_p95) / c1_p95
```

C2 qualifies only when one principal path is >=0.20, another is >=0.10, and restore provenance remains valid.

- [ ] **Step 3: If C2 does not qualify, run C4 and apply the identical rule.**

- [ ] **Step 4: If C4 does not qualify, run C8 and apply the identical rule.**

- [ ] **Step 5: Stop at the first qualifying concurrency and write selection JSON programmatically.**

The script writes schema version 1, the selected integer, candidate-relative-change evidence, and reason `first_candidate_satisfying_pre_registered_contention_gate`.

- [ ] **Step 6: If C8 still does not qualify, stop expansion.**

Write JSON null for selected concurrency with reason `no_material_contention_through_c8`; add an Issue #15 decision-log comment. Do not run C16/C32.

- [ ] **Step 7: Generate formal 7B configs outside Git from selection JSON.**

A Python script reads the selected integer, copies the two sentinel YAML mappings, replaces prompt list with `[128, 192, 256, 1024, 4096]`, replaces concurrency with a one-element list containing the selected value, writes under `/code/results/cache/issue15-configs/`, and records SHA256 values. It exits non-zero if selected concurrency is JSON null.

---

### Task 12: Run formal C-load and C-model measurements and build condition datasets

**Files:** raw runs plus `/code/results/cache/issue15-structured/c-load.json` and `c-model.json`.

- [ ] **Step 1: Run only formal C-load `eviction-restore` cases for the five anchors.**

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

Require condition ID `c-load`, selected concurrency, GPU UUID matching the manifest-selected physical index, explicit accepted/excluded rows, and positive per-request external tokens for accepted restores.

- [ ] **Step 4: Run Qwen2.5-14B/C1 formal five-anchor cases.**

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

- [ ] **Step 3: Compute SHA256 for both primary evaluation files before diagnostics.**

- [ ] **Step 4: Run `--diagnose` only to separate output files for failed curves/conditions.**

Never overwrite primary evaluation JSON.

- [ ] **Step 5: Refine a boundary only when an approved trigger fires.**

Triggers are: P95 absolute actual margin <=1 ms; adjacent formal anchors flip actual preferred path; or a clearly non-boundary wrong frozen prediction. Add only local anchors between implicated neighbors and stop once a reliable bracket/failure region exists.

- [ ] **Step 6: Add an Issue #15 comment at the first meaningful failure boundary or stop/expand decision.**

Record condition/tier, requested and external-token region, raw error/decision, scalar diagnostic result, implicated missing input, and whether the matrix stops.

---

### Task 14: Produce final evidence and Issue #16 eligibility handoff

**Files:**

- Create: `docs/engineering/validation/2026-08-11-issue15-generalization-validation.json`
- Create: `docs/engineering/validation/2026-08-11-issue15-generalization-validation.md`
- Create: `docs/engineering/handoffs/2026-08-11-issue16-active-decision-handoff.md`
- Modify: `docs/engineering/CURRENT_STATE.md`
- Modify: `docs/engineering/README.md` only if index completeness requires it.

- [ ] **Step 1: Build final JSON from immutable primary evaluations and diagnostics.**

Include design/plan paths, worktree base provenance, archived C0 summary, C-load selection trace, both new-condition metadata, accepted/excluded samples, primary evaluation SHA256 values, high/low-confidence partitions, diagnostics, parameter classifications, failure boundaries, candidate required features, and eligibility map.

- [ ] **Step 2: Write the validation report to answer all ten #15 research questions.**

Explicitly answer recompute transfer, CPU restore transfer, machine/memory/PCIe/NUMA implications, secondary path specificity, concurrency scale-vs-shape behavior, fixed parameters, online-calibrated parameters, low-cardinality runtime observations, low-confidence/systematic-error regions, and whether a new feature should precede a wider sweep.

- [ ] **Step 3: Write Issue #16 handoff as a bounded eligibility map.**

Each entry states model/load region, source tier, observed external-token region, fixed-profile verdict, confidence, active-design eligibility (`eligible`, `ineligible`, or `needs_online_calibration`), and reason. Low-confidence extrapolation and failed regions are never eligible.

- [ ] **Step 4: Update current-state docs only to facts supported by evidence.**

Do not describe #15 as completed until close criteria are met.

- [ ] **Step 5: Run final local verification.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_issue15_environment_provenance.py' -v
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

Use the established content-addressed handoff pattern if needed. Verify delivered file content hashes against Pod versions.

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
- GPU index/UUID and explicit CUDA visibility provenance: Tasks 3, 7, 8, and 10.
- Runtime external tokens versus requested anchors: Tasks 4 and 7.
- Frozen holdout without derive/recalibrate: Tasks 5 and 6.
- High-confidence gate and `insufficient_evidence`: Task 5.
- One-scalar diagnostic without rewriting primary evidence: Tasks 5 and 13.
- Workload fairness and actual restore provenance: Task 7.
- Pre-registered config/tier capacities: Tasks 8 and 10.
- Minimum C2/C4/C8 contention selection: Task 11.
- Qwen2.5-14B/C1 model-scale axis: Tasks 8, 10, and 12.
- Five formal anchors and deterministic workload preflight: Tasks 8, 10, and 12.
- Boundary-only refinement: Task 13.
- Machine-readable results and validation report: Task 14.
- Transferable/environment-specific/missing-feature classification: Tasks 5, 13, and 14.
- Bounded Issue #16 eligibility: Task 14.
- Issue comment decision journal: Tasks 10, 11, 13, and 15.
- No active enforcement and explicit merge authorization: Global Constraints and Task 15.
