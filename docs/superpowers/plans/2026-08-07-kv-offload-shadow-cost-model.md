# KV Offload Shadow Cost Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in shadow model that distinguishes CPU-primary restores from secondary-tier promotions, predicts restore versus recompute cost, calibrates secondary promotion cost online, and records predictions without changing execution.

**Architecture:** Put pure curve/config/EWMA logic in `vllm/v1/kv_offload/cost_model.py`. `TieringOffloadingManager` owns logical provenance and promotion observations; `OffloadingConnectorScheduler` reads the final matched-prefix provenance and records the shadow decision. Reuse `OffloadingConnectorStats` and existing Prometheus plumbing.

**Tech Stack:** Python 3.11, dataclasses, vLLM V1 KV offloading/tiering, pytest, `unittest.mock`, Qwen2.5-7B cache benchmark suite.

## Global Constraints

- Shadow is opt-in; default is off.
- Do not add `LookupResult` values.
- Do not change Attention, PagedAttention, CUDA graphs, kernels, hashing, or KV layout.
- Do not online-learn recompute or CPU-primary restore cost in v1.
- Benchmark curves have no production defaults.
- Equal estimated costs choose `recompute`.
- No request ID, path, block hash, or device identifier in Prometheus labels.
- Shadow must not change matched-token count, async flag, lookup outcome, allocation, transfer jobs, cache contents, or actual restore path.
- `mode: enforce` is rejected by v1 config validation.
- Invalid config is a startup error. Runtime shadow/provenance/EWMA/metric exceptions are fail-open: log them and continue the existing request path.
- Branch from `main` after PR #6 merge `bcbb26fa8ed90d2bd1de57f70168ec3d188c8c9c`; do not merge benchmark PR #5 into the runtime branch.

---

### Task 1: Pure Cost Model

**Files:**

- Create: `vllm/v1/kv_offload/cost_model.py`
- Create: `tests/v1/kv_offload/test_cost_model.py`

**Produces:**

```python
Confidence = Literal["high", "low"]
PreferredPath = Literal["restore", "recompute"]

@dataclass(frozen=True, slots=True)
class CurveEstimate:
    value_ms: float
    confidence: Confidence

@dataclass(frozen=True, slots=True)
class LoadProvenance:
    source: str
    external_tokens: int
    secondary_promoted_tokens: int | None
    sources: tuple[str, ...]
    confidence: Confidence
    lookup_sync_seconds: float | None = None
    lookup_async_seconds: float | None = None

@dataclass(frozen=True, slots=True)
class ShadowDecision:
    preferred: PreferredPath
    restore_seed_ms: float
    restore_estimate_ms: float
    recompute_estimate_ms: float
    runtime_scale: float
    confidence: Confidence

@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    tier_key: str
    token_bucket: int
    observed_ms: float
    seeded_ms: float
    sample_scale: float
    runtime_scale: float
```

- [ ] **Step 1: Write RED curve/default-off tests**

```python
import pytest

from vllm.v1.kv_offload.cost_model import CostCurve, OffloadCostModel


def test_curve_exact_interpolation_and_outside_confidence():
    curve = CostCurve.from_mapping({256: 20.0, 512: 40.0, 1024: 80.0})
    assert curve.estimate(512) == CurveEstimate(40.0, "high")
    assert curve.estimate(768).value_ms == pytest.approx(60.0)
    assert curve.estimate(768).confidence == "high"
    assert curve.estimate(128).value_ms == pytest.approx(10.0)
    assert curve.estimate(128).confidence == "low"
    assert curve.estimate(2048).value_ms == pytest.approx(160.0)
    assert curve.estimate(2048).confidence == "low"


def test_single_point_curve_scales_with_low_confidence_off_sample():
    curve = CostCurve.from_mapping({1024: 24.49})
    assert curve.estimate(1024).confidence == "high"
    assert curve.estimate(2048).value_ms == pytest.approx(48.98)
    assert curve.estimate(2048).confidence == "low"


def test_cost_model_is_off_by_default():
    assert OffloadCostModel.from_extra_config({}) is None
    assert OffloadCostModel.from_extra_config(
        {"cache_cost_model": {"mode": "off"}}
    ) is None
```

Also parameterize invalid shadow configs: missing/empty profile, token `0`, latency `0`, non-finite latency, alpha outside `(0,1]`, non-positive clamp, clamp min greater than max, and `mode="enforce"`; each raises `ValueError`.

- [ ] **Step 2: Run RED**

```bash
python -m pytest -q tests/v1/kv_offload/test_cost_model.py
```

Expected: module import failure.

- [ ] **Step 3: Implement parsing and `CostCurve`**

`from_mapping()` accepts integer keys and integer-like strings, rejects booleans, requires positive tokens and finite positive milliseconds, sorts samples, and rejects duplicate token counts after coercion.

Estimate rules are exact:

```text
exact sample     -> measured value, high confidence
inside range     -> linear interpolation, high confidence
below range      -> first_ms * tokens / first_tokens, low confidence
above range      -> last_ms * tokens / last_tokens, low confidence
```

`bucket_for(tokens)` returns the first sample token count `>= tokens`, else the last sample token count.

`from_extra_config()` accepts only `off` and `shadow`. Shadow requires `profile.recompute_ms` and at least one `profile.tiers.<key>.restore_ms`. `promotion_ms` is optional. Defaults: alpha `0.2`, clamp `[0.25, 4.0]`; curves have no defaults.

- [ ] **Step 4: Write RED decision/EWMA tests**

Use this test-only profile:

```python
PROFILE = {
    "cache_cost_model": {
        "mode": "shadow",
        "ewma_alpha": 0.2,
        "sample_scale_min": 0.25,
        "sample_scale_max": 4.0,
        "profile": {
            "recompute_ms": {256: 26.414, 512: 44.961, 1024: 81.705,
                             2048: 152.461, 4096: 308.424},
            "tiers": {
                "cpu_primary": {"restore_ms": {1024: 24.490}},
                "filesystem": {
                    "restore_ms": {256: 31.119, 512: 56.979, 1024: 108.132,
                                   2048: 244.266, 4096: 651.127},
                    "promotion_ms": {256: 13.916, 512: 35.230, 1024: 81.458,
                                     2048: 171.505, 4096: 498.874},
                },
            },
        },
    }
}
```

Concrete assertions:

```python
model = OffloadCostModel.from_extra_config(PROFILE)
assert model is not None

cpu = model.shadow_decide(
    LoadProvenance("cpu_primary", 1024, 0, ("cpu_primary",), "high")
)
assert cpu is not None
assert cpu.preferred == "restore"
assert cpu.restore_estimate_ms == pytest.approx(24.490)
assert cpu.recompute_estimate_ms == pytest.approx(81.705)

for tokens in (256, 512, 1024, 2048, 4096):
    fs = model.shadow_decide(
        LoadProvenance(
            "secondary:filesystem", tokens, tokens,
            ("secondary:filesystem",), "high"
        )
    )
    assert fs is not None
    assert fs.preferred == "recompute"

observation = model.observe_secondary_promotion("filesystem", 1024, 162.916)
assert observation is not None
assert observation.token_bucket == 1024
assert observation.sample_scale == pytest.approx(2.0)
assert observation.runtime_scale == pytest.approx(1.2)

clamped = OffloadCostModel.from_extra_config(PROFILE)
assert clamped is not None
huge = clamped.observe_secondary_promotion("filesystem", 1024, 8145.8)
assert huge is not None
assert huge.sample_scale == pytest.approx(4.0)
assert huge.runtime_scale == pytest.approx(1.6)
```

Add equal-cost -> recompute, bucket isolation, and mixed-source tests. Mixed source estimates every component at full external tokens, uses the maximum restore estimate, and forces low confidence; missing component curve returns `None`.

- [ ] **Step 5: Implement decision/EWMA**

Profile key mapping:

```python
def _profile_key(source: str) -> str | None:
    if source == "cpu_primary":
        return "cpu_primary"
    if source.startswith("secondary:"):
        return source.removeprefix("secondary:")
    return None
```

For secondary source, runtime scale is keyed by `(tier_key, promotion_curve.bucket_for(tokens))`, default `1.0`.

Approved equation:

```python
restore_estimate_ms = restore_seed_ms * runtime_scale
```

EWMA:

```python
sample_scale = observed_ms / promotion_seed_ms
sample_scale = min(max(sample_scale, sample_scale_min), sample_scale_max)
new_scale = alpha * sample_scale + (1.0 - alpha) * old_scale
```

`observe_secondary_promotion()` returns `None` if the tier has no promotion curve.

- [ ] **Step 6: Verify and commit**

```bash
python -m pytest -q tests/v1/kv_offload/test_cost_model.py
python -m compileall -q vllm/v1/kv_offload/cost_model.py
git add vllm/v1/kv_offload/cost_model.py tests/v1/kv_offload/test_cost_model.py
git commit -m "feat: add KV offload shadow cost model"
```

---

### Task 2: Base and Tiering Spec Wiring

**Files:**

- Modify: `vllm/v1/kv_offload/base.py`
- Modify: `vllm/v1/kv_offload/tiering/spec.py`
- Create: `tests/v1/kv_offload/tiering/test_shadow_cost_spec.py`

- [ ] **Step 1: Write RED compatibility/tier-key tests**

```python
import pytest

from vllm.v1.kv_offload.base import OffloadingManager, OffloadingSpec, ReqContext
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


def test_base_shadow_interfaces_are_noop():
    assert OffloadingManager.get_load_provenance(
        object(), (), ReqContext("r"), 64
    ) is None
    assert OffloadingSpec.get_cost_model(object()) is None


def test_unique_secondary_types_default_to_type_key():
    assert TieringOffloadingSpec._resolve_cost_model_tier_keys(
        [{"type": "fs"}, {"type": "network"}], enabled=True
    ) == ("fs", "network")


def test_duplicate_types_require_explicit_keys():
    with pytest.raises(ValueError, match="cost_model_tier_key"):
        TieringOffloadingSpec._resolve_cost_model_tier_keys(
            [{"type": "fs"}, {"type": "fs"}], enabled=True
        )


def test_explicit_keys_disambiguate_duplicate_types():
    assert TieringOffloadingSpec._resolve_cost_model_tier_keys(
        [
            {"type": "fs", "cost_model_tier_key": "local_ssd"},
            {"type": "fs", "cost_model_tier_key": "slow_disk"},
        ],
        enabled=True,
    ) == ("local_ssd", "slow_disk")
```

Also test empty/non-string/duplicate explicit keys while enabled. Disabled mode returns type-derived keys and introduces no new validation failure.

- [ ] **Step 2: Run RED**

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_spec.py
```

- [ ] **Step 3: Add no-op base interfaces**

Under `TYPE_CHECKING`, import the two cost-model types. Add concrete, non-abstract methods:

```python
def get_load_provenance(
    self,
    keys: Collection[OffloadKey],
    req_context: ReqContext,
    external_tokens: int,
) -> "LoadProvenance | None":
    return None
```

and:

```python
def get_cost_model(self) -> "OffloadCostModel | None":
    return None
```

- [ ] **Step 4: Parse/share one model in `TieringOffloadingSpec`**

After validating `secondary_tier_configs`:

```python
self._cost_model = OffloadCostModel.from_extra_config(self.extra_config)
self._cost_model_tier_keys = self._resolve_cost_model_tier_keys(
    self.secondary_tier_configs,
    enabled=self._cost_model is not None,
)
```

`_resolve_cost_model_tier_keys()` uses explicit `cost_model_tier_key` when present, otherwise `type`; enabled keys must be non-empty strings and unique.

Override `get_cost_model()` to return `self._cost_model`. Construct manager with:

```python
tiering_manager = TieringOffloadingManager(
    primary_tier=primary_tier,
    secondary_tiers=secondary_tiers,
    cost_model=self._cost_model,
    secondary_tier_keys=self._cost_model_tier_keys,
    tokens_per_chunk_by_group=tuple(
        tokens_per_block * self.blocks_per_chunk
        for tokens_per_block in self.tokens_per_block
    ),
)
```

The scheduler and manager must share this exact model instance.

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_spec.py
python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
python -m compileall -q vllm/v1/kv_offload/base.py vllm/v1/kv_offload/tiering/spec.py
git add vllm/v1/kv_offload/base.py vllm/v1/kv_offload/tiering/spec.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_spec.py
git commit -m "feat: wire shadow cost model into tiering"
```

---

### Task 3: Tiering Provenance and Promotion Observation

**Files:**

- Modify: `vllm/v1/kv_offload/tiering/manager.py`
- Create: `tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py`

- [ ] **Step 1: Write RED manager tests**

Helper:

```python
from unittest.mock import MagicMock

from vllm.v1.kv_offload.base import LookupResult, RequestOffloadingContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.tiering.base import SecondaryTierManager


def make_manager(model):
    primary = MagicMock(spec=CPUOffloadingManager)
    secondary = MagicMock(spec=SecondaryTierManager)
    secondary.tier_type = "fs"
    secondary.get_finished_jobs.return_value = []
    secondary.on_new_request.return_value = RequestOffloadingContext()
    manager = TieringOffloadingManager(
        primary,
        [secondary],
        cost_model=model,
        secondary_tier_keys=("filesystem",),
        tokens_per_chunk_by_group=(64,),
    )
    return manager, primary, secondary
```

Direct CPU assertion:

```python
primary.lookup.return_value = LookupResult.HIT
manager.on_new_request(ctx)
assert manager.lookup(key, ctx) is LookupResult.HIT
first = manager.get_load_provenance([key], ctx, 64)
second = manager.get_load_provenance([key], ctx, 64)
assert first == second
assert first is not None
assert first.source == "cpu_primary"
assert first.secondary_promoted_tokens == 0
```

Secondary persistence assertion:

```python
primary.lookup.side_effect = [LookupResult.MISS, LookupResult.HIT]
secondary.lookup.return_value = LookupResult.HIT
monkeypatch.setattr(manager, "_initiate_promotion", lambda tier, block, context: True)
manager.on_new_request(ctx)
assert manager.lookup(key, ctx) is LookupResult.RETRY
assert manager.lookup(key, ctx) is LookupResult.HIT
provenance = manager.get_load_provenance([key], ctx, 64)
assert provenance is not None
assert provenance.source == "secondary:filesystem"
assert provenance.secondary_promoted_tokens == 64
```

Add failed promotion, CPU+secondary mixed, two-secondary mixed, unknown key, request finish cleanup, and active-request reset cleanup.

- [ ] **Step 2: Run RED**

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
```

- [ ] **Step 3: Add cost-only request state**

Nullable fields on `RequestState`:

```python
key_sources: dict[OffloadKey, str] | None = None
promotion_started_at: dict[str, float] | None = None
promotion_keys: dict[str, set[OffloadKey]] | None = None
promotion_pending_jobs: dict[str, int] | None = None
promotion_elapsed_seconds: dict[str, float] | None = None
```

Initialize them only when cost model is enabled.

Constructor adds backward-compatible keyword-only args:

```python
cost_model: OffloadCostModel | None = None
secondary_tier_keys: tuple[str, ...] | None = None
tokens_per_chunk_by_group: tuple[int, ...] = ()
```

Enabled mode requires one tier key per secondary tier and stores object->stable-key mapping.

- [ ] **Step 4: Record source without changing lookup results**

On primary HIT use `setdefault(key, "cpu_primary")`. On successful secondary promotion set `secondary:<stable-key>`, first promotion start time, and promoted key. Failed promotion stores no secondary source. Return branches remain exactly as before.

`get_load_provenance()` is read-only: empty/unknown keys return `None`; all CPU -> CPU/high/0 promoted tokens; one secondary source -> secondary/high/external_tokens promoted; multiple sources -> mixed/low/`None` promoted tokens.

`reset_cache()` clears cost-only dictionaries for retained active requests; existing request lifecycle remains unchanged.

- [ ] **Step 5: Write RED promotion-wave observation test**

Use actual `_initiate_promotion()` with:

```python
primary.prepare_write.return_value = PrepareStoreOutput(
    keys_to_store=[key],
    store_spec=CPULoadStoreSpec([3]),
    evicted_keys=[],
)
```

Spy on `model.observe_secondary_promotion`. Monkeypatch `time.monotonic` so first secondary lookup starts at `100.000`, sync accounting uses `100.001`, and completion uses `100.050`. After lookup, call `_flush_pending_promotions()`, return `JobResult(job_id=0, success=True)` from the secondary, then call `_process_finished_jobs()`.

Assert:

```python
spy.assert_called_once()
tier_key, tokens, observed_ms = spy.call_args.args
assert tier_key == "filesystem"
assert tokens == 64
assert observed_ms == pytest.approx(50.0)
```

For `success=False`, assert no EWMA call and no remaining provenance for the failed key.

- [ ] **Step 6: Implement promotion-wave timing**

Track promotion `job_id -> tier_key`. Increment request/tier pending count on `_flush_pending_promotions()`. On completion, decrement it. Failed completion removes failed job keys from `key_sources`. When the wave count reaches zero, always clear its timer/key/pending bookkeeping; only successful waves update EWMA.

Token span avoids duplicate KV-group accounting:

```python
def _token_span_for_keys(self, keys: Collection[OffloadKey]) -> int:
    per_group: dict[int, int] = {}
    for key in set(keys):
        group_idx = get_offload_group_idx(key)
        per_group[group_idx] = (
            per_group.get(group_idx, 0)
            + self._tokens_per_chunk_by_group[group_idx]
        )
    return max(per_group.values(), default=0)
```

For a successful wave, save elapsed seconds for provenance and call:

```python
try:
    observation = self._cost_model.observe_secondary_promotion(
        tier_key, token_span, elapsed_seconds * 1000.0
    )
except Exception:
    logger.exception("KV offload shadow promotion observation failed")
    observation = None
```

This exception must never alter `complete_write()` or transfer cleanup.

- [ ] **Step 7: Verify and commit**

```bash
python -m pytest -q tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
python -m compileall -q vllm/v1/kv_offload/tiering/manager.py
git add vllm/v1/kv_offload/tiering/manager.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
git commit -m "feat: track KV restore provenance"
```

---

### Task 4: Scheduler Shadow Hook and Metrics

**Files:**

- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py`
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
- Modify: `vllm/v1/kv_offload/tiering/manager.py`
- Modify: `tests/v1/kv_connector/unit/offloading_connector/utils.py`
- Modify: `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py`

- [ ] **Step 1: Add RED metric tests**

Exact names/metadata:

```text
vllm:kv_offload_cost_shadow_decisions                counter labels source,preferred,confidence
vllm:kv_offload_cost_predicted_restore_seconds       histogram label source
vllm:kv_offload_cost_predicted_recompute_seconds     histogram label source
vllm:kv_offload_cost_runtime_scale                   gauge labels source,token_bucket
vllm:kv_offload_cost_observations                    counter label source
```

Prediction buckets: `0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2.5,5,10` seconds.

- [ ] **Step 2: Register through `get_connector_metric_definitions()`**

Add five `_ConnectorMetricName` constants and use existing metadata classes. Do not create direct Prometheus globals.

- [ ] **Step 3: Enable model in `MockOffloadingSpec`**

```python
self.cost_model = OffloadCostModel.from_extra_config(self.extra_config)
```

and:

```python
def get_cost_model(self) -> OffloadCostModel | None:
    return self.cost_model
```

- [ ] **Step 4: Write RED scheduler invariance tests**

For a 12-token chunk (`block_size=4`, `blocks_per_chunk=3`), use the existing warmed-hit shape that validates `expected_loaded=(0, 1, 2)`. Shadow CPU config has recompute `100 ms`, CPU restore `20 ms`; secondary config has recompute `100 ms`, filesystem restore `200 ms`.

Wrap scheduler matching to capture results:

```python
matched_results: list[tuple[int | None, bool]] = []
original = runner.connector_scheduler.get_num_new_matched_tokens


def recording_get_num_new_matched_tokens(request, num_computed_tokens):
    result = original(request, num_computed_tokens)
    matched_results.append(result)
    return result

runner.connector_scheduler.get_num_new_matched_tokens = recording_get_num_new_matched_tokens
```

CPU provenance:

```python
runner.manager.get_load_provenance.return_value = LoadProvenance(
    "cpu_primary", 12, 0, ("cpu_primary",), "high"
)
runner.connector_scheduler._maximal_prefix_lookup = lambda keys, ctx: 1
runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_loaded=(0, 1, 2))
assert (12, True) in matched_results
key = (
    f"{_ConnectorMetricName.COST_SHADOW_DECISIONS}:"
    "('cpu_primary', 'restore', 'high')"
)
assert _reduce_kv_connector_stats(runner)[key] == 1
```

Add secondary provenance expecting `recompute` while the same `expected_loaded` proves actual restore. Add default-off test asserting `manager.get_load_provenance.assert_not_called()`.

Add fail-open test: make `manager.get_load_provenance.side_effect = RuntimeError("shadow failure")`; the warmed restore must still load `(0,1,2)` and matched result stays `(12, True)`.

- [ ] **Step 5: Implement matched-key helper and fail-open shadow hook**

Scheduler stores:

```python
self._cost_model = spec.get_cost_model()
```

Helper:

```python
def _get_matched_external_keys(
    self, req_status: RequestOffloadState, num_hit_tokens: int
) -> tuple[OffloadKey, ...]:
    num_cached_tokens = req_status.num_locally_computed_tokens + num_hit_tokens
    keys: list[OffloadKey] = []
    for group_config, group_state in zip(
        self.config.kv_group_configs, req_status.group_states
    ):
        start = req_status.num_locally_computed_tokens // group_config.tokens_per_chunk
        end = cdiv(num_cached_tokens, group_config.tokens_per_chunk)
        keys.extend(group_state.offload_keys[start:end])
    return tuple(keys)
```

After `update_num_hit_chunks()` and before `_touch()`:

```python
if self._cost_model is not None and num_hit_tokens is not None and num_hit_tokens > 0:
    try:
        matched_keys = self._get_matched_external_keys(req_status, num_hit_tokens)
        provenance = self.manager.get_load_provenance(
            matched_keys, req_status.req_context, num_hit_tokens
        )
        if provenance is not None:
            decision = self._cost_model.shadow_decide(provenance)
            if decision is not None:
                self._record_shadow_decision(
                    request.request_id, provenance, decision
                )
    except Exception:
        logger.exception(
            "KV offload shadow decision failed for request %s",
            request.request_id,
        )
```

Then preserve exactly:

```python
self._touch(req_status)
return num_hit_tokens, bool(num_hit_tokens)
```

- [ ] **Step 6: Record decision and EWMA metrics**

Scheduler decision stats: counter labels `(source, preferred, confidence)`; prediction histograms convert ms to seconds and label only `(source,)`. Debug log contains request ID, external tokens, source, restore seed/estimate, recompute estimate, runtime scale, preferred, confidence, `mode=shadow`, `actual_path=restore`.

Manager observation stats, only when Task 3 returns a non-`None` observation:

```python
source = f"secondary:{observation.tier_key}"
self._stats.increase_counter(
    _ConnectorMetricName.COST_OBSERVATIONS,
    labelvalues=(source,),
)
self._stats.set_gauge(
    _ConnectorMetricName.COST_RUNTIME_SCALE,
    observation.runtime_scale,
    labelvalues=(source, str(observation.token_bucket)),
)
```

Wrap only cost-observation metric recording in the same fail-open block; do not wrap or suppress transfer completion errors.

- [ ] **Step 7: Verify and commit**

```bash
python -m pytest -q tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
python -m pytest -q tests/v1/kv_offload/test_cost_model.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_spec.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_provenance.py
python -m compileall -q \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py \
  vllm/v1/kv_offload/tiering/manager.py
git add \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py \
  vllm/v1/kv_offload/tiering/manager.py \
  tests/v1/kv_connector/unit/offloading_connector/utils.py \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
git commit -m "feat: record KV offload shadow decisions"
```

---

### Task 5: Source Verification Gate

**Files:** No new files. Any failure returns to the owning task with a RED regression test before a fix.

- [ ] **Step 1: Focused test suite**

```bash
python -m pytest -q \
  tests/v1/kv_offload/test_cost_model.py \
  tests/v1/kv_offload/tiering \
  tests/v1/kv_connector/unit/offloading_connector
```

- [ ] **Step 2: Compile**

```bash
python -m compileall -q \
  vllm/v1/kv_offload \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading
```

- [ ] **Step 3: Available lint hooks**

```bash
command -v pre-commit || true
command -v ruff || true
```

If `pre-commit` exists, run it over the touched Python files. If only Ruff exists, run `ruff check` over those files. If neither exists, record that fact rather than claiming lint passed.

- [ ] **Step 4: Shadow-safety diff review**

```bash
git diff main...HEAD -- \
  vllm/v1/kv_offload/base.py \
  vllm/v1/kv_offload/tiering/manager.py \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
git diff --check main...HEAD
git status --short
```

Gate: `LookupResult` unchanged; tiering return branches unchanged; scheduler still ends with original return; decision cannot change skip-reading, promotion, allocation, prepare-load, or jobs; enforce rejected; benchmark values absent from production defaults.

---

### Task 6: Hardware Shadow Validation with PR #5

**Files:** Validation-only edits stay on a disposable local branch and never enter the runtime PR.

- [ ] **Step 1: Build disposable integration checkout**

```bash
cd /code/vllm
git fetch origin main feature/kv-offload-shadow-cost-model feature/cache-eviction-restore-benchmark
git switch -C validation/kv-offload-shadow-cost-model \
  origin/feature/kv-offload-shadow-cost-model
git merge --no-edit origin/feature/cache-eviction-restore-benchmark
```

- [ ] **Step 2: Inject validation profile without changing benchmark schema**

In disposable `benchmarks/cache/scenarios.py`, immediately after `_offloading_config()` creates `extra`, add:

```python
raw_cost_model = os.environ.get("VLLM_CACHE_COST_MODEL_JSON")
if raw_cost_model:
    extra["cache_cost_model"] = json.loads(raw_cost_model)
```

Add to the tiered-fs secondary dict:

```python
"cost_model_tier_key": "filesystem",
```

Export:

```bash
export VLLM_CACHE_COST_MODEL_JSON='{"mode":"shadow","ewma_alpha":0.2,"sample_scale_min":0.25,"sample_scale_max":4.0,"profile":{"recompute_ms":{"256":26.414,"512":44.961,"1024":81.705,"2048":152.461,"4096":308.424},"tiers":{"cpu_primary":{"restore_ms":{"1024":24.49}},"filesystem":{"restore_ms":{"256":31.119,"512":56.979,"1024":108.132,"2048":244.266,"4096":651.127},"promotion_ms":{"256":13.916,"512":35.23,"1024":81.458,"2048":171.505,"4096":498.874}}}}}'
```

- [ ] **Step 3: CPU-primary 1024 anchor**

Keep tiered-fs mode but set `cache.cpu_bytes_to_use: 4294967296`. Run only tiered-fs eviction-restore p1024. Expected shadow `restore`, actual `restore`, source `cpu_primary`, no secondary cost observation. Verify 8 CPU->GPU loads and external tokens remain consistent with prior CPU control.

- [ ] **Step 4: Filesystem 1024 anchor**

Restore `cache.cpu_bytes_to_use: 2147483648`. Run only p1024. Expected shadow `recompute`, actual `restore`, source `secondary:filesystem`; observation counter and runtime-scale gauge appear; external tokens and CPU->GPU bytes match baseline behavior.

- [ ] **Step 5: Five-point filesystem sweep**

Run tiered-fs eviction-restore at 256/512/1024/2048/4096. Expected shadow preference: recompute at all five; actual path remains restore.

Record:

```text
prompt_tokens
actual_p95_ttft_ms
external_tokens
cpu_to_gpu_bytes
shadow_source
shadow_preferred
shadow_confidence
runtime_scale_bucket
runtime_scale
actual_path
```

Compare against prior FS P95 baselines `31.119, 56.979, 108.132, 244.266, 651.127 ms`. Repeat only an affected single case if instrumentation shows a consistent regression beyond ordinary run-to-run noise; do not invent a one-run percentage threshold.

- [ ] **Step 6: Return to clean runtime branch**

```bash
git switch feature/kv-offload-shadow-cost-model
git status --short
git diff --name-only main...HEAD
```

No benchmark validation edit may appear in the runtime diff.

---

## Final Verification Before Completion

Invoke `superpowers:verification-before-completion`, then collect fresh output from:

```bash
python -m pytest -q \
  tests/v1/kv_offload/test_cost_model.py \
  tests/v1/kv_offload/tiering \
  tests/v1/kv_connector/unit/offloading_connector
python -m compileall -q \
  vllm/v1/kv_offload \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading
git diff --check main...HEAD
git status --short
```

Completion evidence must also include Task 6's CPU anchor, filesystem anchor, and all five filesystem shadow decisions. Report unavailable or unrelated-failing lint tooling exactly; never describe an unrun/red check as green.
