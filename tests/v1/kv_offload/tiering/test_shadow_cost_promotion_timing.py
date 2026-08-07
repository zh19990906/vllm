# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

_HELPER_PATH = Path(__file__).with_name("test_shadow_cost_provenance.py")
_SPEC = importlib.util.spec_from_file_location(
    "shadow_cost_provenance_test_helpers", _HELPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HELPERS
_SPEC.loader.exec_module(_HELPERS)

LookupResult = _HELPERS.LookupResult
ReqContext = _HELPERS.ReqContext
JobResult = _HELPERS.JobResult
_MANAGER_MODULE = _HELPERS._MANAGER_MODULE
make_manager = _HELPERS.make_manager
make_offload_key = _HELPERS.make_offload_key
prepared_write = _HELPERS.prepared_write


def test_promotion_observation_starts_at_first_tier_retry() -> None:
    manager, primary, secondary, model = make_manager()
    ctx = ReqContext("r")
    key = make_offload_key(b"deferred", 0)
    manager.on_new_request(ctx)
    primary.lookup_results = [LookupResult.MISS, LookupResult.MISS]
    primary.prepare_write_result = prepared_write(key)
    secondary.lookup_results = [LookupResult.RETRY, LookupResult.HIT]

    now = [1.0]
    original_monotonic = _MANAGER_MODULE.time.monotonic
    _MANAGER_MODULE.time.monotonic = lambda: now[0]
    try:
        assert manager.lookup(key, ctx) is LookupResult.RETRY

        now[0] = 1.1
        assert manager.lookup(key, ctx) is LookupResult.RETRY
        manager._flush_pending_promotions()
        job = secondary.submitted_loads[0]

        now[0] = 1.2
        secondary.finished_jobs = [JobResult(job.job_id, True)]
        manager._process_finished_jobs()
    finally:
        _MANAGER_MODULE.time.monotonic = original_monotonic

    provenance = manager.get_load_provenance([key], ctx, 64)
    assert provenance is not None
    assert provenance.lookup_async_seconds is not None
    assert abs(provenance.lookup_async_seconds - 0.2) < 1e-9

    # Seed is 100 ms. A 200 ms observation is a 2.0 sample scale and,
    # with alpha=0.2 from an initial 1.0, updates the runtime scale to 1.2.
    decision = model.shadow_decide(provenance)
    assert decision is not None
    assert abs(decision.runtime_scale - 1.2) < 1e-9
