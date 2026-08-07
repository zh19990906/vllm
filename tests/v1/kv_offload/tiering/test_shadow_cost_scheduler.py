# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


_REPO_ROOT = Path(__file__).resolve().parents[4]


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_stubs() -> tuple[ModuleType, ModuleType, ModuleType]:
    packages = (
        "vllm",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.kv_offload",
    )
    for name in packages:
        if name not in sys.modules:
            _package(name)

    numpy = ModuleType("numpy")
    numpy.ndarray = list
    numpy.int64 = int
    numpy.array = lambda values, dtype=None: list(values)
    sys.modules["numpy"] = numpy

    torch = ModuleType("torch")
    torch.Tensor = object
    sys.modules["torch"] = torch

    logger_module = ModuleType("vllm.logger")

    class _Logger:
        def warning(self, *args, **kwargs) -> None:
            pass

        def info(self, *args, **kwargs) -> None:
            pass

        def error(self, *args, **kwargs) -> None:
            pass

        def debug(self, *args, **kwargs) -> None:
            pass

        def exception(self, *args, **kwargs) -> None:
            pass

    logger_module.init_logger = lambda name: _Logger()
    sys.modules["vllm.logger"] = logger_module

    config_module = ModuleType("vllm.config")
    config_module.VllmConfig = object
    sys.modules["vllm.config"] = config_module

    offload_config_module = ModuleType("vllm.v1.kv_offload.config")
    offload_config_module.OffloadingConfig = object
    sys.modules["vllm.v1.kv_offload.config"] = offload_config_module

    cost_model = _load(
        "vllm.v1.kv_offload.cost_model",
        _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "cost_model.py",
    )
    base = _load(
        "vllm.v1.kv_offload.base",
        _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "base.py",
    )

    connector_metrics = ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics"
    )

    @dataclass
    class _KVConnectorStats:
        data: dict = field(default_factory=dict)

        def is_empty(self) -> bool:
            return not self.data

    class _KVConnectorPromMetrics:
        def __init__(self, *args, **kwargs) -> None:
            self._labelnames = []
            self.per_engine_labelvalues = {}

    connector_metrics.KVConnectorStats = _KVConnectorStats
    connector_metrics.KVConnectorPromMetrics = _KVConnectorPromMetrics
    connector_metrics.PromMetric = object
    connector_metrics.PromMetricT = object
    sys.modules[connector_metrics.__name__] = connector_metrics

    offload_factory = ModuleType("vllm.v1.kv_offload.factory")
    offload_factory.OffloadingSpecFactory = type(
        "OffloadingSpecFactory", (), {"get_spec_cls": staticmethod(lambda config: object)}
    )
    sys.modules[offload_factory.__name__] = offload_factory

    metrics = _load(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics",
        _REPO_ROOT
        / "vllm"
        / "distributed"
        / "kv_transfer"
        / "kv_connector"
        / "v1"
        / "offloading"
        / "metrics.py",
    )

    kv_events = ModuleType("vllm.distributed.kv_events")
    kv_events.KVCacheEvent = object
    sys.modules[kv_events.__name__] = kv_events

    connector_utils = ModuleType("vllm.distributed.kv_transfer.kv_connector.utils")
    connector_utils.yield_req_data = lambda output: ()
    sys.modules[connector_utils.__name__] = connector_utils

    connector_base = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    connector_base.KVConnectorMetadata = object
    sys.modules[connector_base.__name__] = connector_base

    common = ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common"
    )
    common.OffloadingConnectorMetadata = object
    common.OffloadingWorkerMetadata = object
    common.ReqId = str
    common.TransferJob = object
    sys.modules[common.__name__] = common

    events = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.offloading.events")
    events.OffloadingEventGroupSpec = object

    class _EventsTracker:
        def __init__(self, config) -> None:
            pass

    events.OffloadingEventsTracker = _EventsTracker
    events.get_offloading_event_group_spec = lambda group: object()
    sys.modules[events.__name__] = events

    math_utils = ModuleType("vllm.utils.math_utils")
    math_utils.cdiv = lambda a, b: (a + b - 1) // b
    math_utils.round_down = lambda value, multiple: value // multiple * multiple
    _package("vllm.utils")
    sys.modules[math_utils.__name__] = math_utils

    kv_cache_manager = ModuleType("vllm.v1.core.kv_cache_manager")
    kv_cache_manager.KVCacheBlocks = object
    sys.modules[kv_cache_manager.__name__] = kv_cache_manager

    sched_output = ModuleType("vllm.v1.core.sched.output")
    sched_output.SchedulerOutput = object
    sys.modules[sched_output.__name__] = sched_output

    cache_interface = ModuleType("vllm.v1.kv_cache_interface")
    for name in (
        "FullAttentionSpec",
        "KVCacheConfig",
        "KVCacheSpec",
        "MambaSpec",
        "SlidingWindowSpec",
    ):
        setattr(cache_interface, name, type(name, (), {}))
    sys.modules[cache_interface.__name__] = cache_interface

    outputs = ModuleType("vllm.v1.outputs")
    outputs.KVConnectorOutput = object
    sys.modules[outputs.__name__] = outputs

    request_module = ModuleType("vllm.v1.request")
    request_module.Request = object
    sys.modules[request_module.__name__] = request_module

    scheduler = _load(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler",
        _REPO_ROOT
        / "vllm"
        / "distributed"
        / "kv_transfer"
        / "kv_connector"
        / "v1"
        / "offloading"
        / "scheduler.py",
    )
    return cost_model, metrics, scheduler


_COST_MODEL, _METRICS, _SCHEDULER = _install_stubs()
OffloadCostModel = _COST_MODEL.OffloadCostModel
LoadProvenance = _COST_MODEL.LoadProvenance
OffloadingConnectorStats = _METRICS.OffloadingConnectorStats
_ConnectorMetricName = _METRICS._ConnectorMetricName
get_connector_metric_definitions = _METRICS.get_connector_metric_definitions
OffloadingConnectorScheduler = _SCHEDULER.OffloadingConnectorScheduler
ReqContext = sys.modules["vllm.v1.kv_offload.base"].ReqContext
make_offload_key = sys.modules["vllm.v1.kv_offload.base"].make_offload_key


PROFILE = {
    "cache_cost_model": {
        "mode": "shadow",
        "profile": {
            "recompute_ms": {64: 50.0},
            "tiers": {
                "filesystem": {
                    "restore_ms": {64: 80.0},
                    "promotion_ms": {64: 60.0},
                }
            },
        },
    }
}


class FakeManager:
    def __init__(self, provenance) -> None:
        self.provenance = provenance
        self.provenance_calls = 0

    def get_load_provenance(self, keys, req_context, external_tokens):
        self.provenance_calls += 1
        if isinstance(self.provenance, Exception):
            raise self.provenance
        return self.provenance


class ReqStatus:
    def __init__(self, keys) -> None:
        self.group_states = [SimpleNamespace(block_ids=[], offload_keys=list(keys))]
        self.transfer_jobs = set()
        self.num_locally_computed_tokens = 0
        self.deferred_lookup_start_time = None
        self.req_context = ReqContext("r")
        self.updated_hit_tokens = None

    def update_offload_keys(self) -> None:
        pass

    def update_num_hit_chunks(self, num_cached_tokens: int) -> None:
        self.updated_hit_tokens = num_cached_tokens


def _scheduler(model, manager):
    scheduler = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    scheduler._cost_model = model
    scheduler.manager = manager
    scheduler._connector_stats = OffloadingConnectorStats()
    scheduler.config = SimpleNamespace(
        kv_group_configs=(SimpleNamespace(tokens_per_chunk=64),)
    )
    scheduler._req_status = {}
    scheduler._lookup = lambda status: 64
    scheduler._touch = lambda status: None
    return scheduler


def _request():
    return SimpleNamespace(request_id="r", skip_reading_prefix_cache=False)


def _reduced(stats):
    return stats.reduce()


def test_shadow_metric_definitions_have_bounded_labels() -> None:
    definitions = get_connector_metric_definitions()

    assert definitions[_ConnectorMetricName.COST_SHADOW_DECISIONS].labelnames == (
        "source",
        "preferred",
        "confidence",
    )
    assert definitions[_ConnectorMetricName.COST_PREDICTED_RESTORE].labelnames == (
        "source",
    )
    assert definitions[_ConnectorMetricName.COST_PREDICTED_RECOMPUTE].labelnames == (
        "source",
    )
    assert definitions[_ConnectorMetricName.COST_RUNTIME_SCALE].labelnames == (
        "source",
        "token_bucket",
    )
    assert definitions[_ConnectorMetricName.COST_OBSERVATIONS].labelnames == (
        "source",
    )


def test_shadow_recompute_prediction_keeps_original_scheduler_return() -> None:
    model = OffloadCostModel.from_extra_config(PROFILE)
    assert model is not None
    provenance = LoadProvenance(
        source="secondary:filesystem",
        external_tokens=64,
        secondary_promoted_tokens=64,
        sources=("secondary:filesystem",),
        confidence="high",
    )
    manager = FakeManager(provenance)
    scheduler = _scheduler(model, manager)
    key = make_offload_key(b"fs", 0)
    status = ReqStatus([key])
    scheduler._req_status["r"] = status

    result = scheduler.get_num_new_matched_tokens(_request(), 0)

    assert result == (64, True)
    assert status.updated_hit_tokens == 64
    assert manager.provenance_calls == 1
    reduced = _reduced(scheduler._connector_stats)
    decision_key = (
        f'{_ConnectorMetricName.COST_SHADOW_DECISIONS}:'
        "('secondary:filesystem', 'recompute', 'high')"
    )
    assert reduced[decision_key] == 1


def test_shadow_off_never_queries_provenance() -> None:
    manager = FakeManager(AssertionError("provenance must stay disabled"))
    scheduler = _scheduler(None, manager)
    status = ReqStatus([make_offload_key(b"off", 0)])
    scheduler._req_status["r"] = status

    assert scheduler.get_num_new_matched_tokens(_request(), 0) == (64, True)
    assert manager.provenance_calls == 0


def test_shadow_runtime_error_fails_open() -> None:
    model = OffloadCostModel.from_extra_config(PROFILE)
    assert model is not None
    manager = FakeManager(RuntimeError("telemetry failure"))
    scheduler = _scheduler(model, manager)
    status = ReqStatus([make_offload_key(b"fail-open", 0)])
    scheduler._req_status["r"] = status

    assert scheduler.get_num_new_matched_tokens(_request(), 0) == (64, True)
    assert status.updated_hit_tokens == 64


def test_matched_external_keys_follow_final_token_boundary() -> None:
    scheduler = _scheduler(None, FakeManager(None))
    keys = [
        make_offload_key(b"0", 0),
        make_offload_key(b"1", 0),
        make_offload_key(b"2", 0),
    ]
    status = ReqStatus(keys)
    status.num_locally_computed_tokens = 64

    matched = scheduler._get_matched_external_keys(status, 128)

    assert matched == (keys[1], keys[2])
