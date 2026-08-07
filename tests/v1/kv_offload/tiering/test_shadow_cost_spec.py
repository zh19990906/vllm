# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


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


def _install_import_stubs() -> tuple[ModuleType, ModuleType]:
    for name in ("vllm", "vllm.v1", "vllm.v1.kv_offload", "vllm.v1.kv_offload.tiering"):
        if name not in sys.modules:
            _package(name)

    numpy = ModuleType("numpy")
    numpy.ndarray = object
    numpy.int64 = int
    numpy.array = lambda values, dtype=None: list(values)
    sys.modules["numpy"] = numpy

    torch = ModuleType("torch")
    torch.Tensor = object
    torch.accelerator = SimpleNamespace(current_device_index=lambda: 0)
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

    logger_module.init_logger = lambda name: _Logger()
    sys.modules["vllm.logger"] = logger_module

    config_module = ModuleType("vllm.v1.kv_offload.config")
    config_module.OffloadingConfig = object
    sys.modules["vllm.v1.kv_offload.config"] = config_module

    cost_model = _load(
        "vllm.v1.kv_offload.cost_model",
        _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "cost_model.py",
    )
    base = _load(
        "vllm.v1.kv_offload.base",
        _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "base.py",
    )

    try:
        import typing_extensions  # noqa: F401
    except ModuleNotFoundError:
        typing_extensions = ModuleType("typing_extensions")
        typing_extensions.override = lambda function: function
        sys.modules["typing_extensions"] = typing_extensions

    gpu_worker = ModuleType("vllm.v1.kv_offload.cpu.gpu_worker")
    gpu_worker.CPUOffloadingWorker = type("CPUOffloadingWorker", (), {})
    sys.modules["vllm.v1.kv_offload.cpu.gpu_worker"] = gpu_worker

    shared_region = ModuleType("vllm.v1.kv_offload.cpu.shared_offload_region")

    class _SharedOffloadRegion:
        BLOCK_SIZE_ALIGNMENT = 1

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    shared_region.SharedOffloadRegion = _SharedOffloadRegion
    sys.modules["vllm.v1.kv_offload.cpu.shared_offload_region"] = shared_region

    cpu_spec = ModuleType("vllm.v1.kv_offload.cpu.spec")

    class _CPUOffloadingSpec:
        @classmethod
        def build_metric_definitions(cls, extra_config):
            return {}

        def __init__(self, config) -> None:
            self.config = config
            self.extra_config = config.extra_config
            self.kv_events_config = SimpleNamespace(
                self_describing_kv_events=False,
                enable_kv_cache_events=config.enable_kv_cache_events,
            )
            self.tokens_per_block = tuple(
                group.tokens_per_block for group in config.groups
            )
            self.blocks_per_chunk = config.cache.blocks_per_chunk
            self.num_blocks = 1
            self.kv_bytes_per_chunk = 1
            self.cpu_page_size_per_worker = 1
            self.eviction_policy = "lru"

    cpu_spec.CPUOffloadingSpec = _CPUOffloadingSpec
    sys.modules["vllm.v1.kv_offload.cpu.spec"] = cpu_spec

    tiering_base = ModuleType("vllm.v1.kv_offload.tiering.base")
    tiering_base.TieringOffloadingMetrics = type(
        "TieringOffloadingMetrics",
        (),
        {
            "LOOKUP_SYNC_DELAY": "lookup_sync",
            "LOOKUP_ASYNC_DELAY": "lookup_async",
        },
    )
    sys.modules["vllm.v1.kv_offload.tiering.base"] = tiering_base

    factory = ModuleType("vllm.v1.kv_offload.tiering.factory")

    class _TierClass:
        @classmethod
        def build_metric_definitions(cls, tier_config):
            return {}

    class _SecondaryTierFactory:
        @staticmethod
        def get_tier_class(tier_config):
            return _TierClass

        @staticmethod
        def create_secondary_tier(tier_config, primary_kv_view, offloading_spec):
            return SimpleNamespace(tier_type=tier_config["type"])

    factory.SecondaryTierFactory = _SecondaryTierFactory
    sys.modules["vllm.v1.kv_offload.tiering.factory"] = factory

    manager_module = ModuleType("vllm.v1.kv_offload.tiering.manager")

    class _PrimaryManager:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def get_kv_memoryview(self):
            return memoryview(bytearray(1))

    class _TieringManager:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    manager_module.CPUPrimaryTierOffloadingManager = _PrimaryManager
    manager_module.TieringOffloadingManager = _TieringManager
    sys.modules["vllm.v1.kv_offload.tiering.manager"] = manager_module

    tiering_spec = _load(
        "vllm.v1.kv_offload.tiering.spec",
        _REPO_ROOT / "vllm" / "v1" / "kv_offload" / "tiering" / "spec.py",
    )
    return base, tiering_spec


_BASE, _TIERING_SPEC = _install_import_stubs()
OffloadingManager = _BASE.OffloadingManager
OffloadingSpec = _BASE.OffloadingSpec
ReqContext = _BASE.ReqContext
TieringOffloadingSpec = _TIERING_SPEC.TieringOffloadingSpec


def test_base_shadow_interfaces_are_noop() -> None:
    assert OffloadingManager.get_load_provenance(
        object(), (), ReqContext("r"), 64
    ) is None
    assert OffloadingSpec.get_cost_model(object()) is None


def test_unique_secondary_types_default_to_type_key() -> None:
    assert TieringOffloadingSpec._resolve_cost_model_tier_keys(
        [{"type": "fs"}, {"type": "network"}], enabled=True
    ) == ("fs", "network")


def test_duplicate_types_require_explicit_keys() -> None:
    with pytest.raises(ValueError, match="cost_model_tier_key"):
        TieringOffloadingSpec._resolve_cost_model_tier_keys(
            [{"type": "fs"}, {"type": "fs"}], enabled=True
        )


def test_explicit_keys_disambiguate_duplicate_types() -> None:
    assert TieringOffloadingSpec._resolve_cost_model_tier_keys(
        [
            {"type": "fs", "cost_model_tier_key": "local_ssd"},
            {"type": "fs", "cost_model_tier_key": "slow_disk"},
        ],
        enabled=True,
    ) == ("local_ssd", "slow_disk")


@pytest.mark.parametrize(
    "configs",
    [
        [{"type": "fs", "cost_model_tier_key": ""}],
        [{"type": "fs", "cost_model_tier_key": 1}],
        [
            {"type": "fs", "cost_model_tier_key": "disk"},
            {"type": "network", "cost_model_tier_key": "disk"},
        ],
    ],
)
def test_enabled_cost_model_rejects_invalid_tier_keys(configs: list[dict]) -> None:
    with pytest.raises(ValueError, match="cost_model_tier_key"):
        TieringOffloadingSpec._resolve_cost_model_tier_keys(configs, enabled=True)


def test_disabled_cost_model_does_not_add_tier_key_validation() -> None:
    configs = [
        {"type": "fs", "cost_model_tier_key": ""},
        {"type": "fs", "cost_model_tier_key": ""},
    ]
    assert TieringOffloadingSpec._resolve_cost_model_tier_keys(
        configs, enabled=False
    ) == ("fs", "fs")


def _config_with_shadow_model():
    return SimpleNamespace(
        extra_config={
            "cpu_bytes_to_use": 1,
            "secondary_tiers": [
                {
                    "type": "fs",
                    "cost_model_tier_key": "filesystem",
                }
            ],
            "cache_cost_model": {
                "mode": "shadow",
                "profile": {
                    "recompute_ms": {1024: 80.0},
                    "tiers": {
                        "filesystem": {
                            "restore_ms": {1024: 100.0},
                            "promotion_ms": {1024: 75.0},
                        }
                    },
                },
            },
        },
        enable_kv_cache_events=False,
        groups=(SimpleNamespace(tokens_per_block=64),),
        cache=SimpleNamespace(tokens_per_hash=64, blocks_per_chunk=1),
        engine_id="engine-test",
        parallel=SimpleNamespace(world_size=1),
        worker_kv_bytes_per_block=1,
    )


def test_tiering_spec_shares_one_cost_model_with_manager() -> None:
    spec = TieringOffloadingSpec(_config_with_shadow_model())

    model = spec.get_cost_model()
    assert model is not None
    assert spec._cost_model_tier_keys == ("filesystem",)

    manager = spec.get_manager()
    assert manager.kwargs["cost_model"] is model
    assert manager.kwargs["secondary_tier_keys"] == ("filesystem",)
    assert manager.kwargs["tokens_per_chunk_by_group"] == (64,)
