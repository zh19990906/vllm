from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
RequestRate = Literal["inf"] | PositiveFloat

OWNERSHIP_MARKER = ".vllm-cache-benchmark-owned"
_SECRET_KEY_PARTS = (
    "TOKEN",
    "PASSWORD",
    "SECRET",
    "CREDENTIAL",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    id: str
    served_name: str
    dtype: str = "auto"
    max_model_len: PositiveInt
    trust_remote_code: bool = False

    @field_validator("id", "served_name", "dtype")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class ParallelismConfig(StrictModel):
    tensor_parallel_size: PositiveInt = 1
    pipeline_parallel_size: PositiveInt = 1


class ServerConfig(StrictModel):
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8100
    startup_timeout_seconds: PositiveInt = 900
    shutdown_timeout_seconds: PositiveInt = 60
    extra_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("host")
    @classmethod
    def require_non_empty_host(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("host must not be empty")
        return value


class FilesystemCacheConfig(StrictModel):
    enabled: bool = True
    root_dir: Path
    read_threads: PositiveInt = 32
    write_threads: PositiveInt = 16


class CacheConfig(StrictModel):
    gpu_memory_utilization: Annotated[float, Field(gt=0.0, le=1.0)] = 0.9
    cpu_bytes_to_use: PositiveInt
    offload_block_size: PositiveInt = 64
    eviction_policy: Literal["lru", "arc"] = "lru"
    filesystem: FilesystemCacheConfig


class WorkloadConfig(StrictModel):
    seed: int = 1
    tokenizer: str
    prompt_tokens: list[PositiveInt]
    output_tokens: PositiveInt
    concurrency: list[PositiveInt]
    request_rate: list[RequestRate]
    requests_per_case: PositiveInt
    shared_prefix_ratios: list[Ratio]
    warmup_requests: Annotated[int, Field(ge=0)] = 2
    token_length_tolerance: Annotated[int, Field(ge=0, le=8)] = 2

    @field_validator("tokenizer")
    @classmethod
    def require_non_empty_tokenizer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tokenizer must not be empty")
        return value

    @field_validator("request_rate", mode="before")
    @classmethod
    def normalize_request_rates(cls, values: object) -> object:
        if not isinstance(values, list):
            return values
        normalized: list[object] = []
        for value in values:
            if isinstance(value, str) and value.lower() == "inf":
                normalized.append("inf")
            elif isinstance(value, (int, float)) and value == float("inf"):
                normalized.append("inf")
            else:
                normalized.append(value)
        return normalized


class ResultsConfig(StrictModel):
    root_dir: Path
    keep_server_logs: bool = True
    fail_fast: bool = False


class SuiteConfig(StrictModel):
    schema_version: Literal[1]
    model: ModelConfig
    parallelism: ParallelismConfig
    server: ServerConfig
    cache: CacheConfig
    workload: WorkloadConfig
    results: ResultsConfig

    @model_validator(mode="after")
    def validate_non_empty_axes(self) -> "SuiteConfig":
        for name in (
            "prompt_tokens",
            "concurrency",
            "request_rate",
            "shared_prefix_ratios",
        ):
            if not getattr(self.workload, name):
                raise ValueError(f"{name} must not be empty")
        return self


def _resolve_config_path(value: object, base_dir: Path, field_name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field_name} must be a filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_suite_config(path: Path) -> SuiteConfig:
    """Load a strict suite configuration and resolve its relative paths."""
    config_path = path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("suite configuration must be a YAML mapping")

    data: dict[str, Any] = dict(raw)
    cache = dict(data.get("cache") or {})
    filesystem = dict(cache.get("filesystem") or {})
    results = dict(data.get("results") or {})

    filesystem["root_dir"] = _resolve_config_path(
        filesystem.get("root_dir"), config_path.parent, "cache.filesystem.root_dir"
    )
    results["root_dir"] = _resolve_config_path(
        results.get("root_dir"), config_path.parent, "results.root_dir"
    )
    cache["filesystem"] = filesystem
    data["cache"] = cache
    data["results"] = results

    return SuiteConfig.model_validate(data)


def sanitize_environment(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy safe to write to benchmark artifacts."""
    sanitized: dict[str, str] = {}
    for key, value in env.items():
        upper_key = key.upper()
        sanitized[key] = (
            "<redacted>"
            if any(secret_part in upper_key for secret_part in _SECRET_KEY_PARTS)
            else value
        )
    return sanitized


def _resolved_owned_child(path: Path, root: Path) -> tuple[Path, Path]:
    resolved_root = root.expanduser().resolve()
    resolved_path = path.expanduser().resolve()
    if resolved_path == resolved_root or not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"path must be below configured root: {resolved_root}")
    return resolved_path, resolved_root


def assert_owned_child(path: Path, root: Path) -> None:
    """Validate that an existing child directory was created by this suite."""
    resolved_path, _ = _resolved_owned_child(path, root)
    if not resolved_path.is_dir():
        raise ValueError(f"owned path is not a directory: {resolved_path}")
    marker = resolved_path / OWNERSHIP_MARKER
    if not marker.is_file():
        raise ValueError(f"ownership marker is missing: {marker}")


def create_owned_directory(path: Path, root: Path) -> Path:
    """Create a suite-owned child without claiming existing user directories."""
    resolved_path, resolved_root = _resolved_owned_child(path, root)
    resolved_root.mkdir(parents=True, exist_ok=True)
    marker = resolved_path / OWNERSHIP_MARKER
    if resolved_path.exists():
        assert_owned_child(resolved_path, resolved_root)
        return resolved_path
    resolved_path.mkdir(parents=True, exist_ok=False)
    marker.touch(exist_ok=False)
    assert_owned_child(resolved_path, resolved_root)
    return resolved_path
