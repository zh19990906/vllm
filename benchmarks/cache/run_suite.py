from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cache.config import (  # noqa: E402
    SuiteConfig,
    create_owned_directory,
    load_suite_config,
    sanitize_environment,
)
from benchmarks.cache.metrics import (  # noqa: E402
    ResourceSampler,
    collect_environment_evidence,
    fetch_prometheus_snapshot,
    normalize_native_result,
)
from benchmarks.cache.process import (  # noqa: E402
    CommandResult,
    ManagedProcess,
    ServerExitedError,
    ServerReadinessTimeout,
    run_command,
    start_server,
    stop_server,
    wait_for_server,
)
from benchmarks.cache.report import append_result, rebuild_reports  # noqa: E402
from benchmarks.cache.scenarios import (  # noqa: E402
    CacheMode,
    ExecutionCase,
    build_execution_cases,
    build_server_command,
    build_server_environment,
)
from benchmarks.cache.workload import (  # noqa: E402
    WorkloadArtifacts,
    build_benchmark_command,
    generate_workload,
)

BASE_COMMIT = "568afb3a13806beb53bb2e6bd518269357b237c0"
BENCHMARK_TIMEOUT_SECONDS = 6 * 60 * 60


class ConfigurationError(RuntimeError):
    pass


class BenchmarkExecutionError(RuntimeError):
    def __init__(self, message: str, result: CommandResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class NativeResultParseError(RuntimeError):
    pass


class MetricsCollectionError(RuntimeError):
    pass


class ShutdownError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sanitize_nested(value: Any) -> Any:
    if isinstance(value, Mapping):
        string_values = {
            str(key): str(item)
            for key, item in value.items()
            if isinstance(item, str)
        }
        sanitized_strings = sanitize_environment(string_values)
        return {
            str(key): (
                sanitized_strings[str(key)]
                if isinstance(item, str)
                else _sanitize_nested(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_nested(item) for item in value]
    return value


def _config_payload(config: SuiteConfig) -> dict[str, Any]:
    return _sanitize_nested(config.model_dump(mode="json"))


def _config_fingerprint(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _run_id(fingerprint: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{fingerprint[:8]}"


def _case_payload(case: ExecutionCase) -> dict[str, Any]:
    payload = asdict(case)
    payload["cache_mode"] = case.cache_mode.value
    payload["result_dir"] = str(case.result_dir)
    payload["filesystem_cache_dir"] = (
        str(case.filesystem_cache_dir) if case.filesystem_cache_dir else None
    )
    return payload


def _planned_population_count(case: ExecutionCase, config: SuiteConfig) -> int:
    if case.workload_kind == "cold-unique":
        return 0
    if case.workload_kind == "shared-prefix":
        return 1
    if case.workload_kind == "mixed-prefix":
        exact = sum(
            1 for index in range(config.workload.requests_per_case) if index % 4 == 1
        )
        shared_groups = {
            index % 4
            for index in range(config.workload.requests_per_case)
            if index % 4 in (2, 3)
        }
        return exact + len(shared_groups)
    return config.workload.requests_per_case


def _planned_scenario(case: ExecutionCase, config: SuiteConfig) -> dict[str, Any]:
    measure_path = case.result_dir / "measure.jsonl"
    measure_result = case.result_dir / "native-result.json"
    population_count = _planned_population_count(case, config)
    population_command = None
    if population_count:
        population_command = build_benchmark_command(
            case,
            config,
            case.result_dir / "populate.jsonl",
            case.result_dir / "population-result.json",
            num_prompts=population_count,
        )
    return {
        **_case_payload(case),
        "commands": {
            "server": build_server_command(case, config),
            "populate": population_command,
            "measure": build_benchmark_command(
                case,
                config,
                measure_path,
                measure_result,
                num_prompts=config.workload.requests_per_case,
            ),
        },
        "paths": {
            "measure_dataset": str(measure_path),
            "native_result": str(measure_result),
        },
    }


def load_tokenizer(config: SuiteConfig):
    """Load the operator-selected tokenizer only for real executions."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.workload.tokenizer,
        trust_remote_code=config.model.trust_remote_code,
    )


def _base_url(config: SuiteConfig) -> str:
    return f"http://{config.server.host}:{config.server.port}"


def _ensure_command_success(result: CommandResult, stage: str) -> None:
    if result.timed_out:
        raise BenchmarkExecutionError(f"{stage} command timed out", result)
    if result.returncode != 0:
        raise BenchmarkExecutionError(
            f"{stage} command exited with return code {result.returncode}", result
        )


def _command_result_payload(result: CommandResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "command": list(result.command),
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "elapsed_seconds": result.elapsed_seconds,
        "stdout_path": str(result.stdout_path),
        "stderr_path": str(result.stderr_path),
    }


def _common_record(
    case: ExecutionCase,
    config: SuiteConfig,
    *,
    commands: Mapping[str, Any],
    logs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "cache_mode": case.cache_mode.value,
        "workload_kind": case.workload_kind,
        "prompt_tokens": case.prompt_tokens,
        "prefix_ratio": case.prefix_ratio,
        "concurrency": case.concurrency,
        "request_rate": case.request_rate,
        "repetition": case.repetition,
        "model_id": config.model.id,
        "model_revision": None,
        "tensor_parallel_size": config.parallelism.tensor_parallel_size,
        "pipeline_parallel_size": config.parallelism.pipeline_parallel_size,
        "commands": dict(commands),
        "logs": dict(logs),
    }


def _categorize_error(error: BaseException) -> tuple[str, bool]:
    if isinstance(error, ConfigurationError):
        return "configuration_error", False
    if isinstance(error, ServerExitedError):
        return "server_start_error", True
    if isinstance(error, ServerReadinessTimeout):
        return "server_timeout", True
    if isinstance(error, BenchmarkExecutionError):
        return "benchmark_error", True
    if isinstance(error, NativeResultParseError):
        return "parse_error", False
    if isinstance(error, MetricsCollectionError):
        return "metrics_error", True
    if isinstance(error, ShutdownError):
        return "shutdown_error", True
    if isinstance(error, KeyboardInterrupt):
        return "interrupted", True
    return "benchmark_error", False


def _stop_running_server(
    server: ManagedProcess | Any | None, config: SuiteConfig
) -> None:
    if server is None:
        return
    try:
        stop_server(server, timeout_seconds=config.server.shutdown_timeout_seconds)
    except Exception as error:
        raise ShutdownError(f"failed to stop server: {error}") from error


def execute_case(
    case: ExecutionCase,
    config: SuiteConfig,
    tokenizer: Any,
    run_dir: Path,
) -> dict[str, Any]:
    """Execute one isolated cache benchmark case and persist its record."""
    started_at = _utc_now()
    server: ManagedProcess | Any | None = None
    sampler: ResourceSampler | Any | None = None
    sampler_summary: dict[str, Any] | None = None
    command_result: CommandResult | None = None
    stage = "configuration"

    server_command = build_server_command(case, config)
    environment = build_server_environment(case, config)
    logs = {
        "server_stdout": str(case.result_dir / "server.stdout.log"),
        "server_stderr": str(case.result_dir / "server.stderr.log"),
        "benchmark_stdout": str(case.result_dir / "benchmark.stdout.log"),
        "benchmark_stderr": str(case.result_dir / "benchmark.stderr.log"),
    }
    commands: dict[str, Any] = {"server": server_command}

    try:
        if case.cache_mode is CacheMode.TIERED_FS:
            if case.filesystem_cache_dir is None:
                raise ConfigurationError(
                    "tiered filesystem case has no filesystem cache directory"
                )
            create_owned_directory(
                case.filesystem_cache_dir, config.cache.filesystem.root_dir
            )

        stage = "workload"
        artifacts: WorkloadArtifacts = generate_workload(case, config, tokenizer)
        native_result_path = case.result_dir / "native-result.json"
        measure_command = build_benchmark_command(
            case,
            config,
            artifacts.measure_path,
            native_result_path,
            num_prompts=artifacts.num_measurement_prompts,
        )
        commands["measure"] = measure_command

        if artifacts.populate_path is not None:
            population_result_path = case.result_dir / "population-result.json"
            population_command = build_benchmark_command(
                case,
                config,
                artifacts.populate_path,
                population_result_path,
                num_prompts=artifacts.num_population_prompts,
            )
            commands["populate"] = population_command

        stage = "server_start"
        server = start_server(
            server_command,
            env=environment,
            stdout_path=Path(logs["server_stdout"]),
            stderr_path=Path(logs["server_stderr"]),
        )
        wait_for_server(server, _base_url(config), config.server.startup_timeout_seconds)

        if artifacts.populate_path is not None:
            stage = "population"
            command_result = run_command(
                commands["populate"],
                env=environment,
                stdout_path=case.result_dir / "population.stdout.log",
                stderr_path=case.result_dir / "population.stderr.log",
                timeout_seconds=BENCHMARK_TIMEOUT_SECONDS,
            )
            _ensure_command_success(command_result, "population")

        if case.workload_kind == "restart-persistence":
            stage = "restart_shutdown"
            _stop_running_server(server, config)
            server = None
            stage = "restart_start"
            server = start_server(
                server_command,
                env=environment,
                stdout_path=Path(logs["server_stdout"]),
                stderr_path=Path(logs["server_stderr"]),
            )
            wait_for_server(
                server, _base_url(config), config.server.startup_timeout_seconds
            )

        stage = "metrics_before"
        try:
            metrics_before = fetch_prometheus_snapshot(_base_url(config))
        except Exception as error:
            raise MetricsCollectionError(
                f"failed to collect pre-benchmark metrics: {error}"
            ) from error

        stage = "benchmark"
        sampler = ResourceSampler(server.process.pid)
        sampler.start()
        try:
            command_result = run_command(
                commands["measure"],
                env=environment,
                stdout_path=Path(logs["benchmark_stdout"]),
                stderr_path=Path(logs["benchmark_stderr"]),
                timeout_seconds=BENCHMARK_TIMEOUT_SECONDS,
            )
        finally:
            sampler_summary = sampler.stop()
            sampler = None
        _ensure_command_success(command_result, "measured benchmark")

        stage = "metrics_after"
        try:
            metrics_after = fetch_prometheus_snapshot(_base_url(config))
        except Exception as error:
            raise MetricsCollectionError(
                f"failed to collect post-benchmark metrics: {error}"
            ) from error

        stage = "normalization"
        try:
            normalized = normalize_native_result(
                native_result_path, before=metrics_before, after=metrics_after
            )
        except Exception as error:
            raise NativeResultParseError(
                f"failed to normalize native result: {error}"
            ) from error

        record = {
            **_common_record(case, config, commands=commands, logs=logs),
            "status": "completed",
            "started_at": started_at,
            "ended_at": _utc_now(),
            "workload_metadata": str(artifacts.metadata_path),
            "normalized": normalized,
            "resources": sampler_summary,
            "command_result": _command_result_payload(command_result),
        }
    except BaseException as error:
        status, retryable = _categorize_error(error)
        record = {
            **_common_record(case, config, commands=commands, logs=logs),
            "status": status,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "error": {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
                "retryable": retryable,
                "command_result": _command_result_payload(
                    getattr(error, "result", None) or command_result
                ),
            },
            "resources": sampler_summary,
        }
        if isinstance(error, KeyboardInterrupt):
            append_result(run_dir / "scenario-results.jsonl", record)
            rebuild_reports(run_dir)
            raise
    finally:
        if sampler is not None:
            sampler_summary = sampler.stop()
        shutdown_failure: ShutdownError | None = None
        if server is not None:
            try:
                _stop_running_server(server, config)
            except ShutdownError as error:
                shutdown_failure = error
        if shutdown_failure is not None:
            shutdown_payload = {
                "stage": "shutdown",
                "type": type(shutdown_failure).__name__,
                "message": str(shutdown_failure),
                "retryable": True,
            }
            if "record" not in locals() or record.get("status") == "completed":
                record = {
                    **_common_record(case, config, commands=commands, logs=logs),
                    "status": "shutdown_error",
                    "started_at": started_at,
                    "ended_at": _utc_now(),
                    "error": shutdown_payload,
                    "resources": sampler_summary,
                }
            else:
                record.setdefault("secondary_errors", []).append(shutdown_payload)

    append_result(run_dir / "scenario-results.jsonl", record)
    rebuild_reports(run_dir)
    return record


def _prepare_run(
    config_path: Path,
    config: SuiteConfig,
    *,
    dry_run: bool,
    selected_case_ids: Sequence[str],
) -> tuple[Path, list[ExecutionCase]]:
    fingerprint = _config_fingerprint(config_path)
    run_dir = create_owned_directory(
        config.results.root_dir / _run_id(fingerprint), config.results.root_dir
    )
    (run_dir / "workloads").mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "scenario-results.jsonl").touch(exist_ok=True)

    cases = build_execution_cases(config, run_dir)
    if selected_case_ids:
        selected = set(selected_case_ids)
        unknown = selected - {case.case_id for case in cases}
        if unknown:
            raise ConfigurationError(
                f"unknown case IDs: {', '.join(sorted(unknown))}"
            )
        cases = [case for case in cases if case.case_id in selected]

    scenarios = [_planned_scenario(case, config) for case in cases]
    _atomic_json(run_dir / "scenarios.json", scenarios)
    manifest = {
        "schema_version": 1,
        "base_commit": BASE_COMMIT,
        "config": _config_payload(config),
        "config_fingerprint": fingerprint,
        "selected_case_ids": [case.case_id for case in cases],
        "dry_run": dry_run,
        "created_at": _utc_now(),
        "cases": [
            {
                "case_id": item["case_id"],
                "cache_mode": item["cache_mode"],
                "workload_kind": item["workload_kind"],
            }
            for item in scenarios
        ],
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    return run_dir, cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark native vLLM cache modes")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--rebuild-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rebuild_report is not None:
        rebuild_reports(args.rebuild_report.expanduser().resolve())
        return 0
    if args.config is None:
        parser.error("--config is required unless --rebuild-report is present")

    config_path = args.config.expanduser().resolve()
    try:
        config = load_suite_config(config_path)
        run_dir, cases = _prepare_run(
            config_path,
            config,
            dry_run=args.dry_run,
            selected_case_ids=args.case_id,
        )
    except (OSError, ValueError, ConfigurationError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        _atomic_json(
            run_dir / "environment.json",
            {"status": "not_collected", "reason": "dry_run"},
        )
        rebuild_reports(run_dir)
        print(f"Dry run planned {len(cases)} cases in {run_dir}")
        return 0

    _atomic_json(run_dir / "environment.json", collect_environment_evidence())
    tokenizer = load_tokenizer(config)
    failed = False
    try:
        for case in cases:
            record = execute_case(case, config, tokenizer, run_dir)
            if record["status"] != "completed":
                failed = True
                if config.results.fail_fast:
                    break
    except KeyboardInterrupt:
        return 130
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
