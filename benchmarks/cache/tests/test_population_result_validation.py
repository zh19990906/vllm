from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from benchmarks.cache.config import SuiteConfig
from benchmarks.cache.metrics import PrometheusSnapshot
from benchmarks.cache.process import CommandResult
from benchmarks.cache.scenarios import CacheMode, ExecutionCase


class FakeTokenizer:
    all_special_ids: list[int] = []
    vocab_size = 10000

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(part) for part in text.split()] if text else []

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _command_result(command: list[str], stdout: Path, stderr: Path) -> CommandResult:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    return CommandResult(
        command=tuple(command),
        returncode=0,
        timed_out=False,
        started_at="2026-08-07T00:00:00+00:00",
        ended_at="2026-08-07T00:00:01+00:00",
        elapsed_seconds=1.0,
        stdout_path=stdout,
        stderr_path=stderr,
    )


def _config(valid_config_dict: dict) -> SuiteConfig:
    payload = json.loads(json.dumps(valid_config_dict))
    payload["workload"]["prompt_tokens"] = [16]
    payload["workload"]["concurrency"] = [1]
    payload["workload"]["request_rate"] = ["inf"]
    payload["workload"]["requests_per_case"] = 24
    payload["workload"]["output_tokens"] = 1
    return SuiteConfig.model_validate(payload)


def _case(config: SuiteConfig, run_dir: Path) -> ExecutionCase:
    return ExecutionCase(
        case_id="population-result-validation",
        cache_mode=CacheMode.NO_CACHE,
        workload_kind="warm-exact-prefix",
        prompt_tokens=16,
        prefix_ratio=0.0,
        concurrency=1,
        request_rate="inf",
        repetition=0,
        result_dir=run_dir / "raw" / "population-result-validation",
        filesystem_cache_dir=None,
    )


def _patch_runtime(monkeypatch) -> None:
    from benchmarks.cache import run_suite

    monkeypatch.setattr(
        run_suite,
        "start_server",
        lambda command, **kwargs: SimpleNamespace(
            command=tuple(command), process=SimpleNamespace(pid=123, poll=lambda: None)
        ),
    )
    monkeypatch.setattr(run_suite, "wait_for_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_suite, "stop_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_suite,
        "fetch_prometheus_snapshot",
        lambda base_url: PrometheusSnapshot("now", {}, ""),
    )

    class FakeSampler:
        def __init__(self, pid):
            self.pid = pid

        def start(self):
            return None

        def stop(self):
            return {}

    monkeypatch.setattr(run_suite, "ResourceSampler", FakeSampler)


def test_partial_population_result_stops_before_measurement(
    monkeypatch, valid_config_dict: dict, tmp_path: Path
) -> None:
    from benchmarks.cache import run_suite

    config = _config(valid_config_dict)
    run_dir = config.results.root_dir / "partial-population"
    run_dir.mkdir(parents=True)
    case = _case(config, run_dir)
    _patch_runtime(monkeypatch)

    result_names: list[str] = []

    def fake_run(command, *, env, stdout_path, stderr_path, timeout_seconds):
        result_name = command[command.index("--result-filename") + 1]
        result_names.append(result_name)
        result_path = Path(command[command.index("--result-dir") + 1]) / result_name
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if result_name == "population-result.json":
            payload = {"completed": 20, "failed": 4}
        else:
            payload = {"completed": 24, "failed": 0}
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return _command_result(command, stdout_path, stderr_path)

    monkeypatch.setattr(run_suite, "run_command", fake_run)

    record = run_suite.execute_case(case, config, FakeTokenizer(), run_dir)

    assert record["status"] == "benchmark_error"
    assert record["error"]["stage"] == "population"
    assert "expected=24" in record["error"]["message"]
    assert "completed=20" in record["error"]["message"]
    assert "failed=4" in record["error"]["message"]
    assert result_names == ["population-result.json"]


def test_complete_population_result_allows_measurement(
    monkeypatch, valid_config_dict: dict, tmp_path: Path
) -> None:
    from benchmarks.cache import run_suite

    config = _config(valid_config_dict)
    run_dir = config.results.root_dir / "complete-population"
    run_dir.mkdir(parents=True)
    case = _case(config, run_dir)
    _patch_runtime(monkeypatch)

    result_names: list[str] = []

    def fake_run(command, *, env, stdout_path, stderr_path, timeout_seconds):
        result_name = command[command.index("--result-filename") + 1]
        result_names.append(result_name)
        result_path = Path(command[command.index("--result-dir") + 1]) / result_name
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({"completed": 24, "failed": 0}), encoding="utf-8"
        )
        return _command_result(command, stdout_path, stderr_path)

    monkeypatch.setattr(run_suite, "run_command", fake_run)

    record = run_suite.execute_case(case, config, FakeTokenizer(), run_dir)

    assert record["status"] == "completed"
    assert result_names == ["population-result.json", "native-result.json"]
