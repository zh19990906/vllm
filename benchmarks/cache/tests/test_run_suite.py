# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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


def _write_config(tmp_path: Path, valid_config_dict: dict) -> Path:
    config = json.loads(json.dumps(valid_config_dict))
    config["workload"]["prompt_tokens"] = [16]
    config["workload"]["concurrency"] = [1]
    config["workload"]["request_rate"] = ["inf"]
    config["workload"]["requests_per_case"] = 2
    config["results"]["root_dir"] = str(tmp_path / "results")
    config["cache"]["filesystem"]["root_dir"] = str(tmp_path / "kv")
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _command_result(command: list[str], stdout: Path, stderr: Path, returncode=0):
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    return CommandResult(
        command=tuple(command),
        returncode=returncode,
        timed_out=False,
        started_at="2026-08-06T00:00:00+00:00",
        ended_at="2026-08-06T00:00:01+00:00",
        elapsed_seconds=1.0,
        stdout_path=stdout,
        stderr_path=stderr,
    )


def test_dry_run_writes_manifest_without_starting_processes(
    monkeypatch, capsys, valid_config_dict: dict, tmp_path: Path
) -> None:
    from benchmarks.cache import run_suite

    config_path = _write_config(tmp_path, valid_config_dict)
    monkeypatch.setattr(
        run_suite,
        "start_server",
        lambda *args, **kwargs: pytest.fail("start_server called in dry run"),
    )
    monkeypatch.setattr(
        run_suite,
        "load_tokenizer",
        lambda *args, **kwargs: pytest.fail("tokenizer loaded in dry run"),
    )
    monkeypatch.setattr(
        run_suite,
        "collect_environment_evidence",
        lambda: pytest.fail("environment collected in dry run"),
    )

    exit_code = run_suite.main(["--config", str(config_path), "--dry-run"])

    assert exit_code == 0
    run_dirs = [path for path in (tmp_path / "results").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert {item["cache_mode"] for item in manifest["cases"]} == {
        "no-cache",
        "gpu-apc",
        "cpu-offload",
        "tiered-fs",
    }
    assert not (tmp_path / "kv").exists()
    output = capsys.readouterr().out
    assert "Dry run planned" in output
    assert str(run_dirs[0]) in output


def test_partial_results_survive_and_later_cases_continue(
    monkeypatch, valid_config_dict: dict, tmp_path: Path
) -> None:
    from benchmarks.cache import run_suite

    config_path = _write_config(tmp_path, valid_config_dict)
    cases: list[ExecutionCase] = []
    for index, mode in enumerate(
        [CacheMode.NO_CACHE, CacheMode.GPU_APC, CacheMode.CPU_OFFLOAD]
    ):
        cases.append(
            ExecutionCase(
                case_id=f"case-{index}",
                cache_mode=mode,
                workload_kind="cold-unique",
                prompt_tokens=16,
                prefix_ratio=0.0,
                concurrency=1,
                request_rate="inf",
                repetition=0,
                result_dir=(
                    tmp_path / "results" / "placeholder" / "raw" / f"case-{index}"
                ),
                filesystem_cache_dir=None,
            )
        )
    monkeypatch.setattr(
        run_suite,
        "build_execution_cases",
        lambda config, run_dir: [
            replace(case, result_dir=run_dir / "raw" / case.case_id) for case in cases
        ],
    )
    monkeypatch.setattr(run_suite, "load_tokenizer", lambda config: FakeTokenizer())
    monkeypatch.setattr(run_suite, "collect_environment_evidence", lambda: {})
    monkeypatch.setattr(
        run_suite,
        "fetch_prometheus_snapshot",
        lambda base_url: PrometheusSnapshot("now", {}, ""),
    )
    stops: list[str] = []
    monkeypatch.setattr(
        run_suite,
        "start_server",
        lambda command, **kwargs: SimpleNamespace(
            command=tuple(command), process=SimpleNamespace(pid=123, poll=lambda: None)
        ),
    )
    monkeypatch.setattr(run_suite, "wait_for_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_suite,
        "stop_server",
        lambda server, **kwargs: stops.append(server.command[2]) or None,
    )

    class FakeSampler:
        def __init__(self, pid):
            self.pid = pid

        def start(self):
            return None

        def stop(self):
            return {}

    monkeypatch.setattr(run_suite, "ResourceSampler", FakeSampler)
    calls = 0

    def fake_run(command, *, env, stdout_path, stderr_path, timeout_seconds):
        nonlocal calls
        calls += 1
        result_path = (
            Path(command[command.index("--result-dir") + 1])
            / command[command.index("--result-filename") + 1]
        )
        if calls == 2:
            return _command_result(command, stdout_path, stderr_path, returncode=2)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({"completed": 2, "failed": 0, "p95_ttft_ms": 10.0}),
            encoding="utf-8",
        )
        return _command_result(command, stdout_path, stderr_path)

    monkeypatch.setattr(run_suite, "run_command", fake_run)

    assert run_suite.main(["--config", str(config_path)]) == 1
    run_dir = next(path for path in (tmp_path / "results").iterdir() if path.is_dir())
    records = [
        json.loads(line)
        for line in (run_dir / "scenario-results.jsonl").read_text().splitlines()
    ]
    assert [record["status"] for record in records] == [
        "completed",
        "benchmark_error",
        "completed",
    ]
    assert len(stops) == 3


def test_restart_persistence_restarts_before_measured_run(
    monkeypatch, valid_config_dict: dict, tmp_path: Path
) -> None:
    from benchmarks.cache import run_suite

    config_path = _write_config(tmp_path, valid_config_dict)
    events: list[str] = []

    def one_case(config, run_dir):
        return [
            ExecutionCase(
                case_id="restart",
                cache_mode=CacheMode.TIERED_FS,
                workload_kind="restart-persistence",
                prompt_tokens=16,
                prefix_ratio=0.0,
                concurrency=1,
                request_rate="inf",
                repetition=0,
                result_dir=run_dir / "raw" / "restart",
                filesystem_cache_dir=(
                    config.cache.filesystem.root_dir / run_dir.name / "restart"
                ),
            )
        ]

    monkeypatch.setattr(run_suite, "build_execution_cases", one_case)
    monkeypatch.setattr(run_suite, "load_tokenizer", lambda config: FakeTokenizer())
    monkeypatch.setattr(run_suite, "collect_environment_evidence", lambda: {})

    def fake_start(command, **kwargs):
        events.append("start")
        return SimpleNamespace(
            command=tuple(command), process=SimpleNamespace(pid=123, poll=lambda: None)
        )

    monkeypatch.setattr(run_suite, "start_server", fake_start)
    monkeypatch.setattr(
        run_suite,
        "wait_for_server",
        lambda *args, **kwargs: events.append("ready"),
    )
    monkeypatch.setattr(
        run_suite,
        "stop_server",
        lambda server, **kwargs: events.append("stop") or None,
    )

    snapshots = iter(["metrics-before", "metrics-after"])

    def fake_snapshot(base_url):
        name = next(snapshots)
        events.append(name)
        return PrometheusSnapshot(name, {}, "")

    monkeypatch.setattr(run_suite, "fetch_prometheus_snapshot", fake_snapshot)

    class FakeSampler:
        def __init__(self, pid):
            self.pid = pid

        def start(self):
            events.append("sampler-start")

        def stop(self):
            events.append("sampler-stop")
            return {}

    monkeypatch.setattr(run_suite, "ResourceSampler", FakeSampler)

    def fake_run(command, *, env, stdout_path, stderr_path, timeout_seconds):
        filename = command[command.index("--result-filename") + 1]
        events.append("population" if filename.startswith("population") else "measure")
        result_path = Path(command[command.index("--result-dir") + 1]) / filename
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({"completed": 2, "failed": 0, "p95_ttft_ms": 10.0}),
            encoding="utf-8",
        )
        return _command_result(command, stdout_path, stderr_path)

    monkeypatch.setattr(run_suite, "run_command", fake_run)

    assert run_suite.main(["--config", str(config_path)]) == 0
    assert events == [
        "start",
        "ready",
        "population",
        "stop",
        "start",
        "ready",
        "metrics-before",
        "sampler-start",
        "measure",
        "sampler-stop",
        "metrics-after",
        "stop",
    ]


def test_shutdown_failure_replaces_completed_status(
    monkeypatch, suite_config, cold_case, tmp_path: Path
) -> None:
    from benchmarks.cache import run_suite

    run_dir = suite_config.results.root_dir / "shutdown-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_suite, "load_tokenizer", lambda config: FakeTokenizer())
    monkeypatch.setattr(
        run_suite,
        "start_server",
        lambda command, **kwargs: SimpleNamespace(
            command=tuple(command), process=SimpleNamespace(pid=123, poll=lambda: None)
        ),
    )
    monkeypatch.setattr(run_suite, "wait_for_server", lambda *args, **kwargs: None)
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

    def fake_run(command, *, env, stdout_path, stderr_path, timeout_seconds):
        result_path = (
            Path(command[command.index("--result-dir") + 1])
            / command[command.index("--result-filename") + 1]
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({"completed": 2, "failed": 0, "p95_ttft_ms": 10.0}),
            encoding="utf-8",
        )
        return _command_result(command, stdout_path, stderr_path)

    monkeypatch.setattr(run_suite, "run_command", fake_run)
    monkeypatch.setattr(
        run_suite,
        "stop_server",
        lambda server, **kwargs: (_ for _ in ()).throw(RuntimeError("stop failed")),
    )

    adjusted_case = replace(cold_case, result_dir=run_dir / "raw" / cold_case.case_id)
    record = run_suite.execute_case(
        adjusted_case, suite_config, FakeTokenizer(), run_dir
    )

    assert record["status"] == "shutdown_error"
    assert record["error"]["stage"] == "shutdown"


def test_invalid_config_returns_nonzero_code(tmp_path: Path) -> None:
    from benchmarks.cache import run_suite

    assert run_suite.main(["--config", str(tmp_path / "missing.yaml")]) == 2


@pytest.mark.parametrize(
    "name", ["example-7b.yaml", "example-70b.yaml", "example-397b.yaml"]
)
def test_example_config_is_valid(name: str) -> None:
    from benchmarks.cache.config import load_suite_config

    config = load_suite_config(Path("benchmarks/cache/configs") / name)
    assert config.schema_version == 1
    assert config.parallelism.tensor_parallel_size in {1, 2, 4, 8}


def test_fake_executable_end_to_end(
    monkeypatch, valid_config_dict: dict, tmp_path: Path
) -> None:
    from benchmarks.cache import run_suite
    from benchmarks.cache.config import load_suite_config
    from benchmarks.cache.scenarios import build_execution_cases

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_vllm = bin_dir / "vllm"
    fake_vllm.write_text(
        """#!/usr/bin/env python3
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

args = sys.argv[1:]
if args == ['--version']:
    print('vllm fake')
    raise SystemExit(0)
if args[:1] == ['serve']:
    host = args[args.index('--host') + 1]
    port = int(args[args.index('--port') + 1])
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/v1/models':
                body = b'{"data":[{"id":"example"}]}'
                content_type = 'application/json'
            elif self.path == '/metrics':
                body = b'vllm:prefix_cache_hit_tokens_total 10\n'
                content_type = 'text/plain'
            else:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    HTTPServer((host, port), Handler).serve_forever()
if args[:2] == ['bench', 'serve']:
    result_dir = Path(args[args.index('--result-dir') + 1])
    result_name = args[args.index('--result-filename') + 1]
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / result_name).write_text(json.dumps({
        'completed': 2,
        'failed': 0,
        'duration': 1.0,
        'request_throughput': 2.0,
        'output_throughput': 10.0,
        'total_token_throughput': 42.0,
        'p50_ttft_ms': 10.0,
        'p95_ttft_ms': 15.0,
        'p99_ttft_ms': 20.0,
        'p50_tpot_ms': 2.0,
        'p95_tpot_ms': 3.0,
        'p99_tpot_ms': 4.0,
    }), encoding='utf-8')
    print('benchmark complete')
    raise SystemExit(0)
raise SystemExit(2)
""",
        encoding="utf-8",
    )
    fake_vllm.chmod(0o755)
    fake_nvidia = bin_dir / "nvidia-smi"
    fake_nvidia.write_text(
        """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
if any('query-compute-apps' in arg for arg in args):
    raise SystemExit(0)
if args[:2] == ['topo', '-m']:
    print('GPU0')
else:
    print('Fake GPU, 72 GiB, fake-driver')
""",
        encoding="utf-8",
    )
    fake_nvidia.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path('/usr/bin')}:{Path('/bin')}")
    monkeypatch.setattr(run_suite, "load_tokenizer", lambda config: FakeTokenizer())

    config_path = _write_config(tmp_path, valid_config_dict)
    config = load_suite_config(config_path)
    selected = next(
        case.case_id
        for case in build_execution_cases(config, tmp_path / "preview")
        if case.cache_mode is CacheMode.NO_CACHE and case.workload_kind == "cold-unique"
    )

    assert run_suite.main(["--config", str(config_path), "--case-id", selected]) == 0

    run_dir = next(path for path in (tmp_path / "results").iterdir() if path.is_dir())
    for name in (
        "manifest.json",
        "environment.json",
        "scenario-results.jsonl",
        "summary.csv",
        "report.md",
        "scenarios.json",
    ):
        assert (run_dir / name).exists()
    record = json.loads((run_dir / "scenario-results.jsonl").read_text().strip())
    assert record["status"] == "completed"
    assert record["normalized"]["benchmark"]["ttft_ms"]["p95"] == 15.0
    assert (run_dir / "raw" / selected / "server.stdout.log").exists()
