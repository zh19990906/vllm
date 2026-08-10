# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

import pytest

from benchmarks.cache.process import (
    ServerExitedError,
    ServerReadinessTimeout,
    run_command,
    start_server,
    stop_server,
    wait_for_server,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_run_command_captures_output(tmp_path: Path) -> None:
    result = run_command(
        [sys.executable, "-c", "print('ok')"],
        env={},
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        timeout_seconds=5,
    )
    assert result.returncode == 0
    assert result.timed_out is False
    assert (tmp_path / "stdout.log").read_text().strip() == "ok"


def test_run_command_returns_timeout_record(tmp_path: Path) -> None:
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        env={},
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        timeout_seconds=0.1,
    )
    assert result.timed_out is True
    assert result.returncode is None


def test_wait_for_server_accepts_non_empty_models_response(tmp_path: Path) -> None:
    port = _free_port()
    script = (
        "from http.server import BaseHTTPRequestHandler,HTTPServer\n"
        "class H(BaseHTTPRequestHandler):\n"
        " def do_GET(self):\n"
        '  body=b\'{"data":[{"id":"m"}]}\'\n'
        "  self.send_response(200); "
        "self.send_header('Content-Type','application/json'); "
        "self.send_header('Content-Length',str(len(body))); "
        "self.end_headers(); self.wfile.write(body)\n"
        " def log_message(self,*args): pass\n"
        f"HTTPServer(('127.0.0.1',{port}),H).serve_forever()\n"
    )
    server = start_server(
        [sys.executable, "-c", script],
        env={},
        stdout_path=tmp_path / "server.out",
        stderr_path=tmp_path / "server.err",
    )
    try:
        wait_for_server(server, f"http://127.0.0.1:{port}", 5)
    finally:
        stop_server(server, timeout_seconds=2)


def test_wait_for_server_reports_early_exit(tmp_path: Path) -> None:
    server = start_server(
        [sys.executable, "-c", "raise SystemExit(3)"],
        env={},
        stdout_path=tmp_path / "server.out",
        stderr_path=tmp_path / "server.err",
    )
    with pytest.raises(ServerExitedError, match="exited"):
        wait_for_server(server, "http://127.0.0.1:1", 2)
    stop_server(server, timeout_seconds=1)


def test_wait_for_server_times_out(tmp_path: Path) -> None:
    server = start_server(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        env={},
        stdout_path=tmp_path / "server.out",
        stderr_path=tmp_path / "server.err",
    )
    try:
        with pytest.raises(ServerReadinessTimeout, match="not ready"):
            wait_for_server(server, "http://127.0.0.1:1", 0.2)
    finally:
        stop_server(server, timeout_seconds=1)


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_stop_server_terminates_process_group(tmp_path: Path) -> None:
    script = (
        "import subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(30)\n"
    )
    server = start_server(
        [sys.executable, "-c", script],
        env={},
        stdout_path=tmp_path / "server.out",
        stderr_path=tmp_path / "server.err",
    )
    deadline = time.monotonic() + 3
    child_pid = None
    while time.monotonic() < deadline:
        text = (tmp_path / "server.out").read_text(encoding="utf-8")
        if text.strip():
            child_pid = int(text.strip())
            break
        time.sleep(0.05)
    assert child_pid is not None

    result = stop_server(server, timeout_seconds=1)
    assert result.returncode is not None
    assert server.process.poll() is not None

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and Path(f"/proc/{child_pid}").exists():
        time.sleep(0.05)
    assert not Path(f"/proc/{child_pid}").exists()
