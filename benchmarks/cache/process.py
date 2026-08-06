from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import requests


@dataclass(frozen=True, slots=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int | None
    timed_out: bool
    started_at: str
    ended_at: str
    elapsed_seconds: float
    stdout_path: Path
    stderr_path: Path


@dataclass(slots=True)
class ManagedProcess:
    command: tuple[str, ...]
    process: subprocess.Popen[bytes]
    stdout_handle: BinaryIO
    stderr_handle: BinaryIO
    stdout_path: Path
    stderr_path: Path
    started_at: str


class ServerExitedError(RuntimeError):
    """Raised when a managed server exits before becoming ready."""


class ServerReadinessTimeout(TimeoutError):
    """Raised when a managed server never becomes ready."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_since(started_at: str) -> float:
    started = datetime.fromisoformat(started_at)
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _signal_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), sig)
    else:
        if sig is signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def _terminate_group(
    process: subprocess.Popen[bytes], timeout_seconds: float
) -> int | None:
    if process.poll() is not None:
        return process.returncode
    _signal_group(process, signal.SIGTERM)
    try:
        return process.wait(timeout=max(0.0, timeout_seconds))
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            return process.wait(timeout=5)
    return process.poll()


def run_command(
    command: list[str],
    *,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> CommandResult:
    """Run one command with captured logs and process-group timeout cleanup."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    timed_out = False
    returncode: int | None

    with stdout_path.open("wb") as stdout_handle, stderr_path.open(
        "wb"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            shell=False,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_group(process, timeout_seconds=1.0)
            returncode = None

    return CommandResult(
        command=tuple(command),
        returncode=returncode,
        timed_out=timed_out,
        started_at=started_at,
        ended_at=_utc_now(),
        elapsed_seconds=time.monotonic() - started_monotonic,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def start_server(
    command: list[str],
    *,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> ManagedProcess:
    """Start a long-running server in an isolated process group."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("wb")
    stderr_handle = stderr_path.open("wb")
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            shell=False,
            start_new_session=True,
        )
    except BaseException:
        stdout_handle.close()
        stderr_handle.close()
        raise
    return ManagedProcess(
        command=tuple(command),
        process=process,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=_utc_now(),
    )


def wait_for_server(
    server: ManagedProcess,
    base_url: str,
    timeout_seconds: float,
) -> None:
    """Wait for ``/v1/models`` to return HTTP 200 with a non-empty data list."""
    deadline = time.monotonic() + timeout_seconds
    models_url = f"{base_url.rstrip('/')}/v1/models"
    last_reason = "no response"

    while time.monotonic() < deadline:
        returncode = server.process.poll()
        if returncode is not None:
            raise ServerExitedError(
                f"server exited with return code {returncode} before readiness"
            )
        try:
            response = requests.get(models_url, timeout=(2, 2))
            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                    if payload["data"]:
                        return
                    last_reason = "models response contained an empty data list"
                else:
                    last_reason = "models response did not contain a data list"
            else:
                last_reason = f"HTTP {response.status_code}"
        except (requests.RequestException, ValueError) as error:
            last_reason = str(error)

        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.5, remaining))

    returncode = server.process.poll()
    if returncode is not None:
        raise ServerExitedError(
            f"server exited with return code {returncode} before readiness"
        )
    raise ServerReadinessTimeout(
        f"server was not ready within {timeout_seconds} seconds: {last_reason}"
    )


def stop_server(server: ManagedProcess, *, timeout_seconds: float) -> CommandResult:
    """Stop a server process group gracefully, escalating to SIGKILL."""
    try:
        returncode = _terminate_group(server.process, timeout_seconds)
    finally:
        server.stdout_handle.close()
        server.stderr_handle.close()
    return CommandResult(
        command=server.command,
        returncode=returncode,
        timed_out=False,
        started_at=server.started_at,
        ended_at=_utc_now(),
        elapsed_seconds=_elapsed_since(server.started_at),
        stdout_path=server.stdout_path,
        stderr_path=server.stderr_path,
    )
