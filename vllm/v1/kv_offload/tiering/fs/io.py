# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random
import threading

# O_DIRECT is Linux-specific and not available on macOS
O_DIRECT = getattr(os, "O_DIRECT", 0)

# Thread-local storage for unique temporary file suffixes
_thread_local = threading.local()


def _get_tmp_suffix() -> str:
    """Generate a thread-local unique suffix for temporary files."""
    try:
        return _thread_local.tmp_suffix
    except AttributeError:
        _thread_local.tmp_suffix = f"_{random.randint(0, 2**63 - 1)}.tmp"
        return _thread_local.tmp_suffix


def make_temp_path(dest_path: str) -> str:
    """Return the recognized temporary path for a destination block."""
    return dest_path + _get_tmp_suffix()


def _ensure_dirs(path: str) -> None:
    """Create parent directories of *path* if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def store_block(
    dest_path: str,
    tmp_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """
    Write one KV block through the caller-owned temporary path.

    Capacity/admission cleanup decisions belong to FileSystemTierManager.
    This raw helper only performs I/O and propagates failures.
    """
    _ensure_dirs(dest_path)

    view_slice = buffer.cast("B")[offset : offset + block_size]
    fd = os.open(
        tmp_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | O_DIRECT,
        0o644,
    )
    try:
        written = os.write(fd, view_slice)
        if written < len(view_slice):
            raise OSError(
                f"Short write: expected {len(view_slice)} bytes, "
                f"wrote {written}"
            )
    finally:
        os.close(fd)

    os.replace(tmp_path, dest_path)

def load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """
    Read one KV block from disk.

    Final-file invalidation/deletion belongs to FileSystemTierManager and
    FileSystemCapacityManager; this helper never mutates cache metadata.
    """
    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]
    try:
        fd = os.open(source_path, os.O_RDONLY | O_DIRECT)
        bytes_read = os.readv(fd, [view_slice])
        if bytes_read < block_size:
            raise OSError(
                f"Short read: expected {block_size} bytes, "
                f"read {bytes_read}"
            )
    finally:
        if fd is not None:
            os.close(fd)
