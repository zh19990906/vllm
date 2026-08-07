# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import errno
import mmap
import os
import uuid

import pytest

from vllm.v1.kv_offload.cpu import shared_offload_region as sor


PAGE_SIZE = mmap.PAGESIZE


def _anonymous_mmap(size: int = 3 * PAGE_SIZE) -> mmap.mmap:
    return mmap.mmap(
        -1,
        size,
        flags=mmap.MAP_SHARED,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )


def test_get_populate_write_fn_keeps_native_path(monkeypatch):
    calls: list[tuple[int, int]] = []

    def native(mm: mmap.mmap, offset: int, length: int) -> None:
        calls.append((offset, length))

    monkeypatch.setattr(sor, "_madvise_populate_write", native)

    mm = _anonymous_mmap()
    try:
        populate = sor._get_populate_write_fn(mm)
        assert populate is native
        assert calls == [(0, PAGE_SIZE)]
    finally:
        mm.close()


def test_get_populate_write_fn_falls_back_on_einval(monkeypatch):
    def unsupported(mm: mmap.mmap, offset: int, length: int) -> None:
        raise OSError(errno.EINVAL, "unsupported")

    monkeypatch.setattr(sor, "_madvise_populate_write", unsupported)

    mm = _anonymous_mmap()
    try:
        assert sor._get_populate_write_fn(mm) is sor._fallback_populate_write
    finally:
        mm.close()


def test_fallback_populate_write_preserves_existing_bytes():
    mm = _anonymous_mmap()
    try:
        sentinels = [0x11, 0x7F, 0xE3]
        for page, value in enumerate(sentinels):
            mm[page * PAGE_SIZE] = value

        sor._fallback_populate_write(mm, 0, 3 * PAGE_SIZE)

        assert [mm[page * PAGE_SIZE] for page in range(3)] == sentinels
    finally:
        mm.close()


def test_get_populate_write_fn_propagates_unexpected_oserror(monkeypatch):
    def failed(mm: mmap.mmap, offset: int, length: int) -> None:
        raise OSError(errno.EIO, "simulated I/O failure")

    monkeypatch.setattr(sor, "_madvise_populate_write", failed)

    mm = _anonymous_mmap()
    try:
        with pytest.raises(OSError) as exc_info:
            sor._get_populate_write_fn(mm)
        assert exc_info.value.errno == errno.EIO
    finally:
        mm.close()


def test_shared_region_constructor_uses_fallback_on_einval(monkeypatch):
    fallback_calls: list[tuple[int, int]] = []
    real_fallback = sor._fallback_populate_write

    def unsupported(mm: mmap.mmap, offset: int, length: int) -> None:
        raise OSError(errno.EINVAL, "unsupported")

    def fallback(mm: mmap.mmap, offset: int, length: int) -> None:
        fallback_calls.append((offset, length))
        real_fallback(mm, offset, length)

    monkeypatch.setattr(sor, "_madvise_populate_write", unsupported)
    monkeypatch.setattr(sor, "_fallback_populate_write", fallback)

    engine_id = f"madvise_fallback_{uuid.uuid4().hex}"
    path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
    region = None
    try:
        region = sor.SharedOffloadRegion(
            engine_id=engine_id,
            num_blocks=3,
            rank=0,
            kv_bytes_per_block=PAGE_SIZE,
            cpu_page_size=PAGE_SIZE,
        )
        assert fallback_calls == [
            (0, PAGE_SIZE),
            (PAGE_SIZE, PAGE_SIZE),
            (2 * PAGE_SIZE, PAGE_SIZE),
        ]
    finally:
        if region is not None:
            region.cleanup()
        if os.path.exists(path):
            os.unlink(path)
