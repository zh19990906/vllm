# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vllm.v1.kv_offload.tiering.fs.capacity import (
    FileSystemCapacityManager,
)


def managed_path(
    root: Path,
    *,
    hash_hex: str = "0011223344556677",
    group: int = 0,
) -> Path:
    path = (
        root
        / hash_hex[:3]
        / f"{hash_hex[3:5]}_g{group}"
        / f"{hash_hex}.bin"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def recognized_temp(final_path: Path, token: int = 17) -> Path:
    # Matches the temp naming shape already used by the capacity tests:
    # <managed-final>.bin_<token>.tmp
    return Path(f"{final_path}_{token}.tmp")


class FileSystemCapacityRestartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def manager(
        self,
        *,
        max_bytes: int = 100,
        expected_file_size: int | None = None,
    ) -> FileSystemCapacityManager:
        return FileSystemCapacityManager(
            namespace_root=str(self.root),
            max_bytes=max_bytes,
            expected_file_size=expected_file_size,
        )

    def test_namespace_has_single_capacity_owner(self) -> None:
        first = self.manager()
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "already|owned|lock",
            ):
                self.manager()
        finally:
            first.close()

        # Releasing the lifetime owner must allow a later restart.
        with self.manager():
            pass

    def test_capacity_lock_is_control_metadata_not_accounted_data(self) -> None:
        with self.manager(max_bytes=1) as cap:
            lock_path = self.root / ".capacity.lock"
            self.assertTrue(lock_path.exists())

            snap = cap.snapshot()
            self.assertEqual(snap.accounted_bytes, 0)
            self.assertEqual(snap.reserved_bytes, 0)

        # The persistent lock file must not make the next restart fail as an
        # unknown artifact.
        with self.manager(max_bytes=1) as cap:
            self.assertEqual(cap.snapshot().accounted_bytes, 0)

    def test_low_physical_free_space_warns_but_keeps_logical_max(
        self,
    ) -> None:
        disk_usage = mock.Mock(
            total=1_000,
            used=950,
            free=50,
        )

        with mock.patch(
            "vllm.v1.kv_offload.tiering.fs.capacity.shutil.disk_usage",
            return_value=disk_usage,
        ):
            with self.assertLogs(
                "vllm.v1.kv_offload.tiering.fs.capacity",
                level="WARNING",
            ) as logs:
                with self.manager(max_bytes=100) as cap:
                    snap = cap.snapshot()
                    self.assertEqual(snap.max_bytes, 100)
                    self.assertEqual(snap.accounted_bytes, 0)

        self.assertTrue(
            any(
                "physical" in message.lower()
                and "enospc" in message.lower()
                for message in logs.output
            )
        )

    def test_restart_removes_recognized_temp_before_ready(self) -> None:
        final_path = managed_path(self.root)
        temp_path = recognized_temp(final_path)
        temp_path.write_bytes(b"x" * 40)

        with self.manager(max_bytes=100) as cap:
            self.assertFalse(temp_path.exists())
            self.assertEqual(cap.snapshot().accounted_bytes, 0)

    def test_restart_fails_if_recognized_temp_cannot_be_removed(self) -> None:
        final_path = managed_path(self.root)
        temp_path = recognized_temp(final_path)
        temp_path.write_bytes(b"x" * 40)

        real_unlink = os.unlink

        def selective_unlink(path: str) -> None:
            if os.path.abspath(path) == os.path.abspath(temp_path):
                raise PermissionError("injected temp cleanup failure")
            real_unlink(path)

        with mock.patch(
            "vllm.v1.kv_offload.tiering.fs.capacity.os.unlink",
            side_effect=selective_unlink,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "temp|temporary|cleanup",
            ):
                self.manager(max_bytes=100)

        self.assertTrue(temp_path.exists())

    def test_failed_restart_releases_capacity_lock(self) -> None:
        unknown = self.root / "mystery.dat"
        unknown.write_bytes(b"unknown")

        with self.assertRaisesRegex(
            RuntimeError,
            "unknown|artifact",
        ):
            self.manager()

        # Constructor failure must not leak the lifetime flock.
        unknown.unlink()

        with self.manager(max_bytes=1) as cap:
            self.assertEqual(cap.snapshot().accounted_bytes, 0)

    def test_restart_fails_fast_on_unknown_regular_file(self) -> None:
        unknown = self.root / "mystery.dat"
        unknown.write_bytes(b"unknown")

        with self.assertRaisesRegex(
            RuntimeError,
            "unknown|artifact",
        ):
            self.manager()

        self.assertTrue(unknown.exists())

    def test_restart_fails_fast_on_symlink(self) -> None:
        unknown = self.root / "mystery-link"
        unknown.symlink_to("/dev/null")

        with self.assertRaisesRegex(
            RuntimeError,
            "symlink|unknown|artifact",
        ):
            self.manager()

        self.assertTrue(unknown.is_symlink())

    @unittest.skipUnless(
        hasattr(os, "mkfifo"),
        "FIFO creation is unavailable on this platform",
    )
    def test_restart_fails_fast_on_special_file(self) -> None:
        fifo = self.root / "mystery-fifo"
        os.mkfifo(fifo)

        with self.assertRaisesRegex(
            RuntimeError,
            "special|unknown|artifact",
        ):
            self.manager()

        self.assertTrue(fifo.exists())

    def test_wrong_size_final_is_removed_during_restart(self) -> None:
        final_path = managed_path(self.root)
        final_path.write_bytes(b"x" * 30)

        with self.manager(
            max_bytes=100,
            expected_file_size=40,
        ) as cap:
            self.assertFalse(final_path.exists())
            self.assertEqual(cap.snapshot().accounted_bytes, 0)

    def test_restart_fails_if_wrong_size_final_cannot_be_removed(self) -> None:
        final_path = managed_path(self.root)
        final_path.write_bytes(b"x" * 30)

        real_unlink = os.unlink

        def selective_unlink(path: str) -> None:
            if os.path.abspath(path) == os.path.abspath(final_path):
                raise PermissionError("injected corrupt-final cleanup failure")
            real_unlink(path)

        with mock.patch(
            "vllm.v1.kv_offload.tiering.fs.capacity.os.unlink",
            side_effect=selective_unlink,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "size|corrupt|cleanup",
            ):
                self.manager(
                    max_bytes=100,
                    expected_file_size=40,
                )

        self.assertTrue(final_path.exists())

    def test_restart_shrinks_over_capacity_by_mtime_lru(self) -> None:
        oldest = managed_path(
            self.root,
            hash_hex="0011223344556677",
        )
        middle = managed_path(
            self.root,
            hash_hex="1111223344556677",
        )
        newest = managed_path(
            self.root,
            hash_hex="2221223344556677",
        )

        for path in (oldest, middle, newest):
            path.write_bytes(b"x" * 40)

        os.utime(oldest, ns=(1, 1))
        os.utime(middle, ns=(2, 2))
        os.utime(newest, ns=(3, 3))

        with self.manager(
            max_bytes=80,
            expected_file_size=40,
        ) as cap:
            self.assertFalse(oldest.exists())
            self.assertTrue(middle.exists())
            self.assertTrue(newest.exists())
            self.assertEqual(cap.snapshot().accounted_bytes, 80)

    def test_restart_lru_uses_path_as_stable_mtime_tiebreak(self) -> None:
        lexical_first = managed_path(
            self.root,
            hash_hex="0011223344556677",
        )
        lexical_second = managed_path(
            self.root,
            hash_hex="1111223344556677",
        )

        lexical_first.write_bytes(b"x" * 40)
        lexical_second.write_bytes(b"x" * 40)

        os.utime(lexical_first, ns=(7, 7))
        os.utime(lexical_second, ns=(7, 7))

        with self.manager(
            max_bytes=40,
            expected_file_size=40,
        ) as cap:
            self.assertFalse(lexical_first.exists())
            self.assertTrue(lexical_second.exists())
            self.assertEqual(cap.snapshot().accounted_bytes, 40)

    def test_restart_fails_before_ready_if_required_shrink_cannot_unlink(
        self,
    ) -> None:
        oldest = managed_path(
            self.root,
            hash_hex="0011223344556677",
        )
        newest = managed_path(
            self.root,
            hash_hex="1111223344556677",
        )
        oldest.write_bytes(b"x" * 40)
        newest.write_bytes(b"x" * 40)
        os.utime(oldest, ns=(1, 1))
        os.utime(newest, ns=(2, 2))

        real_unlink = os.unlink

        def selective_unlink(path: str) -> None:
            if os.path.abspath(path) == os.path.abspath(oldest):
                raise PermissionError("injected restart shrink failure")
            real_unlink(path)

        with mock.patch(
            "vllm.v1.kv_offload.tiering.fs.capacity.os.unlink",
            side_effect=selective_unlink,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "capacity|shrink|evict",
            ):
                self.manager(
                    max_bytes=40,
                    expected_file_size=40,
                )

        self.assertTrue(oldest.exists())
        self.assertTrue(newest.exists())


if __name__ == "__main__":
    unittest.main()
