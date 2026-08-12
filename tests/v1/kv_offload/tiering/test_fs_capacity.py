# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from vllm.v1.kv_offload.tiering.fs import io as fs_io
from vllm.v1.kv_offload.tiering.fs import manager as fs_manager
from vllm.v1.kv_offload.tiering.fs.capacity import (
    AdmissionStatus,
    FileSystemCapacityManager,
)


def managed_path(
    root: Path,
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


class FileSystemCapacityManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def manager(
        self,
        max_bytes: int,
        expected_file_size: int | None = None,
    ) -> FileSystemCapacityManager:
        return FileSystemCapacityManager(
            namespace_root=str(self.root),
            max_bytes=max_bytes,
            expected_file_size=expected_file_size,
        )

    def commit_existing(
        self,
        cap: FileSystemCapacityManager,
        path: Path,
        size: int,
    ) -> None:
        result = cap.admit_write(str(path), size)
        self.assertEqual(result.status, AdmissionStatus.RESERVED)
        path.write_bytes(b"x" * size)
        cap.commit_write(result.reservation)

    def test_new_write_reserves_before_commit(self) -> None:
        with self.manager(max_bytes=100) as cap:
            path = managed_path(self.root)
            result = cap.admit_write(str(path), 40)

            self.assertEqual(result.status, AdmissionStatus.RESERVED)
            self.assertEqual(cap.snapshot().reserved_bytes, 40)
            self.assertEqual(cap.snapshot().accounted_bytes, 0)

            path.write_bytes(b"x" * 40)
            cap.commit_write(result.reservation)

            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (40, 0),
            )
            self.assertLessEqual(
                snap.accounted_bytes + snap.reserved_bytes,
                100,
            )

    def test_abort_is_idempotent(self) -> None:
        with self.manager(max_bytes=100) as cap:
            result = cap.admit_write(
                str(managed_path(self.root)),
                40,
            )
            cap.abort_write(result.reservation)
            cap.abort_write(result.reservation)

            self.assertEqual(cap.snapshot().reserved_bytes, 0)

    def test_oversized_does_not_reserve(self) -> None:
        with self.manager(max_bytes=32) as cap:
            result = cap.admit_write(
                str(managed_path(self.root)),
                64,
            )

            self.assertEqual(result.status, AdmissionStatus.OVERSIZED)
            self.assertIsNone(result.reservation)
            self.assertEqual(cap.snapshot().reserved_bytes, 0)
            self.assertEqual(cap.snapshot().accounted_bytes, 0)

    def test_duplicate_inflight_has_one_reservation(self) -> None:
        with self.manager(max_bytes=100) as cap:
            path = str(managed_path(self.root))

            first = cap.admit_write(path, 40)
            second = cap.admit_write(path, 40)

            self.assertEqual(first.status, AdmissionStatus.RESERVED)
            self.assertEqual(
                second.status,
                AdmissionStatus.DUPLICATE_INFLIGHT,
            )
            self.assertIsNone(second.reservation)
            self.assertEqual(cap.snapshot().reserved_bytes, 40)

    def test_committed_same_key_is_already_present(self) -> None:
        with self.manager(max_bytes=100) as cap:
            path = managed_path(self.root)
            self.commit_existing(cap, path, 40)

            result = cap.admit_write(str(path), 40)

            self.assertEqual(
                result.status,
                AdmissionStatus.ALREADY_PRESENT,
            )
            self.assertIsNone(result.reservation)
            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (40, 0),
            )

    def test_same_size_replacement_reserves_full_new_size(self) -> None:
        with self.manager(max_bytes=100) as cap:
            path = managed_path(self.root)
            self.commit_existing(cap, path, 40)

            result = cap.admit_write(
                str(path),
                40,
                replace=True,
            )

            self.assertEqual(result.status, AdmissionStatus.RESERVED)
            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (40, 40),
            )

            path.write_bytes(b"y" * 40)
            cap.commit_write(result.reservation)
            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (40, 0),
            )

    def test_larger_replacement_reserves_full_new_size(self) -> None:
        with self.manager(max_bytes=100) as cap:
            path = managed_path(self.root)
            self.commit_existing(cap, path, 20)

            result = cap.admit_write(
                str(path),
                40,
                replace=True,
            )

            self.assertEqual(result.status, AdmissionStatus.RESERVED)
            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (20, 40),
            )

            path.write_bytes(b"y" * 40)
            cap.commit_write(result.reservation)
            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (40, 0),
            )

    def test_smaller_replacement_reserves_full_new_size(self) -> None:
        with self.manager(max_bytes=100) as cap:
            path = managed_path(self.root)
            self.commit_existing(cap, path, 40)

            result = cap.admit_write(
                str(path),
                20,
                replace=True,
            )

            self.assertEqual(result.status, AdmissionStatus.RESERVED)
            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (40, 20),
            )

            path.write_bytes(b"y" * 20)
            cap.commit_write(result.reservation)
            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (20, 0),
            )

    def test_existing_final_is_recovered_for_replacement(self) -> None:
        path = managed_path(self.root)
        path.write_bytes(b"x" * 30)

        with self.manager(max_bytes=100) as cap:
            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (30, 0),
            )

            result = cap.admit_write(
                str(path),
                40,
                replace=True,
            )
            self.assertEqual(result.status, AdmissionStatus.RESERVED)

            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (30, 40),
            )

            path.write_bytes(b"y" * 40)
            cap.commit_write(result.reservation)

            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (40, 0),
            )

    def test_replacement_abort_preserves_old_accounting(self) -> None:
        path = managed_path(self.root)
        path.write_bytes(b"x" * 30)

        with self.manager(max_bytes=100) as cap:
            result = cap.admit_write(
                str(path),
                40,
                replace=True,
            )
            self.assertEqual(result.status, AdmissionStatus.RESERVED)

            cap.abort_write(result.reservation)

            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (30, 0),
            )

    def test_capacity_reject_does_not_reserve(self) -> None:
        with self.manager(max_bytes=100) as cap:
            first = managed_path(self.root)
            first_result = cap.admit_write(str(first), 70)
            self.assertEqual(
                first_result.status,
                AdmissionStatus.RESERVED,
            )

            second = managed_path(
                self.root,
                hash_hex="1111223344556677",
            )
            result = cap.admit_write(str(second), 40)

            self.assertEqual(result.status, AdmissionStatus.CAPACITY)
            self.assertIsNone(result.reservation)
            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (0, 70),
            )

    def test_new_admission_reaps_orphan_and_releases_charge(self) -> None:
        with self.manager(max_bytes=100) as cap:
            first = managed_path(self.root)
            first_result = cap.admit_write(str(first), 60)
            self.assertEqual(
                first_result.status,
                AdmissionStatus.RESERVED,
            )

            temp_path = f"{first}_7.tmp"
            Path(temp_path).write_bytes(b"x" * 20)
            cap.retain_orphan_temp(
                first_result.reservation,
                temp_path,
            )

            snap = cap.snapshot()
            self.assertEqual(snap.reserved_bytes, 60)
            self.assertEqual(snap.orphan_temp_count, 1)

            second = managed_path(
                self.root,
                hash_hex="1111223344556677",
            )
            second_result = cap.admit_write(str(second), 60)

            self.assertEqual(
                second_result.status,
                AdmissionStatus.RESERVED,
            )
            self.assertFalse(Path(temp_path).exists())

            snap = cap.snapshot()
            self.assertEqual(snap.reserved_bytes, 60)
            self.assertEqual(snap.orphan_temp_count, 0)

    def test_orphan_temp_retains_reservation_charge(self) -> None:
        with self.manager(max_bytes=100) as cap:
            path = managed_path(self.root)
            result = cap.admit_write(str(path), 40)
            temp_path = f"{path}.tmp-orphan"
            Path(temp_path).write_bytes(b"x" * 40)

            cap.retain_orphan_temp(
                result.reservation,
                temp_path,
            )

            snap = cap.snapshot()
            self.assertEqual(snap.accounted_bytes, 0)
            self.assertEqual(snap.reserved_bytes, 40)
            self.assertEqual(snap.orphan_temp_count, 1)


    def test_contains_many_reports_only_committed_entries(self) -> None:
        present = managed_path(self.root)
        present.write_bytes(b"x" * 20)
        missing = managed_path(
            self.root,
            hash_hex="1111223344556677",
        )

        with self.manager(max_bytes=100) as cap:
            self.assertTrue(cap.contains(str(present)))
            self.assertFalse(cap.contains(str(missing)))
            self.assertEqual(
                cap.contains_many([str(present), str(missing)]),
                [True, False],
            )

    def test_touch_changes_lru_victim(self) -> None:
        first = managed_path(
            self.root,
            hash_hex="0011223344556677",
        )
        second = managed_path(
            self.root,
            hash_hex="1111223344556677",
        )
        third = managed_path(
            self.root,
            hash_hex="2221223344556677",
        )

        for path in (first, second, third):
            path.write_bytes(b"x" * 30)

        os.utime(first, ns=(1, 1))
        os.utime(second, ns=(2, 2))
        os.utime(third, ns=(3, 3))

        with self.manager(max_bytes=90) as cap:
            # first starts as oldest; touching it must make second oldest.
            cap.touch([str(first)])

            incoming = managed_path(
                self.root,
                hash_hex="3331223344556677",
            )
            result = cap.admit_write(str(incoming), 30)

            self.assertEqual(result.status, AdmissionStatus.RESERVED)
            self.assertTrue(first.exists())
            self.assertFalse(second.exists())
            self.assertTrue(third.exists())

            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (60, 30),
            )

    def test_pinned_only_victim_blocks_eviction_until_release(self) -> None:
        victim = managed_path(self.root)
        victim.write_bytes(b"x" * 80)

        with self.manager(max_bytes=80) as cap:
            pin = cap.pin_for_read(str(victim))
            self.assertIsNotNone(pin)

            incoming = managed_path(
                self.root,
                hash_hex="1111223344556677",
            )
            blocked = cap.admit_write(str(incoming), 40)

            self.assertEqual(blocked.status, AdmissionStatus.CAPACITY)
            self.assertTrue(victim.exists())

            cap.release_read(pin)

            admitted = cap.admit_write(str(incoming), 40)
            self.assertEqual(admitted.status, AdmissionStatus.RESERVED)
            self.assertFalse(victim.exists())

    def test_release_read_with_invalidate_unlinks_and_unaccounts(self) -> None:
        victim = managed_path(self.root)
        victim.write_bytes(b"x" * 40)

        with self.manager(max_bytes=100) as cap:
            pin = cap.pin_for_read(str(victim))
            self.assertIsNotNone(pin)

            cap.release_read(pin, invalidate=True)

            self.assertFalse(cap.contains(str(victim)))
            self.assertFalse(victim.exists())
            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (0, 0),
            )

    def test_invalidate_unlink_failure_keeps_entry_accounted_and_invisible(
        self,
    ) -> None:
        victim = managed_path(self.root)
        victim.write_bytes(b"x" * 40)

        with self.manager(max_bytes=40) as cap:
            pin = cap.pin_for_read(str(victim))
            self.assertIsNotNone(pin)

            with self.assertLogs(
                "vllm.v1.kv_offload.tiering.fs.capacity",
                level="WARNING",
            ) as logs:
                with mock.patch(
                    "vllm.v1.kv_offload.tiering.fs.capacity.os.unlink",
                    side_effect=PermissionError(
                        "injected invalid cleanup failure"
                    ),
                ):
                    cap.release_read(pin, invalidate=True)

            self.assertTrue(
                any(
                    "failed to remove invalid filesystem KV cache entry"
                    in message
                    for message in logs.output
                )
            )

            # INVALID entries remain conservatively accounted, but can no
            # longer be observed or pinned as cache hits.
            self.assertFalse(cap.contains(str(victim)))
            self.assertIsNone(cap.pin_for_read(str(victim)))
            self.assertTrue(victim.exists())

            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (40, 0),
            )

            incoming = managed_path(
                self.root,
                hash_hex="1111223344556677",
            )
            result = cap.admit_write(str(incoming), 1)
            self.assertEqual(result.status, AdmissionStatus.CAPACITY)
            self.assertIsNone(result.reservation)

    def test_failed_oldest_unlink_tries_later_victim(self) -> None:
        oldest = managed_path(
            self.root,
            hash_hex="0011223344556677",
        )
        later = managed_path(
            self.root,
            hash_hex="1111223344556677",
        )
        oldest.write_bytes(b"x" * 40)
        later.write_bytes(b"x" * 40)
        os.utime(oldest, ns=(1, 1))
        os.utime(later, ns=(2, 2))

        with self.manager(max_bytes=80) as cap:
            incoming = managed_path(
                self.root,
                hash_hex="2221223344556677",
            )
            real_unlink = os.unlink

            def selective_unlink(path: str) -> None:
                if os.path.abspath(path) == os.path.abspath(oldest):
                    raise PermissionError("injected eviction failure")
                real_unlink(path)

            with self.assertLogs(
                "vllm.v1.kv_offload.tiering.fs.capacity",
                level="WARNING",
            ) as logs:
                with mock.patch(
                    "vllm.v1.kv_offload.tiering.fs.capacity.os.unlink",
                    side_effect=selective_unlink,
                ):
                    result = cap.admit_write(str(incoming), 40)

            self.assertTrue(
                any(
                    "failed to evict filesystem KV cache entry" in message
                    for message in logs.output
                )
            )

            self.assertEqual(result.status, AdmissionStatus.RESERVED)
            self.assertTrue(oldest.exists())
            self.assertFalse(later.exists())

            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (40, 40),
            )

    def test_all_victims_unavailable_returns_capacity(self) -> None:
        victim = managed_path(self.root)
        victim.write_bytes(b"x" * 80)

        with self.manager(max_bytes=80) as cap:
            incoming = managed_path(
                self.root,
                hash_hex="1111223344556677",
            )

            with self.assertLogs(
                "vllm.v1.kv_offload.tiering.fs.capacity",
                level="WARNING",
            ) as logs:
                with mock.patch(
                    "vllm.v1.kv_offload.tiering.fs.capacity.os.unlink",
                    side_effect=PermissionError(
                        "injected eviction failure"
                    ),
                ):
                    result = cap.admit_write(str(incoming), 40)

            self.assertTrue(
                any(
                    "failed to evict filesystem KV cache entry" in message
                    for message in logs.output
                )
            )

            self.assertEqual(result.status, AdmissionStatus.CAPACITY)
            self.assertIsNone(result.reservation)
            self.assertTrue(victim.exists())

            snap = cap.snapshot()
            self.assertEqual(
                (snap.accounted_bytes, snap.reserved_bytes),
                (80, 0),
            )

    def test_concurrent_admission_never_exceeds_capacity(self) -> None:
        with self.manager(max_bytes=100) as cap:
            barrier = threading.Barrier(3)
            statuses: list[AdmissionStatus] = []
            observed_totals: list[int] = []
            result_lock = threading.Lock()

            paths = (
                managed_path(
                    self.root,
                    hash_hex="0011223344556677",
                ),
                managed_path(
                    self.root,
                    hash_hex="1111223344556677",
                ),
            )

            def worker(path: Path) -> None:
                barrier.wait()
                result = cap.admit_write(str(path), 60)
                snap = cap.snapshot()
                with result_lock:
                    statuses.append(result.status)
                    observed_totals.append(
                        snap.accounted_bytes + snap.reserved_bytes
                    )

            threads = [
                threading.Thread(target=worker, args=(path,))
                for path in paths
            ]
            for thread in threads:
                thread.start()

            barrier.wait()

            for thread in threads:
                thread.join()

            self.assertEqual(
                sorted(status.value for status in statuses),
                sorted(
                    [
                        AdmissionStatus.RESERVED.value,
                        AdmissionStatus.CAPACITY.value,
                    ]
                ),
            )
            self.assertTrue(observed_totals)
            self.assertTrue(
                all(total <= 100 for total in observed_totals)
            )


    def test_raw_load_failure_does_not_delete_final(self) -> None:
        final_path = managed_path(self.root)
        final_path.write_bytes(b"x" * 4)
        target = bytearray(8)

        with mock.patch.object(
            fs_io.os,
            "open",
            side_effect=FileNotFoundError("injected read failure"),
        ):
            with mock.patch.object(fs_io.os, "remove") as remove:
                with self.assertRaises(FileNotFoundError):
                    fs_io.load_block(
                        str(final_path),
                        memoryview(target),
                        0,
                        4,
                    )

        remove.assert_not_called()
        self.assertTrue(final_path.exists())

    def test_raw_store_uses_caller_supplied_temp_path(self) -> None:
        final_path = managed_path(self.root)
        temp_path = Path(f"{final_path}_123.tmp")
        payload = bytearray(b"abcdefgh")

        opened_paths: list[str] = []

        def fake_open(path: str, flags: int, mode: int = 0o777) -> int:
            opened_paths.append(path)
            return 17

        with mock.patch.object(
            fs_io.os,
            "open",
            side_effect=fake_open,
        ):
            with mock.patch.object(
                fs_io.os,
                "write",
                return_value=4,
            ):
                with mock.patch.object(fs_io.os, "close"):
                    with mock.patch.object(fs_io.os, "replace") as replace:
                        with mock.patch.object(
                            fs_io,
                            "_ensure_dirs",
                        ) as ensure_dirs:
                            with mock.patch.object(
                                fs_io.os.path,
                                "exists",
                                side_effect=AssertionError(
                                    "raw store must not check "
                                    "destination existence"
                                ),
                            ):
                                fs_io.store_block(
                                    str(final_path),
                                    str(temp_path),
                                    memoryview(payload),
                                    0,
                                    4,
                                )

        ensure_dirs.assert_called_once_with(str(final_path))

        self.assertEqual(opened_paths, [str(temp_path)])
        replace.assert_called_once_with(
            str(temp_path),
            str(final_path),
        )

    def test_raw_store_failure_leaves_temp_cleanup_to_manager(self) -> None:
        final_path = managed_path(self.root)
        temp_path = Path(f"{final_path}_456.tmp")
        payload = bytearray(b"abcdefgh")

        with mock.patch.object(fs_io.os, "open", return_value=17):
            with mock.patch.object(
                fs_io.os,
                "write",
                side_effect=OSError("injected write failure"),
            ):
                with mock.patch.object(fs_io.os, "close"):
                    with mock.patch.object(fs_io.os, "remove") as remove:
                        with self.assertRaisesRegex(
                            OSError,
                            "injected write failure",
                        ):
                            fs_io.store_block(
                                str(final_path),
                                str(temp_path),
                                memoryview(payload),
                                0,
                                4,
                            )

        # Raw I/O must not make reservation/accounting cleanup decisions.
        remove.assert_not_called()

    def test_make_temp_path_matches_restart_recognized_shape(self) -> None:
        final_path = managed_path(self.root)

        temp_path = fs_io.make_temp_path(str(final_path))

        self.assertTrue(temp_path.startswith(f"{final_path}_"))
        self.assertTrue(temp_path.endswith(".tmp"))
        suffix = temp_path[len(str(final_path)) + 1 : -4]
        self.assertTrue(suffix.isdigit())

    def test_store_cleanup_failure_retains_orphan_reservation(
        self,
    ) -> None:
        manager = object.__new__(
            fs_manager.FileSystemTierManager
        )
        manager._block_size = 4
        manager._primary_kv_view = memoryview(
            bytearray(b"abcdefgh")
        )
        manager._capacity = mock.MagicMock()

        reservation = object()
        admission = mock.MagicMock()
        admission.status = AdmissionStatus.RESERVED
        admission.reservation = reservation
        manager._capacity.admit_write.return_value = admission

        final_path = str(managed_path(self.root))
        temp_path = f"{final_path}_999.tmp"

        with mock.patch.object(
            fs_manager,
            "make_temp_path",
            return_value=temp_path,
        ):
            with mock.patch.object(
                fs_manager,
                "store_block",
                side_effect=OSError("injected store failure"),
            ):
                with mock.patch.object(
                    fs_manager.os,
                    "unlink",
                    side_effect=PermissionError(
                        "injected temp cleanup failure"
                    ),
                ):
                    with self.assertRaisesRegex(
                        OSError,
                        "injected store failure",
                    ):
                        manager._store_one(
                            final_path,
                            0,
                            None,
                            0,
                        )

        manager._capacity.retain_orphan_temp.assert_called_once_with(
            reservation,
            temp_path,
        )
        manager._capacity.abort_write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
