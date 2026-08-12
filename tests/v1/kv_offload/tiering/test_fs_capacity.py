# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
import unittest
from pathlib import Path

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
        first = managed_path(self.root)
        first.write_bytes(b"x" * 70)

        with self.manager(max_bytes=100) as cap:
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
                (70, 0),
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


if __name__ == "__main__":
    unittest.main()
