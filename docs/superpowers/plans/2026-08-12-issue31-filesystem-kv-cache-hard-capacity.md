# Issue #31 Filesystem KV Cache Hard Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a required user-bounded `max_bytes` hard ceiling to the filesystem KV-cache tier, with concurrency-safe reservation, deterministic LRU eviction, crash/restart recovery, single-owner namespace safety, accurate events/metrics, and real-filesystem validation without changing inference correctness.

**Architecture:** Introduce a focused `FileSystemCapacityManager` in `vllm/v1/kv_offload/tiering/fs/capacity.py`. Startup acquires exclusive namespace ownership, scans/reconciles the mapper-owned data namespace, and synchronously shrinks it to the configured bound. Runtime uses a short metadata lock plus a separate admission lock: capacity decisions and victim claims are serialized, while actual KV payload reads/writes remain concurrent after reservations are granted. `FileSystemTierManager` remains the orchestration layer and delegates all cache-owned final-file mutation/accounting to the capacity manager.

**Tech Stack:** Python 3.11, stdlib `dataclasses`/`enum`/`fcntl`/`os`/`shutil`/`threading`, existing vLLM `FileMapper`, `DualQueueThreadPool`, `AsyncLookupManager`, `OffloadingConnectorStats`, PyTorch/memoryview filesystem-tier path, stdlib `unittest` for new locally runnable RED/GREEN coverage, existing repository pytest/pre-commit/GitHub Actions as authoritative CI.

## Global Constraints

- Source of truth precedence remains: live GitHub Issue/PR/branch/commit metadata, current repository source, structured artifacts, engineering docs, launch context, old chat history.
- Approved design: `docs/superpowers/specs/2026-08-12-filesystem-kv-cache-hard-capacity-design.md` on branch `agent/issue31-fs-hard-capacity`.
- Product contract: explicit filesystem tiers are bounded. An explicit `type: "fs"` tier must provide a positive integer `max_bytes`; no compatibility fallback may silently create an unbounded filesystem tier.
- Core invariant from READY onward: `accounted_bytes + reserved_bytes <= max_bytes`.
- `reserved_bytes` covers the full incoming payload before the first payload write, so temp-file peak bytes are bounded.
- Conservative cleanup clarification derived from the approved invariant: if a failed write leaves a temp file that cannot be confirmed deleted, **do not release its reservation**. Keep those bytes conservatively in `reserved_bytes`, track the temp as an orphan reservation, retry cleanup on later admission/shutdown, and let restart recovery clean/fail-fast if it survives process exit.
- `max_bytes` is logical cache-payload capacity based on cache-owned file `st_size` plus reservations. Do not claim exact control of inode/journal/allocation-block overhead.
- `max_bytes` must not enter the `FileMapper` namespace hash.
- Logical quota is independent of current filesystem free space. Free-space information is diagnostic only; `ENOSPC`/`EDQUOT` is an I/O failure, not a logical capacity rejection.
- Runtime cache pressure is best-effort degradation. Oversize/no-evictable-capacity skips do not fail inference and do not emit false `BlockStored` events.
- Real filesystem I/O failure remains a failed async job. Partial real I/O failure keeps the existing conservative no-stored-event job behavior.
- Normal same-key runtime stores remain idempotent. The low-level capacity API supports full-new-size reservation for replacement accounting tests.
- Runtime LRU is in-memory only. Restart LRU cold-start order is `(st_mtime_ns, normalized_relative_path)`.
- One bounded manager owns one mapper data namespace at a time using a non-blocking advisory lock. Do not implement a cross-process quota coordinator.
- Do not implement #16 active restore/recompute, #19 cost-aware admission/eviction, #18 multi-device placement, adaptive capacity recommendations, CPU redesign, or a persistent metadata DB/WAL.
- Do not call container-local/overlay filesystem evidence “NVMe” without physical provenance.
- Scheduler-facing tier methods must remain lightweight/non-blocking. No large payload read/write runs while capacity locks are held.
- GitHub is the authoritative remote/write path. The Pod is for build, focused tests, and filesystem validation. Never `git push` from the Pod.
- At execution time, use `/code/vllm-issue31` only as the clean mirror source; do not touch the old `/code/vllm` checkout.
- Invoke `superpowers:using-git-worktrees` before execution-time Pod repository changes.
- Never use `git clean -fd`, `git reset --hard`, or destructive checkout/reset commands. Never delete formal evidence under `/code/results/cache`.
- Run one safe shell command block at a time. Do not use `set -e` in operator-facing Pod command blocks.
- Do not assume pytest exists on the Pod and do not install it solely for #31. New core RED/GREEN tests must be runnable with stdlib `unittest`; pytest-only compatibility tests are additionally covered by authoritative GitHub CI when available.
- The implementation PR is Draft by default and must use `Closes #31`. Green CI is necessary but never merge authorization.
- Do not start #16 before #31 is merged and closed.

---

## File Structure

Create:

- `vllm/v1/kv_offload/tiering/fs/capacity.py` — namespace ownership, startup recovery, committed/reserved accounting, LRU, admission/eviction, read pins, orphan-temp reservations, snapshots/counters.
- `tests/v1/kv_offload/tiering/test_fs_capacity.py` — stdlib-unittest-compatible focused capacity/recovery/concurrency tests using real small filesystem files and no GPU.
- `benchmarks/cache/tests/test_issue31_fs_capacity_config.py` — stdlib tests for benchmark `max_bytes` validation and server-config forwarding.
- `benchmarks/cache/issue31_fs_capacity_smoke.py` — deterministic real-filesystem hard-cap smoke runner using the production capacity manager and real temp/final files.
- `benchmarks/cache/tests/test_issue31_fs_capacity_smoke.py` — smoke-runner JSON-contract test.
- `docs/engineering/validation/2026-08-12-issue31-filesystem-hard-capacity-validation.json` — structured final local/smoke evidence.
- `docs/engineering/validation/2026-08-12-issue31-filesystem-hard-capacity-validation.md` — human-readable validation report.

Modify:

- `vllm/v1/kv_offload/tiering/fs/manager.py` — require `max_bytes`; instantiate capacity manager before workers; bounded lookup; worker-side store admission; load pin/invalidation; touch; per-key committed event bookkeeping; metrics; shutdown ordering.
- `vllm/v1/kv_offload/tiering/fs/io.py` — expose temp-path creation; raw store leaves final ownership/accounting to manager; raw load no longer deletes final files.
- `vllm/v1/kv_offload/file_mapper.py` — add `get_data_dir_path()` only; do not add capacity policy/hash inputs.
- `vllm/v1/kv_offload/tiering/base.py` — give secondary tier instances a default deterministic runtime `instance_id` equal to `tier_type`.
- `vllm/v1/kv_offload/tiering/factory.py` — optional `instance_id` argument; set it after tier construction without changing subclass constructor signatures.
- `vllm/v1/kv_offload/tiering/spec.py` — pass deterministic `<type>:<config-index>` instance IDs when constructing configured secondary tiers.
- `tests/v1/kv_offload/tiering/test_fs_tier.py` — add `max_bytes` to existing direct constructions/fixtures; replace raw-path lookup dispatch expectation with capacity-state lookup behavior; add integration/event assertions.
- `tests/v1/kv_offload/tiering/test_factory.py` — verify optional instance identity while preserving existing factory callers.
- `tests/v1/kv_offload/tiering/test_shadow_cost_spec.py` — adjust the factory stub/signature for the new optional instance identity and verify config-index identity does not alter cost-model tier keys.
- `benchmarks/cache/config.py` — benchmark high-level filesystem config gains `max_bytes`; require it only when `filesystem.enabled` is true.
- `benchmarks/cache/scenarios.py` — forward benchmark filesystem `max_bytes` into the runtime FS secondary-tier config.
- `benchmarks/cache/tests/test_config.py` and `benchmarks/cache/tests/test_scenarios.py` — preserve existing pytest coverage with new capacity field/forwarding assertions.
- `benchmarks/cache/configs/example-7b.yaml`
- `benchmarks/cache/configs/example-70b.yaml`
- `benchmarks/cache/configs/example-397b.yaml`
- `benchmarks/cache/configs/local-crossover.yaml`
- `benchmarks/cache/configs/issue15-7b-load-sentinel-fs.yaml`
- `benchmarks/cache/configs/issue15-14b-formal-fs.yaml` — add explicit non-binding legacy-benchmark `max_bytes` so prior benchmark intent is not accidentally turned into a capacity-pressure experiment.
- `docs/engineering/CURRENT_STATE.md` — update only after focused verification and real-filesystem evidence exist; keep #16 blocked until merge.

Do not modify `DualQueueThreadPool` for #31 unless a new failing test proves a capacity-specific need. Reservations are deliberately acquired only when worker callables begin, so queued tasks cancelled by the existing shutdown path never own capacity.

---

### Task 1: Create and verify an isolated execution worktree

**Files:** No repository changes.

**Interfaces:**

- Consumes: live GitHub `main`, live `agent/issue31-fs-hard-capacity`, and the Gitee mirror refs visible from `/code/vllm-issue31`.
- Produces: `/code/vllm-issue31-worktrees/hard-capacity` at the exact live GitHub implementation-branch head containing this plan.

- [ ] **Step 1: Invoke `superpowers:using-git-worktrees`.**

Do this before any execution-time Pod checkout/worktree operation.

- [ ] **Step 2: Re-read live GitHub refs.**

Record current `main` and `agent/issue31-fs-hard-capacity`. If `main` moved beyond the design base `c4d9fce61ec5a8eadc24dab8698eca7705d005bf`, compare the new commits before implementation. Do not silently code against stale source if the changed files overlap #31.

- [ ] **Step 3: Inspect the clean Gitee mirror source without changing it.**

```bash
cd /code/vllm-issue31 || exit 1
printf 'branch='; git branch --show-current
printf 'head='; git rev-parse HEAD
printf 'origin='; git remote get-url origin
git status --short --branch
```

Expected: clean checkout; remote remains `https://gitee.com/zh19990906/vllm-zhangheng.git`.

- [ ] **Step 4: Fetch only the relevant mirror refs.**

```bash
cd /code/vllm-issue31 || exit 1
git fetch origin main agent/issue31-fs-hard-capacity
printf 'origin/main='; git rev-parse origin/main
printf 'origin/agent='; git rev-parse origin/agent/issue31-fs-hard-capacity
```

Compare both SHAs to live GitHub. If Gitee does not yet expose the exact implementation branch head, stop the Pod execution path and report a mirror-sync mismatch rather than building on a different tree.

- [ ] **Step 5: Create the isolated worktree.**

```bash
cd /code/vllm-issue31 || exit 1
mkdir -p /code/vllm-issue31-worktrees
git worktree add -b local/issue31-fs-hard-capacity \
  /code/vllm-issue31-worktrees/hard-capacity \
  origin/agent/issue31-fs-hard-capacity
cd /code/vllm-issue31-worktrees/hard-capacity || exit 1
printf 'head='; git rev-parse HEAD
git status --short --branch
```

Expected: clean isolated worktree at the exact verified branch head.

- [ ] **Step 6: Probe test tooling; do not install pytest.**

```bash
cd /code/vllm-issue31-worktrees/hard-capacity || exit 1
python - <<'PY'
import importlib.util
for name in ('torch', 'vllm', 'pytest'):
    print(name, bool(importlib.util.find_spec(name)))
PY
```

Record the result. All new capacity tests below must remain runnable via `unittest` regardless of pytest availability.

---

### Task 2: Make filesystem capacity explicit in runtime and benchmark configuration

**Files:**

- Modify: `vllm/v1/kv_offload/tiering/fs/manager.py`
- Modify: `vllm/v1/kv_offload/file_mapper.py`
- Modify: `tests/v1/kv_offload/tiering/test_fs_tier.py`
- Create: `benchmarks/cache/tests/test_issue31_fs_capacity_config.py`
- Modify: `benchmarks/cache/config.py`
- Modify: `benchmarks/cache/scenarios.py`
- Modify: `benchmarks/cache/tests/test_config.py`
- Modify: `benchmarks/cache/tests/test_scenarios.py`
- Modify: the six enabled filesystem YAML configs listed in File Structure.

**Interfaces:**

- Consumes: existing tier-specific config forwarding from `SecondaryTierFactory` and benchmark `FilesystemCacheConfig`.
- Produces: `FileSystemTierManager(..., max_bytes: int, ...)`, `FileMapper.get_data_dir_path() -> str`, and benchmark runtime config containing `secondary_tiers[0]["max_bytes"]`.

- [ ] **Step 1: Invoke `superpowers:test-driven-development` before the first production-code edit.**

- [ ] **Step 2: Write RED stdlib config tests.**

Create `benchmarks/cache/tests/test_issue31_fs_capacity_config.py` with these exact core assertions:

```python
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from benchmarks.cache.config import FilesystemCacheConfig
from benchmarks.cache.scenarios import CacheMode, build_execution_cases, build_server_command


class Issue31FilesystemCapacityConfigTests(unittest.TestCase):
    def test_enabled_filesystem_requires_positive_max_bytes(self) -> None:
        with self.assertRaises(ValidationError):
            FilesystemCacheConfig(enabled=True, root_dir=Path('/tmp/cache'))
        with self.assertRaises(ValidationError):
            FilesystemCacheConfig(
                enabled=True, root_dir=Path('/tmp/cache'), max_bytes=0
            )

    def test_disabled_filesystem_may_omit_max_bytes(self) -> None:
        cfg = FilesystemCacheConfig(enabled=False, root_dir=Path('/tmp/cache'))
        self.assertIsNone(cfg.max_bytes)

    def test_tiered_server_config_forwards_max_bytes(self) -> None:
        # Reuse the checked-in example config so the test covers real YAML plumbing.
        from benchmarks.cache.config import load_suite_config
        config = load_suite_config(Path('benchmarks/cache/configs/example-7b.yaml'))
        with tempfile.TemporaryDirectory() as td:
            case = next(
                c
                for c in build_execution_cases(config, Path(td))
                if c.cache_mode is CacheMode.TIERED_FS
                and c.workload_kind == 'cold-unique'
            )
        command = build_server_command(case, config)
        payload = json.loads(command[command.index('--kv-transfer-config') + 1])
        fs = payload['kv_connector_extra_config']['secondary_tiers'][0]
        self.assertEqual(fs['max_bytes'], config.cache.filesystem.max_bytes)
```

- [ ] **Step 3: Run RED.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_issue31_fs_capacity_config.py' -v
```

Expected: FAIL because benchmark `FilesystemCacheConfig` has no `max_bytes` and runtime forwarding omits it.

- [ ] **Step 4: Add the benchmark high-level field and forwarding.**

Implement:

```python
class FilesystemCacheConfig(StrictModel):
    enabled: bool = True
    root_dir: Path
    max_bytes: PositiveInt | None = None
    read_threads: PositiveInt = 32
    write_threads: PositiveInt = 16

    @model_validator(mode='after')
    def require_max_when_enabled(self) -> FilesystemCacheConfig:
        if self.enabled and self.max_bytes is None:
            raise ValueError('filesystem.max_bytes is required when enabled')
        return self
```

And in `_offloading_config()`:

```python
assert config.cache.filesystem.max_bytes is not None
...
'max_bytes': config.cache.filesystem.max_bytes,
```

- [ ] **Step 5: Add explicit non-binding capacity to legacy enabled benchmark configs.**

Use exactly `1099511627776` (1 TiB) for the six existing enabled filesystem benchmark configs. This keeps those historical benchmark configs bounded without intentionally converting them into new capacity-pressure experiments. Do not add `max_bytes` to disabled CPU-only filesystem sections unless the schema test requires it; disabled high-level filesystem config is allowed to omit the value.

Example:

```yaml
filesystem:
  enabled: true
  root_dir: /tmp/vllm-kv-cache
  max_bytes: 1099511627776
  read_threads: 32
  write_threads: 16
```

- [ ] **Step 6: Add runtime constructor RED coverage and update old constructions.**

In `tests/v1/kv_offload/tiering/test_fs_tier.py`, define:

```python
_DEFAULT_MAX_BYTES = 8 * _BLOCK_ELEMENTS * torch.tensor([], dtype=_DTYPE).element_size()
```

Pass `max_bytes=_DEFAULT_MAX_BYTES` to every valid direct `FileSystemTierManager` construction and add:

```python
@pytest.mark.parametrize('value', [0, -1, 1.5, '1024', True, None])
def test_invalid_max_bytes_raises_at_construction(tmp_path, value):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    kwargs = {} if value is None else {'max_bytes': value}
    with pytest.raises((TypeError, ValueError), match='max_bytes'):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=memoryview(tensor.numpy()),
            tier_type='fs',
            root_dir=str(tmp_path),
            **kwargs,
        )
```

Also add one test that a small positive `max_bytes < _block_size` constructs successfully.

- [ ] **Step 7: Implement strict manager validation and mapper data-root accessor.**

In `FileSystemTierManager.__init__`:

```python
if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
    raise ValueError('max_bytes must be a positive integer number of bytes')
self.max_bytes = max_bytes
```

In `FileMapper`:

```python
def get_data_dir_path(self) -> str:
    return f'{self.base_path}_r{self.rank}'
```

Do not add `max_bytes` to `self.fields` or `_compute_base_path()`.

- [ ] **Step 8: Run GREEN for locally runnable config coverage and parse all benchmark YAMLs.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_issue31_fs_capacity_config.py' -v
python - <<'PY'
from pathlib import Path
from benchmarks.cache.config import load_suite_config
for path in sorted(Path('benchmarks/cache/configs').glob('*.yaml')):
    load_suite_config(path)
    print('OK', path)
PY
```

Expected: all checked-in configs parse.

- [ ] **Step 9: Run pytest compatibility tests if pytest is available; otherwise defer them explicitly to GitHub CI.**

```bash
python - <<'PY'
import importlib.util, subprocess, sys
if importlib.util.find_spec('pytest'):
    raise SystemExit(subprocess.call([
        sys.executable, '-m', 'pytest',
        'tests/v1/kv_offload/tiering/test_fs_tier.py',
        'benchmarks/cache/tests/test_config.py',
        'benchmarks/cache/tests/test_scenarios.py',
        '-q',
    ]))
print('pytest unavailable: recorded for authoritative GitHub CI')
PY
```

- [ ] **Step 10: Checkpoint commit.**

Commit message: `feat: require filesystem kv cache capacity`

Do not push from the Pod; publish the reviewed checkpoint through the GitHub write path when execution reaches the publication checkpoint.

---

### Task 3: Implement the core capacity ledger, reservation lifecycle, and replacement accounting

**Files:**

- Create: `vllm/v1/kv_offload/tiering/fs/capacity.py`
- Create: `tests/v1/kv_offload/tiering/test_fs_capacity.py`

**Interfaces:**

- Produces:
  - `EntryState(COMMITTED, EVICTING, INVALID)`
  - `AdmissionStatus(RESERVED, ALREADY_PRESENT, DUPLICATE_INFLIGHT, OVERSIZED, CAPACITY)`
  - `WriteReservation`
  - `ReadPin`
  - `CapacitySnapshot`
  - `FileSystemCapacityManager(namespace_root: str, max_bytes: int, expected_file_size: int | None)`
  - `admit_write(path: str, size: int, *, replace: bool = False) -> AdmissionResult`
  - `commit_write(reservation: WriteReservation, final_size: int | None = None) -> None`
  - `abort_write(reservation: WriteReservation) -> None`
  - `retain_orphan_temp(reservation: WriteReservation, temp_path: str) -> None`
  - `snapshot() -> CapacitySnapshot`
- Later tasks consume these exact names.

- [ ] **Step 1: Write RED accounting tests in `test_fs_capacity.py`.**

Use stdlib `unittest`, `tempfile.TemporaryDirectory`, and a helper that creates mapper-shaped paths:

```python
from pathlib import Path


def managed_path(root: Path, hash_hex: str = '0011223344556677', group: int = 0) -> Path:
    path = root / hash_hex[:3] / f'{hash_hex[3:5]}_g{group}' / f'{hash_hex}.bin'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
```

Add exact tests for:

```python
def test_new_write_reserves_before_commit(self):
    with self.manager(max_bytes=100) as cap:
        path = str(managed_path(self.root))
        result = cap.admit_write(path, 40)
        self.assertEqual(result.status, AdmissionStatus.RESERVED)
        self.assertEqual(cap.snapshot().reserved_bytes, 40)
        self.assertEqual(cap.snapshot().accounted_bytes, 0)
        Path(path).write_bytes(b'x' * 40)
        cap.commit_write(result.reservation)
        snap = cap.snapshot()
        self.assertEqual((snap.accounted_bytes, snap.reserved_bytes), (40, 0))
        self.assertLessEqual(snap.accounted_bytes + snap.reserved_bytes, 100)


def test_abort_is_idempotent(self):
    with self.manager(max_bytes=100) as cap:
        result = cap.admit_write(str(managed_path(self.root)), 40)
        cap.abort_write(result.reservation)
        cap.abort_write(result.reservation)
        self.assertEqual(cap.snapshot().reserved_bytes, 0)


def test_oversized_does_not_reserve(self):
    with self.manager(max_bytes=32) as cap:
        result = cap.admit_write(str(managed_path(self.root)), 64)
        self.assertEqual(result.status, AdmissionStatus.OVERSIZED)
        self.assertEqual(cap.snapshot().reserved_bytes, 0)


def test_duplicate_inflight_has_one_reservation(self):
    with self.manager(max_bytes=100) as cap:
        path = str(managed_path(self.root))
        first = cap.admit_write(path, 40)
        second = cap.admit_write(path, 40)
        self.assertEqual(first.status, AdmissionStatus.RESERVED)
        self.assertEqual(second.status, AdmissionStatus.DUPLICATE_INFLIGHT)
        self.assertEqual(cap.snapshot().reserved_bytes, 40)
```

Add replacement tests using a manager with `expected_file_size=None`: recover an existing 30-byte final, then assert same/larger/smaller replacement reserves the **full new size**, keeps the old 30 bytes accounted until commit, computes `old -> new` accounting on success, and preserves the old accounting on abort.

- [ ] **Step 2: Run RED.**

```bash
python -m unittest discover -s tests/v1/kv_offload/tiering \
  -p 'test_fs_capacity.py' -v
```

Expected: import failure because `fs.capacity` does not exist.

- [ ] **Step 3: Implement the data model and locks.**

Start with:

```python
class EntryState(Enum):
    COMMITTED = 'committed'
    EVICTING = 'evicting'
    INVALID = 'invalid'


class AdmissionStatus(Enum):
    RESERVED = 'reserved'
    ALREADY_PRESENT = 'already_present'
    DUPLICATE_INFLIGHT = 'duplicate_inflight'
    OVERSIZED = 'oversized'
    CAPACITY = 'capacity'


@dataclass(slots=True)
class EntryRecord:
    path: str
    size: int
    recency: int
    readers: int
    state: EntryState
    generation: int


@dataclass(slots=True)
class WriteReservation:
    token: int
    path: str
    size: int
    replaced_generation: int | None
    active: bool = True


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    status: AdmissionStatus
    reservation: WriteReservation | None = None
```

`FileSystemCapacityManager` owns:

```python
self._metadata_lock = threading.Lock()
self._admission_lock = threading.Lock()
self._entries: dict[str, EntryRecord] = {}
self._pending_writes: dict[str, WriteReservation] = {}
self._orphan_temps: dict[str, WriteReservation] = {}
self._accounted_bytes = 0
self._reserved_bytes = 0
self._clock = 0
self._generation = 0
self._reservation_token = 0
```

- [ ] **Step 4: Implement admission/commit/abort with exact invariant checks.**

Admission order under `admission_lock`:

```python
# reap known orphan temps first (implemented in Step 6)
# inspect existing/pending state under metadata_lock
# existing valid + replace=False -> ALREADY_PRESENT + touch
# pending writer -> DUPLICATE_INFLIGHT
# size > max -> increment oversize skip counter and return OVERSIZED
# if enough capacity -> install full-size reservation atomically
# otherwise return CAPACITY until Task 4 adds victim eviction
```

Every transition that changes bytes calls an internal assertion equivalent to:

```python
assert self._accounted_bytes >= 0
assert self._reserved_bytes >= 0
assert self._accounted_bytes + self._reserved_bytes <= self.max_bytes
```

Replacement admission excludes the replaced path from future victim selection, keeps its old `EntryRecord` accounted, and reserves the entire new size.

- [ ] **Step 5: Add RED orphan-temp reservation retention tests.**

```python
def test_failed_temp_cleanup_keeps_bytes_reserved(self):
    with self.manager(max_bytes=100) as cap:
        final = str(managed_path(self.root))
        result = cap.admit_write(final, 40)
        tmp = final + '_7.tmp'
        Path(tmp).write_bytes(b'x' * 20)
        cap.retain_orphan_temp(result.reservation, tmp)
        snap = cap.snapshot()
        self.assertEqual(snap.reserved_bytes, 40)
        self.assertEqual(snap.orphan_temp_count, 1)
```

This is the conservative rule required to avoid under-accounting a temp file whose deletion failed.

- [ ] **Step 6: Implement orphan-temp retention and reaping.**

`retain_orphan_temp()` removes the reservation from `_pending_writes` but leaves its full size in `_reserved_bytes` and records `tmp_path -> reservation`. Before every new admission, `_reap_orphan_temps()` attempts `os.unlink()` outside `metadata_lock` while `admission_lock` is held. Successful deletion/`ENOENT` releases the reservation exactly once; other errors keep it reserved and log the failure.

- [ ] **Step 7: Run GREEN.**

```bash
python -m unittest discover -s tests/v1/kv_offload/tiering \
  -p 'test_fs_capacity.py' -v
```

Expected: all Task 3 accounting/replacement/orphan tests pass.

- [ ] **Step 8: Checkpoint commit.**

Commit message: `feat: add filesystem capacity accounting`

---

### Task 4: Add runtime LRU eviction, read pins, invalidation, and atomic concurrent admission

**Files:**

- Modify: `vllm/v1/kv_offload/tiering/fs/capacity.py`
- Modify: `tests/v1/kv_offload/tiering/test_fs_capacity.py`

**Interfaces:**

- Produces:
  - `contains(path: str) -> bool`
  - `contains_many(paths: list[str]) -> list[bool]`
  - `touch(paths: Iterable[str]) -> None`
  - `pin_for_read(path: str) -> ReadPin | None`
  - `release_read(pin: ReadPin, *, invalidate: bool = False) -> None`
- Extends `admit_write()` to perform deterministic LRU reclaim before returning `CAPACITY`.

- [ ] **Step 1: Write RED LRU/read-pin tests.**

Add tests that recover three small managed finals and assert:

```python
cap.touch([newest_path])
result = cap.admit_write(incoming_path, incoming_size)
self.assertEqual(result.status, AdmissionStatus.RESERVED)
self.assertFalse(Path(oldest_path).exists())
self.assertTrue(Path(newest_path).exists())
```

Add:

```python
pin = cap.pin_for_read(oldest_path)
self.assertIsNotNone(pin)
result = cap.admit_write(incoming_path, incoming_size)
self.assertNotEqual(result.status, AdmissionStatus.RESERVED)
self.assertTrue(Path(oldest_path).exists())
cap.release_read(pin)
```

Add a test where one victim unlink raises `PermissionError`, a later victim succeeds, and admission still succeeds without decrementing the failed victim’s accounting. Add a test where all candidates fail/pinned and admission returns `CAPACITY` with no reservation.

- [ ] **Step 2: Add RED concurrent writer test.**

Use `threading.Barrier` to start two admissions against a nearly full manager. Record snapshots after each reservation and assert every observed `accounted + reserved <= max_bytes`. Exactly one writer may consume any one unit of free capacity before eviction makes room.

- [ ] **Step 3: Run RED.**

```bash
python -m unittest discover -s tests/v1/kv_offload/tiering \
  -p 'test_fs_capacity.py' -v
```

- [ ] **Step 4: Implement deterministic runtime recency and victim claim.**

- successful commit gets the next monotonic recency
- `touch()` updates only `COMMITTED` entries
- victim eligibility is `state is COMMITTED and readers == 0`
- never pick the incoming/replacement path
- claim `COMMITTED -> EVICTING` while holding `metadata_lock`
- release `metadata_lock`, unlink while still holding `admission_lock`
- only confirmed success/`ENOENT` removes the entry and decrements bytes
- non-`ENOENT` failure restores that exact entry generation to `COMMITTED`, increments eviction-failure counter, and adds the path to a per-admission exclusion set so the loop cannot select the same failing victim forever

- [ ] **Step 5: Implement read pins and INVALID cleanup.**

`pin_for_read()` succeeds only for the exact current `COMMITTED` generation and increments `readers` under `metadata_lock`.

`release_read(pin, invalidate=True)`:

1. verifies generation identity;
2. transitions the entry to `INVALID` before decrementing the reader count;
3. prevents all future HIT/pin/touch operations;
4. once the last reader leaves, attempts accounting-aware unlink under `admission_lock`;
5. on success/`ENOENT`, removes the entry and decrements accounted bytes;
6. on other unlink failure, keeps the entry `INVALID` and accounted and logs the cleanup failure without masking the original read error.

- [ ] **Step 6: Run GREEN plus repeated concurrency coverage.**

```bash
python -m unittest discover -s tests/v1/kv_offload/tiering \
  -p 'test_fs_capacity.py' -v
python - <<'PY'
import subprocess, sys
for _ in range(10):
    rc = subprocess.call([
        sys.executable, '-m', 'unittest', 'discover',
        '-s', 'tests/v1/kv_offload/tiering', '-p', 'test_fs_capacity.py', '-v'
    ])
    if rc:
        raise SystemExit(rc)
PY
```

- [ ] **Step 7: Checkpoint commit.**

Commit message: `feat: add filesystem lru capacity eviction`

---

### Task 5: Add exclusive namespace ownership and deterministic restart recovery

**Files:**

- Modify: `vllm/v1/kv_offload/tiering/fs/capacity.py`
- Modify: `tests/v1/kv_offload/tiering/test_fs_capacity.py`

**Interfaces:**

- Construction becomes READY-only: lock acquisition, temp/corrupt cleanup, accounting rebuild, over-limit startup shrink, and free-space diagnostic all finish before the constructor returns.
- `close() -> None` releases namespace ownership after retrying orphan cleanup.

- [ ] **Step 1: Write RED ownership/recovery tests.**

Add exact cases for:

- second manager on the same namespace raises while first is alive;
- after first `close()`, a new manager acquires the same namespace;
- valid finals rebuild `accounted_bytes` from `st_size`;
- recognized `*.bin_<digits>.tmp` is deleted at startup;
- temp cleanup failure causes construction failure;
- mapper-shaped `.bin` with wrong `expected_file_size` is treated as known corrupt cache data and deleted; cleanup failure is fatal;
- unknown regular file is not deleted and construction fails;
- startup usage above a smaller `max_bytes` evicts deterministically by `(mtime_ns, relative_path)` until bounded;
- startup victim unlink failure is immediately fatal;
- a historical entry larger than `max_bytes` is removed during startup shrink;
- lock/control files are excluded from payload accounting.

- [ ] **Step 2: Run RED.**

```bash
python -m unittest discover -s tests/v1/kv_offload/tiering \
  -p 'test_fs_capacity.py' -v
```

- [ ] **Step 3: Implement lifetime ownership lock.**

Use a dedicated file at:

```python
lock_path = os.path.join(namespace_root, '.capacity.lock')
```

Open it once, acquire `fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)`, retain the fd for the manager lifetime, and release/close on `close()` or constructor failure. If advisory locking is unavailable/unreliable in the running environment, fail construction rather than silently sharing accounting ownership.

- [ ] **Step 4: Implement strict namespace classification.**

A recognized final must match mapper layout:

```text
<root>/<first-3-hex>/<next-2-hex>_g<nonnegative-int>/<same-hash-hex>.bin
```

and directory prefixes must agree with the filename hash. A recognized temp must be the same valid final filename plus `_<digits>.tmp`.

- `.capacity.lock` is control metadata and ignored by payload scan.
- directories are allowed.
- any other non-directory artifact is unknown: do not delete it; fail construction.
- if `expected_file_size` is not `None`, a recognized final with a different `st_size` is known corrupt cache data: delete it or fail if deletion cannot be confirmed.

- [ ] **Step 5: Implement recovery LRU and startup shrink.**

For valid recovered finals, sort by:

```python
(stat.st_mtime_ns, os.path.relpath(path, namespace_root))
```

Assign monotonic recency in that order. If recovered usage exceeds `max_bytes`, synchronously unlink oldest entries until `accounted_bytes <= max_bytes`. Any non-`ENOENT` startup unlink failure aborts construction.

- [ ] **Step 6: Add logical-vs-physical free-space diagnostic test and implementation.**

Monkeypatch `shutil.disk_usage` so `free < max_bytes`; assert construction succeeds and emits a warning that physical `ENOSPC` may occur before logical capacity. Never rewrite `max_bytes` from free space.

- [ ] **Step 7: Run GREEN.**

```bash
python -m unittest discover -s tests/v1/kv_offload/tiering \
  -p 'test_fs_capacity.py' -v
```

- [ ] **Step 8: Checkpoint commit.**

Commit message: `feat: recover bounded filesystem cache on startup`

---

### Task 6: Integrate worker-side store admission, raw I/O ownership, and accurate stored events

**Files:**

- Modify: `vllm/v1/kv_offload/tiering/fs/io.py`
- Modify: `vllm/v1/kv_offload/tiering/fs/manager.py`
- Modify: `tests/v1/kv_offload/tiering/test_fs_tier.py`
- Modify: `tests/v1/kv_offload/tiering/test_fs_capacity.py`

**Interfaces:**

- `io.make_temp_path(dest_path: str) -> str`
- `io.store_block(dest_path: str, tmp_path: str, buffer: memoryview, offset: int, block_size: int) -> None`
- `io.load_block(...)` performs read only and never deletes final files.
- `FileSystemTierManager._store_one(...)` performs admission, real I/O, temp cleanup/orphan retention, commit, and per-key outcome recording.

- [ ] **Step 1: Write RED raw-I/O ownership tests.**

In `test_fs_capacity.py`, verify:

- `load_block()` raising on a short/missing read does **not** unlink the final path itself;
- `store_block()` uses the caller-supplied temp path and does not perform its own destination-exists idempotency check;
- a store exception may leave the temp for the manager wrapper to classify, rather than silently warning and releasing accounting behind the capacity manager.

- [ ] **Step 2: Write RED integration/event tests.**

In `test_fs_tier.py`, add cases:

```python
# max == one block: key1 commits, key2 forces LRU eviction, job succeeds.
# max < block: store returns successful JobResult but creates no file.
# mixed [committed, capacity-skipped] successful job emits event only for committed key.
# all capacity-skipped successful job emits no event.
# already-present idempotent store emits no new stored event.
# injected real store I/O failure keeps JobResult.success == False and no stored event.
```

- [ ] **Step 3: Run RED locally where possible.**

```bash
python -m unittest discover -s tests/v1/kv_offload/tiering \
  -p 'test_fs_capacity.py' -v
```

If pytest exists, also run the exact new `test_fs_tier.py` nodes. Otherwise record them for CI.

- [ ] **Step 4: Refactor raw I/O helpers.**

`store_block()` must:

1. ensure destination parent dirs;
2. open the supplied temp path with `O_CREAT|O_EXCL|O_WRONLY|O_TRUNC|O_DIRECT`;
3. write the exact payload;
4. close the fd;
5. call `os.replace(tmp_path, dest_path)`;
6. propagate exceptions without making capacity/accounting decisions.

`load_block()` only opens/reads/closes and propagates exceptions. Remove its final-file deletion behavior.

- [ ] **Step 5: Instantiate capacity manager before the worker pool.**

After mapper/config setup and before `DualQueueThreadPool(...)`:

```python
self._capacity = FileSystemCapacityManager(
    namespace_root=self.file_mapper.get_data_dir_path(),
    max_bytes=self.max_bytes,
    expected_file_size=self._block_size,
)
```

The tier is not usable until this construction returns.

- [ ] **Step 6: Implement `_store_one`.**

Exact flow:

```text
admission = capacity.admit_write(final_path, block_size)
if ALREADY_PRESENT / DUPLICATE_INFLIGHT / OVERSIZED / CAPACITY:
    return normally
reservation = admission.reservation
tmp_path = make_temp_path(final_path)
try:
    store_block(final_path, tmp_path, primary_view, offset, block_size)
except Exception:
    try unlink tmp_path
      success/ENOENT -> capacity.abort_write(reservation)
      other error    -> capacity.retain_orphan_temp(reservation, tmp_path)
    raise
else:
    capacity.commit_write(reservation)
    record this exact key index as committed
```

Do not acquire reservations in `submit_store()` itself. Build worker callables only; reservation starts when the worker begins.

- [ ] **Step 7: Replace whole-job key event bookkeeping with per-key commit bookkeeping.**

Use a small thread-safe job record containing original key order and committed indices. On `get_finished_jobs()`:

- `success=False`: emit no stored event, clear record;
- `success=True`: emit exactly the committed subset in original order;
- empty committed subset: no event.

Capacity/already-present/duplicate skips remain normal worker completion and never masquerade as a new commit.

- [ ] **Step 8: Run GREEN.**

```bash
python -m unittest discover -s tests/v1/kv_offload/tiering \
  -p 'test_fs_capacity.py' -v
python - <<'PY'
import importlib.util, subprocess, sys
if importlib.util.find_spec('pytest'):
    raise SystemExit(subprocess.call([
        sys.executable, '-m', 'pytest',
        'tests/v1/kv_offload/tiering/test_fs_tier.py', '-q'
    ]))
print('pytest unavailable: fs-tier pytest coverage deferred to GitHub CI')
PY
```

- [ ] **Step 9: Checkpoint commit.**

Commit message: `feat: enforce filesystem capacity on stores`

---

### Task 7: Integrate bounded lookup, load revalidation/read pins, invalidation, and touch

**Files:**

- Modify: `vllm/v1/kv_offload/tiering/fs/manager.py`
- Modify: `tests/v1/kv_offload/tiering/test_fs_tier.py`
- Modify: `tests/v1/kv_offload/tiering/test_fs_capacity.py`

**Interfaces:**

- `FsAsyncLookupManager.batch_lookup()` uses capacity membership, not `os.path.exists()`/batch C lookup, while preserving async lookup scheduling.
- `FileSystemTierManager._load_one(...)` revalidates/pins immediately before real read.
- `FileSystemTierManager.touch(...)` updates capacity LRU in memory only.

- [ ] **Step 1: Write RED lookup/read-race tests.**

Cover:

- committed capacity state -> HIT;
- pending write -> no HIT;
- EVICTING/INVALID -> no HIT;
- lookup may resolve HIT, then eviction removes the entry before `submit_load()`; `submit_load()` revalidation fails, performs no filesystem read, and reports failed promotion job;
- active real load pin prevents eviction until the worker releases the pin;
- load read failure invalidates and removes the file through capacity-aware deletion; if invalid cleanup unlink fails, the entry remains INVALID and accounted but can no longer HIT;
- `touch([key])` changes LRU order without changing file mtime.

- [ ] **Step 2: Run RED.**

Run new stdlib-capacity tests and the focused pytest nodes if pytest exists.

- [ ] **Step 3: Replace runtime existence lookup with capacity membership.**

`FsAsyncLookupManager.batch_lookup()` maps keys to final paths and returns `self._tier._capacity.contains_many(paths)`. Remove the manager’s batch-C dispatch dependency from bounded lookup. Keep the standalone C-extension unit test if it still has independent value, but replace `test_batch_lookup_dispatch` with a bounded-state dispatch test.

- [ ] **Step 4: Implement `_load_one`.**

```python
pin = self._capacity.pin_for_read(path)
if pin is None:
    raise FileNotFoundError(f'filesystem cache entry is no longer committed: {path}')
try:
    load_block(path, self._primary_kv_view, offset, self._block_size)
except Exception:
    self._capacity.release_read(pin, invalidate=True)
    raise
else:
    self._capacity.release_read(pin)
```

No filesystem I/O occurs when the pin cannot be acquired. The thread-pool job fails, allowing the existing tiering manager to call `primary.complete_write(..., success=False)` and release the reserved primary promotion slot.

- [ ] **Step 5: Implement `touch()`.**

Map keys to paths and call capacity `touch(paths)` under the capacity manager’s short metadata lock. Do not call `stat`, `utime`, or write persistent recency metadata.

- [ ] **Step 6: Run GREEN and regression.**

Run `test_fs_capacity.py`, all of `test_fs_tier.py` when pytest is available, and `python -m compileall -q vllm/v1/kv_offload/tiering/fs`.

- [ ] **Step 7: Checkpoint commit.**

Commit message: `feat: coordinate filesystem reads with eviction`

---

### Task 8: Add low-cardinality capacity metrics and deterministic multi-FS identity

**Files:**

- Modify: `vllm/v1/kv_offload/tiering/base.py`
- Modify: `vllm/v1/kv_offload/tiering/factory.py`
- Modify: `vllm/v1/kv_offload/tiering/spec.py`
- Modify: `vllm/v1/kv_offload/tiering/fs/manager.py`
- Modify: `tests/v1/kv_offload/tiering/test_factory.py`
- Modify: `tests/v1/kv_offload/tiering/test_shadow_cost_spec.py`
- Modify: `tests/v1/kv_offload/tiering/test_fs_capacity.py`
- Modify: `tests/v1/kv_offload/tiering/test_fs_tier.py`

**Interfaces:**

- `SecondaryTierManager.instance_id: str` defaults to `tier_type`.
- `SecondaryTierFactory.create_secondary_tier(..., instance_id: str | None = None)` preserves existing callers and, when provided, sets the created instance identity after construction.
- `TieringOffloadingSpec.get_manager()` passes `f'{tier_type}:{i}'` for configured secondary tiers.
- FS metrics use fixed low-cardinality labels only.

- [ ] **Step 1: Write RED identity tests.**

In `test_factory.py`:

```python
tier = SecondaryTierFactory.create_secondary_tier(
    {'type': 'example'}, primary_kv_view, offloading_spec, instance_id='example:3'
)
assert tier.instance_id == 'example:3'
```

Preserve a test that old three-argument factory calls still work and default identity remains the tier type.

In `test_shadow_cost_spec.py`, update the stub to accept `instance_id=None` and assert configured duplicate FS tiers receive deterministic `fs:0`, `fs:1` runtime identities without altering `_cost_model_tier_keys` behavior.

- [ ] **Step 2: Implement minimal identity plumbing.**

Base:

```python
self.tier_type = tier_type
self.instance_id = tier_type
```

Factory:

```python
def create_secondary_tier(..., instance_id: str | None = None):
    ...
    tier = tier_cls(...)
    if instance_id is not None:
        tier.instance_id = instance_id
    return tier
```

Spec creation loop passes `instance_id=f"{runtime_tier_config['type']}:{i}"`.

Do not reuse or alter `cost_model_tier_key`.

- [ ] **Step 3: Write RED metric-definition/snapshot tests.**

Required FS names:

```text
vllm:kv_offload_fs_capacity_bytes
vllm:kv_offload_fs_accounted_bytes
vllm:kv_offload_fs_reserved_bytes
vllm:kv_offload_fs_evictions
vllm:kv_offload_fs_evicted_bytes
vllm:kv_offload_fs_capacity_skips
vllm:kv_offload_fs_eviction_failures
```

Label contract:

- gauges/eviction counters: `('tier',)`
- capacity skips: `('tier', 'reason')`
- fixed reasons only: `oversized`, `no_evictable_capacity`
- never path/hash/request/temp labels.

Create two FS managers with `instance_id='fs:0'` and `'fs:1'`, aggregate their `OffloadingConnectorStats`, and assert the gauge label tuples remain distinct instead of last-writer overwrite.

- [ ] **Step 4: Implement metric definitions and `get_stats()`.**

Capacity `snapshot()` must take all gauges/counters under one metadata-lock snapshot. FS manager retains the prior cumulative counter snapshot and emits only positive counter deltas; gauges always emit current values. This avoids double-counting while keeping startup evictions visible on the first stats poll.

- [ ] **Step 5: Run GREEN.**

Run stdlib capacity tests. If pytest is present, run `test_factory.py`, `test_shadow_cost_spec.py`, and `test_fs_tier.py`; otherwise record CI deferral.

- [ ] **Step 6: Checkpoint commit.**

Commit message: `feat: expose filesystem capacity metrics`

---

### Task 9: Prove shutdown/cancellation and full integration invariants

**Files:**

- Modify: `vllm/v1/kv_offload/tiering/fs/manager.py`
- Modify: `tests/v1/kv_offload/tiering/test_fs_capacity.py`
- Modify: `tests/v1/kv_offload/tiering/test_fs_tier.py`

**Interfaces:**

- `FileSystemTierManager.shutdown()` ordering becomes: stop lookup manager -> stop/join pool -> close capacity manager/ownership lock.
- No queued task owns a reservation before a worker starts.

- [ ] **Step 1: Add RED shutdown tests.**

Use a worker barrier to hold one active write after reservation and enqueue additional store jobs behind it. Assert before release:

- active worker owns exactly one reservation;
- queued jobs own none;
- after releasing the active worker and calling shutdown, normal-path `reserved_bytes == 0` and the ownership lock can be reacquired by a fresh capacity manager.

Add an orphan-temp cleanup-failure test that verifies shutdown does **not** fake `reserved_bytes=0`; it logs/surfaces the surviving orphan reservation and restart recovery remains responsible for cleanup/fail-fast.

- [ ] **Step 2: Implement shutdown ordering and assertions.**

After `_pool.shutdown(wait=True)`, no active worker may still mutate capacity state. Call capacity `close()`, which retries orphan cleanup once. Normal shutdown asserts no non-orphan active reservations remain. Surviving orphan temp bytes stay explicit until process exit; never silently subtract them while the temp may exist.

- [ ] **Step 3: Run the full focused local suite.**

```bash
python -m unittest discover -s tests/v1/kv_offload/tiering \
  -p 'test_fs_capacity.py' -v
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_issue31_fs_capacity_config.py' -v
python -m compileall -q vllm/v1/kv_offload/tiering/fs \
  vllm/v1/kv_offload/tiering/spec.py \
  vllm/v1/kv_offload/tiering/factory.py \
  benchmarks/cache
```

If pytest exists, also run:

```bash
python -m pytest \
  tests/v1/kv_offload/tiering/test_fs_tier.py \
  tests/v1/kv_offload/tiering/test_factory.py \
  tests/v1/kv_offload/tiering/test_shadow_cost_spec.py \
  benchmarks/cache/tests/test_config.py \
  benchmarks/cache/tests/test_scenarios.py -q
```

- [ ] **Step 4: Run static checks on only changed Python files.**

Use the repository-provided Ruff/pre-commit entrypoints available in the Pod; do not install new tooling. Also run:

```bash
git diff --check
git status --short
```

- [ ] **Step 5: Checkpoint commit.**

Commit message: `test: cover filesystem capacity lifecycle`

---

### Task 10: Add and run a deterministic real-filesystem hard-cap smoke

**Files:**

- Create: `benchmarks/cache/issue31_fs_capacity_smoke.py`
- Create: `benchmarks/cache/tests/test_issue31_fs_capacity_smoke.py`

**Interfaces:**

- CLI: `python benchmarks/cache/issue31_fs_capacity_smoke.py --root <owned-empty-dir> --output <json>`
- Output JSON contains at least: schema/version, root, max/block sizes, peak accounted/reserved/combined, eviction counters, capacity skips, temp peak observation, restart recovered bytes, startup shrink result, ownership-lock result, final payload apparent bytes, filesystem provenance label.

- [ ] **Step 1: Write RED smoke-contract test.**

The unittest creates a temporary root, invokes the module’s `run_smoke(root: Path) -> dict`, and asserts:

```python
self.assertLessEqual(result['peak_accounted_plus_reserved_bytes'], result['max_bytes'])
self.assertTrue(result['temp_peak_observed'])
self.assertTrue(result['runtime_eviction_observed'])
self.assertTrue(result['restart_recovery_ok'])
self.assertTrue(result['startup_shrink_ok'])
self.assertTrue(result['ownership_conflict_rejected'])
self.assertEqual(result['filesystem_provenance'], 'filesystem')
```

- [ ] **Step 2: Run RED.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_issue31_fs_capacity_smoke.py' -v
```

- [ ] **Step 3: Implement the smoke with real small files, not mocks.**

Use a small fixed logical unit such as 4096 bytes and a mapper-shaped path helper. The smoke must:

1. instantiate `FileSystemCapacityManager` on an empty owned root;
2. create/commit real final files until capacity is full;
3. touch one entry and admit another, observing a real LRU `unlink`;
4. obtain concurrent reservations, create actual temp files with `open(..., 'wb')`, write them fully, pause before `os.replace`, and record `sum(final st_size)`, temp apparent bytes, `accounted`, and `reserved` while temp bytes physically exist;
5. assert `accounted + reserved <= max_bytes` at that peak;
6. replace/commit and verify no stale temp remains;
7. close/reopen to prove accounting rebuild;
8. close and reopen with a smaller max to prove synchronous startup shrink;
9. attempt a second owner while the first is alive and record rejection;
10. record `shutil.disk_usage(root)` only as diagnostic provenance, never as quota authority.

Do not fill the Pod disk to simulate ENOSPC. ENOSPC rollback remains fault-injection coverage in unit tests.

- [ ] **Step 4: Run GREEN contract test.**

```bash
python -m unittest discover -s benchmarks/cache/tests \
  -p 'test_issue31_fs_capacity_smoke.py' -v
```

- [ ] **Step 5: Run the formal Pod filesystem smoke and preserve evidence.**

Use a new run directory; never reuse/delete an old formal evidence directory:

```bash
mkdir -p /code/results/cache/issue31-fs-capacity-20260812
python benchmarks/cache/issue31_fs_capacity_smoke.py \
  --root /code/results/cache/issue31-fs-capacity-20260812/cache-root \
  --output /code/results/cache/issue31-fs-capacity-20260812/smoke.json
cat /code/results/cache/issue31-fs-capacity-20260812/smoke.json
```

Expected: exit 0 and `peak_accounted_plus_reserved_bytes <= max_bytes`.

- [ ] **Step 6: Checkpoint commit.**

Commit message: `validate: add filesystem capacity smoke`

---

### Task 11: Produce final local validation artifacts and update engineering state

**Files:**

- Create: `docs/engineering/validation/2026-08-12-issue31-filesystem-hard-capacity-validation.json`
- Create: `docs/engineering/validation/2026-08-12-issue31-filesystem-hard-capacity-validation.md`
- Modify: `docs/engineering/CURRENT_STATE.md`

**Interfaces:**

- Consumes: final focused test logs, formal `/code/results/cache/issue31-fs-capacity-20260812/smoke.json`, exact GitHub base/head SHAs, design/plan paths.
- Produces: checked-in evidence sufficient to review hard-cap correctness without claiming CI has run yet.

- [ ] **Step 1: Invoke `superpowers:verification-before-completion`.**

Do not claim #31 implemented/validated before fresh verification output exists.

- [ ] **Step 2: Re-run fresh verification from the final worktree.**

Run the Task 9 focused suite, Task 10 smoke-contract test, `compileall`, targeted Ruff/pre-commit, and `git diff --check` again. Capture concise command/result summaries outside `/code/results/cache`; do not dump huge logs into chat.

- [ ] **Step 3: Generate the structured validation JSON from real evidence.**

The JSON must contain actual values, not placeholders, with this top-level shape:

```json
{
  "schema_version": 1,
  "issue": 31,
  "status": "locally_validated_ci_pending",
  "base_sha": "<actual live base sha>",
  "head_sha": "<actual implementation head sha>",
  "design_spec": "docs/superpowers/specs/2026-08-12-filesystem-kv-cache-hard-capacity-design.md",
  "implementation_plan": "docs/superpowers/plans/2026-08-12-issue31-filesystem-kv-cache-hard-capacity.md",
  "focused_verification": {},
  "filesystem_smoke": {},
  "ci": {"authoritative": true, "status": "pending_pr"}
}
```

The angle-bracket values above describe fields to populate programmatically from `git rev-parse` and the smoke JSON during execution; do not literally write angle-bracket strings into the artifact.

- [ ] **Step 4: Write the human-readable report.**

It must explicitly answer:

- configured capacity and block size;
- peak accounted, reserved, and combined bytes;
- temp peak evidence;
- runtime LRU eviction behavior;
- capacity skip behavior;
- restart recovered usage;
- smaller-max synchronous shrink;
- stale temp/corrupt/unknown-artifact behavior;
- ownership-lock behavior;
- logical quota versus physical free-space/ENOSPC distinction;
- event correctness;
- multiple-FS metric identity;
- inference/cascade correctness;
- remaining non-goals and #16 block.

Call the environment `filesystem`/container-local as appropriate; never upgrade provenance to NVMe without device evidence.

- [ ] **Step 5: Update `CURRENT_STATE.md` conservatively.**

State that #31 implementation is locally validated on the Draft-PR branch and awaiting authoritative GitHub CI/merge; #16 remains blocked until #31 is merged and closed. Do not state #31 is merged before it actually is.

- [ ] **Step 6: Final local scope check and commit.**

```bash
git diff --check
git status --short
git diff --stat
```

Commit message: `validate: record issue 31 filesystem capacity evidence`

---

### Task 12: Publish Draft PR, run authoritative CI, review, and stop at merge authorization gate

**Files:** No new implementation files unless CI/review reveals a defect.

**Interfaces:**

- Produces: Draft PR from `agent/issue31-fs-hard-capacity` to `main`, body containing `Closes #31`, authoritative CI evidence, and an explicit merge-authorization gate.

- [ ] **Step 1: Publish all reviewed checkpoint commits through the GitHub write path.**

Do not `git push` from the Pod. Verify live GitHub branch content/SHAs after publication and compare the final branch against live `main`.

- [ ] **Step 2: Invoke `superpowers:requesting-code-review` before merge readiness.**

Review the final diff against the approved design and this plan, with special attention to:

- all payload mutation paths going through capacity accounting;
- no reservation release before temp absence is confirmed after failure;
- no read/eviction race;
- no gauge overwrite across multiple FS tiers;
- no unbounded explicit FS config path;
- no #16/#19 policy code.

- [ ] **Step 3: Create a Draft PR.**

PR title: `feat: bound filesystem kv cache capacity`

PR body must include:

```text
Closes #31
```

and summarize the hard-cap invariant, reservation/temp accounting, LRU/restart behavior, focused verification, real-filesystem smoke provenance, and explicit non-goals.

- [ ] **Step 4: Wait only in the workflow sense by querying authoritative GitHub CI in the current turn/session; do not claim background work.**

Fetch workflow runs/checks for the exact final head SHA. If checks fail, invoke `superpowers:systematic-debugging` before proposing or applying a fix, add a failing regression test first where applicable, republish the fix, and re-run authoritative CI.

- [ ] **Step 5: Record final CI provenance on Issue #31 without editing the original Issue body.**

Add an Issue comment containing exact PR number, final head SHA, workflow/check names and statuses, local validation artifact paths, and real-filesystem smoke evidence path. This comment completes the `ci` provenance requirement without creating a post-CI code commit that would invalidate the just-observed head SHA.

- [ ] **Step 6: Stop and request explicit user merge authorization.**

Green CI and successful review are not permission to merge. Do not close #31 manually; the PR’s `Closes #31` should close it only after authorized merge.

- [ ] **Step 7: After explicit authorization only, invoke the merge/branch-finish workflow.**

Use `superpowers:finishing-a-development-branch`, merge through GitHub, verify Issue #31 is closed by the merged PR, verify `main` contains the merge, and only then treat #16 as unblocked for a separate design/implementation process.

---

## Plan Self-Review Checklist

Before execution begins, the plan is considered internally consistent only if all of these mappings hold:

1. authoritative capacity source -> Tasks 3-5
2. committed vs reserved -> Task 3
3. temp bytes count -> Tasks 3, 6, 10
4. reservation lifecycle -> Tasks 3, 6
5. atomic concurrent reservation -> Task 4
6. no capacity locks during payload I/O -> Tasks 4, 6, 7
7. victim selection/reservation coordination -> Task 4
8. overwrite/replacement accounting -> Task 3
9. failure/cancellation rollback -> Tasks 3, 6, 9
10. oversized entry -> Tasks 3, 6
11. restart usage rebuild -> Task 5
12. restart over-limit shrink -> Task 5
13. orphan/temp handling -> Tasks 3, 5, 6
14. restart LRU recovery -> Task 5
15. lookup/read versus eviction -> Tasks 4, 7
16. eviction unlink failure -> Task 4
17. physical ENOSPC versus logical max -> Tasks 5, 6, 11
18. metrics -> Task 8
19. disabled/default behavior -> Task 2
20. metadata versioning not needed -> no persistent metadata file is created anywhere in Tasks 2-11
21. event correctness -> Task 6
22. multiple FS tier identity -> Task 8
23. inference/cascade correctness -> Tasks 6, 7, 9 and existing tiering-manager semantics
24. real filesystem peak/restart evidence -> Task 10
25. final structured/human evidence -> Tasks 11-12
26. Draft PR/CI/merge authorization -> Task 12

No task introduces cost-aware policy, active restore/recompute, cross-process quota coordination, or filesystem-capacity auto-sizing.