# Issue #31 Design: Filesystem KV Cache Hard Capacity and Eviction

**Date:** 2026-08-12  
**Issue:** #31 — `[P0] 为 filesystem KV cache 增加硬容量上限与淘汰机制`  
**Base:** `main@c4d9fce61ec5a8eadc24dab8698eca7705d005bf`  
**Status:** Approved design; implementation not started

## 1. Goal and product contract

V1 follows the product principle:

> 用户决定给多少资源，vLLM 决定这些资源怎么用最划算。

For an explicitly configured local filesystem secondary tier, the configured capacity is a **hard logical ceiling** for cache-owned payload bytes, independent of current free space on the underlying filesystem.

The implementation must preserve inference correctness when cache writes cannot be admitted or when filesystem I/O fails. The filesystem tier is an optimization tier; capacity pressure must degrade cache effectiveness, not model correctness.

Issue #31 establishes the bounded safety mechanism only. Later work may improve placement or admission policy, but it must build on the hard-cap accounting and reservation rules defined here.

## 2. Scope and non-goals

### In scope

- required explicit filesystem capacity configuration
- authoritative committed-byte accounting
- reservation/in-flight-byte accounting
- temp-file peak accounting
- concurrency-safe admission
- deterministic LRU eviction
- overwrite/replacement accounting correctness
- failure/cancellation rollback
- oversized-entry handling
- active-read protection from eviction
- restart accounting recovery
- synchronous startup shrink when usage exceeds a new smaller max
- stale temp cleanup and unknown-artifact fail-fast behavior
- single-owner namespace protection
- capacity/event/metric observability
- real-filesystem validation of the hard-cap invariant

### Explicit non-goals

- #19 cost-aware admission or eviction strategy
- #16 active restore/recompute decisions
- #18 multi-device placement
- adaptive capacity recommendations
- deriving `max_bytes` from free space
- a full adaptive controller
- CPU-tier redesign
- cross-process quota coordination
- persistent capacity metadata DB/WAL/index
- claiming physical NVMe provenance unless validation actually runs on NVMe

## 3. Configuration contract

An explicit filesystem secondary tier must include a positive integer `max_bytes`:

```json
{
  "type": "fs",
  "root_dir": "/cache/path",
  "max_bytes": 107374182400
}
```

Rules:

- missing `max_bytes`: configuration error
- `max_bytes == 0`: configuration error
- negative value: configuration error
- float/string/bool: configuration error
- positive value smaller than one cache entry: valid configuration; oversized entries are skipped before I/O

A higher-level user setting representing “local cache size = 0” should omit the filesystem tier rather than instantiate a zero-capacity tier.

`max_bytes` is a resource-policy property and **must not participate in the `FileMapper` namespace hash**. Restarting a compatible cache namespace with a smaller `max_bytes` must reuse existing files and synchronously shrink the namespace before READY.

## 4. Capacity contract and authoritative accounting

The core invariant from READY onward is:

```text
accounted_bytes + reserved_bytes <= max_bytes
```

Definitions:

- `accounted_bytes`: logical sizes of committed cache-owned final payload files still managed by this capacity owner.
- `reserved_bytes`: complete payload sizes of writes that have been admitted but have not yet committed or aborted.

The authoritative source is split by lifecycle:

1. **Startup:** filesystem scan of the owned data namespace rebuilds committed usage.
2. **Runtime:** the in-memory capacity manager becomes the authoritative accounting state. All owned final-file mutations must flow through its state machine.

The implementation must not re-run `du` or a full-directory scan on every admission. Runtime correctness comes from atomic in-memory accounting plus exclusive namespace ownership.

### Temp files count

Reservations cover the full incoming payload size before the first payload write. This ensures temp-file peak usage is part of the hard-cap contract.

For replacement of an existing file, reserve the **full new size**, not only the delta, because old final and new temp may coexist until `os.replace()` succeeds.

The hard-cap applies to cache-owned payload file logical bytes (`st_size`) and active temp-payload reservations. Directory/inode/control-file/journal/filesystem-allocation overhead is outside the exact application-level byte contract.

## 5. State model

The capacity manager tracks final-path identity, size, recency, readers, and a state/generation token.

### Entry states

- `COMMITTED`: final payload is valid and accounted; may serve reads; eligible for eviction only when `readers == 0`.
- `WRITING`: a writer owns an active reservation for this path; at most one writer may own a path at a time.
- `EVICTING`: an admission flow has exclusively claimed this committed entry as a victim; its bytes remain accounted until deletion is confirmed.
- `INVALID`: the file is no longer trusted for serving after a load/corruption failure, but its bytes remain accounted until deletion is confirmed.

Every state transition that affects accounting must verify the expected entry/reservation identity to avoid ABA-style double-accounting.

## 6. Reservation lifecycle

For a new entry:

```text
worker starts
  -> check existing / duplicate writer / oversize
  -> admission
  -> evict if needed
  -> grant reservation
  -> write temp
  -> os.replace(temp, final)
  -> commit reservation
```

Admission must ensure:

```text
accounted + reserved + incoming <= max
```

before the reservation is granted.

### Commit

After filesystem replacement succeeds:

```text
reserved -= incoming
accounted += final_size
WRITING -> COMMITTED
```

A very small window in which the final file exists while bytes are still classified as `reserved` is safe because it over-counts rather than under-counts.

### Abort

On write/replace/cancellation failure after reservation:

```text
cleanup temp if present
reserved -= incoming
remove WRITING ownership
```

Abort must be idempotent so nested exception/finally cleanup cannot release the same reservation twice.

### Queue cancellation

Reservations are acquired only when a worker task actually starts, not when `submit_store()` enqueues it. Therefore queued tasks cancelled during shutdown cannot leak capacity reservations.

## 7. Existing key and replace semantics

Normal content-addressed runtime behavior remains idempotent:

- if a valid committed entry already exists for the key, skip the physical store
- no reservation
- no temp file
- no extra accounted bytes
- update LRU recency

The lower-level capacity/accounting API must nevertheless support replacement correctly for deterministic tests and future callers:

- reserve full new size
- preserve old committed accounting until replacement succeeds
- on success: `accounted = accounted - old + new`, `reserved -= new`
- on failure: old committed entry remains unchanged and reservation is released

Same-size, larger, and smaller replacement accounting must all be tested.

## 8. Concurrency architecture

Use two synchronization domains.

### `metadata_lock`

Short-held lock protecting:

- `accounted_bytes`
- `reserved_bytes`
- entry index and state
- readers
- LRU recency
- writer/reservation identity
- metric snapshot/counter state tied to accounting

### `admission_lock`

Serializes the capacity decision sequence:

- capacity check
- victim selection/claim
- victim unlink/reconcile
- reservation grant

Global lock ordering is:

```text
admission_lock -> metadata_lock
```

Code holding `metadata_lock` must never wait for `admission_lock`.

Large payload reads/writes never hold either capacity lock. Victim `unlink()` runs outside `metadata_lock`, but may remain inside the admission serialization so a second admission cannot consume capacity that is still being reclaimed.

This preserves concurrent payload I/O after reservations are granted while making admission/victim choice deterministic and race-safe.

## 9. Concurrent writer rules

### Different keys

Multiple writers may perform real I/O concurrently once each has a reservation. Reservations participate in the same atomic accounting domain, so two writers cannot both consume the same free capacity.

### Same key

At most one path writer may exist.

- first writer claims `WRITING` and obtains the reservation
- later writer for the same path observes `WRITING` and returns a normal duplicate-inflight skip
- duplicate-inflight is not a capacity rejection and does not write a temp file
- if the first writer fails, a later independent store opportunity may retry

## 10. Read pins, lookup, and eviction

### Read pins

Before real load I/O is queued, a valid committed entry must be pinned under `metadata_lock`:

```text
COMMITTED -> readers += 1
```

The worker unpins in `finally`.

Eviction eligibility requires:

```text
state == COMMITTED and readers == 0
```

This prevents the manager from unlinking an entry being read by its own process and avoids POSIX unlinked-open-file accounting ambiguity.

### Lookup

In bounded mode, runtime lookup should use the in-memory authoritative state rather than `os.path.exists()` as the primary truth:

- `COMMITTED`: HIT candidate
- `WRITING`: MISS/RETRY, never HIT
- `EVICTING`: MISS/RETRY, never HIT
- `INVALID`: MISS
- absent: MISS

Filesystem errors such as external deletion may trigger conservative reconcile paths, but normal runtime lookup does not re-scan the filesystem.

## 11. LRU policy

Issue #31 implements deterministic runtime LRU only.

- successful commit receives newest recency
- `touch()` updates committed-entry recency in memory
- no runtime `mtime` updates
- no frequency/TTL/cost/size-aware victim scoring

Victims are the oldest eligible committed entries (`readers == 0`).

This is deliberately a safety policy, not #19’s future economic policy.

## 12. Eviction semantics

Victim selection and claim are atomic under `metadata_lock`:

```text
COMMITTED -> EVICTING
```

The victim’s size remains part of `accounted_bytes` until the filesystem confirms the file no longer exists.

### Successful unlink

After successful deletion:

```text
remove entry
accounted -= victim.size
```

### `ENOENT`

Because the file is confirmed absent, runtime may reconcile by removing the tracked entry and reducing accounting. The finalize step must verify the same entry/generation is still installed at that path.

### Other unlink failure

- `EVICTING -> COMMITTED`
- `accounted_bytes` unchanged
- runtime admission may try another victim
- if insufficient reclaimable bytes remain, the incoming store becomes a capacity skip

Never release capacity merely because an unlink was attempted.

## 13. Capacity skip versus I/O failure

Capacity pressure is a normal best-effort cache degradation, not a filesystem error.

Normal skip reasons include:

- existing committed key
- duplicate in-flight writer
- oversized incoming entry
- logical capacity cannot be satisfied because no eligible victim can reclaim enough space

Oversized entries (`incoming > max_bytes`) are rejected before eviction, reservation, temp creation, or payload write.

True I/O failures include write, replace, permission, filesystem errors, `ENOSPC`, and `EDQUOT`.

`ENOSPC`/`EDQUOT` may occur even when logical accounting is below `max_bytes`; they must roll back the reservation and temp state but must not change or auto-resize the configured logical quota.

## 14. Store job and KV-event semantics

Internally distinguish at least:

- `COMMITTED`
- `SKIPPED`
- `IO_FAILED`

The public `JobResult` contract does not need to expand for #31.

- committed task: normal completion
- skipped task: normal best-effort completion
- real I/O failure: task failure

### Stored events

Capacity skips must never be reported as stored entries.

For a store job with committed + normal-skipped blocks and no real I/O failure:

- job may complete successfully
- emit `BlockStored` only for the actually committed subset

If all blocks are skipped, emit no stored event.

For a job containing a real I/O failure, preserve the existing conservative contract:

- `JobResult.success == False`
- emit no `BlockStored` event for the failed job, even if some sibling blocks physically committed

This avoids redefining existing partial-I/O-failure event semantics in #31.

## 15. Load corruption and accounting-aware invalidation

Raw load I/O must not directly delete an owned final file.

On load corruption/unreadable failure:

1. mark the entry `INVALID` so it is no longer served to new readers
2. unpin the current reader
3. once readers reach zero, attempt accounting-aware unlink
4. only after confirmed deletion reduce `accounted_bytes`

If invalidation unlink fails, the entry remains unservable but accounted.

“Unusable” does not mean “space is free.”

## 16. Namespace ownership

A bounded filesystem data namespace has exactly one active capacity owner.

At startup:

```text
open dedicated ownership lock file
-> non-blocking exclusive advisory lock
-> hold fd for manager lifetime
```

On Linux, use an advisory primitive such as `fcntl.flock(LOCK_EX | LOCK_NB)`.

- a second bounded manager for the same data namespace fails fast
- normal close releases the lock
- process death releases the kernel lock even if the lock file remains
- the lock file is control metadata and not part of payload `max_bytes`

This is **not** a cross-process quota manager. It forbids two independent bounded accounting owners from concurrently controlling the same namespace.

If reliable required locking semantics are unavailable, construction fails rather than silently weakening the hard-cap guarantee.

## 17. Restart recovery and READY gate

Recovery must finish synchronously before the worker-facing tier becomes READY.

Order:

```text
acquire ownership
-> scan owned data namespace
-> classify artifacts
-> clean recognized temp/corrupt cache artifacts
-> rebuild committed accounting and cold-start LRU
-> shrink if usage > max
-> verify invariants
-> READY
```

No store/load/lookup may begin while recovery is incomplete.

### Scan boundary

Scan only the current `FileMapper` data namespace, not the entire user `root_dir` and not other model/config/rank namespaces.

### Artifact classification

1. **Valid final:** expected mapper layout/name and valid size; restore as `COMMITTED`.
2. **Recognized stale temp:** definitely produced by this tier’s temp naming convention; delete before READY. Cleanup failure is fatal.
3. **Recognized corrupt final:** final naming/layout is ours but size is invalid; do not serve; delete before READY. Cleanup failure is fatal.
4. **Unknown regular artifact:** do not delete and do not ignore; fail construction with diagnostics.

Empty directories are harmless and need not be aggressively cleaned.

### Restart reservations

No reservation survives process restart. Recovered `reserved_bytes` always begins at zero.

## 18. Restart LRU recovery

No persistent LRU metadata DB/WAL is introduced.

For valid recovered finals, cold-start eviction ordering is built from:

```text
(st_mtime_ns, normalized_relative_path)
```

The path tie-break makes ordering deterministic when timestamps match.

After READY, runtime commit/touch operations use an in-memory monotonic recency sequence and do not persist every touch.

Historical LRU order is therefore approximate across restart, but recovery safety and deterministic cold-start behavior are guaranteed.

## 19. Startup over-limit shrink

After scan, if:

```text
accounted_bytes > max_bytes
```

construction must synchronously evict cold-start LRU victims until usage is bounded.

A historical entry larger than the new `max_bytes` is simply a required startup victim; `max_bytes < entry_size` is not itself a configuration error.

Startup behavior is stricter than runtime admission:

- success/`ENOENT`: reconcile and continue
- other victim unlink failure while over limit: fail construction immediately

A tier must never announce READY while recovered usage exceeds the configured hard ceiling.

## 20. External mutation contract

The ownership lock coordinates participating bounded managers; it cannot prevent arbitrary external processes from manually modifying the cache directory.

Runtime guarantees cover manager-controlled mutations. Clear external deletion may be reconciled conservatively when observed.

Arbitrary external writes into the owned namespace are a contract violation; #31 does not add a filesystem watcher or continuous full-directory scanner. The next restart scan will detect unknown pollution and fail fast.

## 21. Shutdown

Shutdown order should ensure:

```text
stop accepting new work
-> finish/cancel queue according to thread-pool contract
-> active worker reservations all commit or abort
-> reserved_bytes == 0
-> close capacity manager
-> release ownership lock fd
```

A nonzero reservation count after active workers have terminated is a correctness bug, not something to silently repair.

## 22. Metrics and observability

Minimum low-cardinality metrics:

- `vllm:kv_offload_fs_capacity_bytes{tier=...}` gauge
- `vllm:kv_offload_fs_accounted_bytes{tier=...}` gauge
- `vllm:kv_offload_fs_reserved_bytes{tier=...}` gauge
- `vllm:kv_offload_fs_evictions{tier=...}` counter
- `vllm:kv_offload_fs_evicted_bytes{tier=...}` counter
- `vllm:kv_offload_fs_capacity_skips{tier=...,reason=...}` counter
- `vllm:kv_offload_fs_eviction_failures{tier=...}` counter

Capacity-skip reason labels must be a bounded enum, such as `oversized` and `no_evictable_capacity`.

Never label by root path, key/hash, request ID, or temp filename.

### Multiple FS tiers

Existing stats aggregation uses the latest gauge value for an identical metric+label tuple, so multiple unlabeled FS tiers would overwrite each other.

Every FS metric therefore carries a deterministic low-cardinality `tier` identity derived from secondary-tier configuration index, e.g. `fs:0`, `fs:1`.

Do not use `root_dir` as a label and do not overload `cost_model_tier_key` for observability identity.

A single capacity stats snapshot should read capacity/accounted/reserved/counters under one short `metadata_lock` critical section to avoid internally inconsistent snapshots.

### Logs

- normal LRU eviction and existing-key skip: no per-block INFO noise
- capacity pressure: metrics first; warnings should be rate-limited/aggregated if needed
- real filesystem failures: clear diagnostics
- startup failures: include namespace, configured max, recovered usage, and failure category

High-cardinality details belong in logs, not metric labels.

## 23. Code decomposition

### New `vllm/v1/kv_offload/tiering/fs/capacity.py`

Owns:

- `EntryRecord`
- `Reservation`
- entry state machine
- accounted/reserved bytes
- LRU/recency
- read pin/unpin
- admission and victim selection
- startup scan/recovery/shrink
- ownership lock
- capacity counters and consistent snapshots

This keeps the hard-capacity algorithm directly unit-testable without requiring real KV memory I/O.

### `fs/manager.py`

Remains the tier orchestration layer:

- validate and accept `max_bytes`
- build `FileMapper`
- initialize capacity manager before accepting work
- bounded lookup
- `touch()`
- worker-side store/load wrappers
- per-key store outcomes
- job/event aggregation
- `get_stats()` integration

### `fs/io.py`

Raw filesystem primitives only. Load helpers must stop owning final-file deletion/accounting semantics.

### `file_mapper.py`

At most add a small explicit accessor for the data namespace root. Capacity policy does not belong in `FileMapper`.

### Generic tier plumbing

If required for per-tier metric identity, add only minimal deterministic secondary-tier instance identity plumbing based on configured index.

## 24. TDD acceptance matrix

Implementation must follow failing-test-first development. Minimum coverage:

1. missing/zero/negative/wrong-type `max_bytes` fails
2. positive `max_bytes < block_size` is valid
3. basic committed accounting
4. existing-key store does not double count
5. reservation is granted before payload write
6. commit converts reserved to committed
7. failure abort releases reservation and cleans temp
8. same/larger/smaller replacement accounting
9. replacement reserves full new size
10. oversized entry skips without eviction/temp/reservation
11. concurrent writers cannot exceed the invariant
12. same-key concurrent store produces only one real writer
13. deterministic LRU victim order
14. `touch()` updates recency
15. active reader cannot be evicted
16. eviction unlink success updates accounting
17. eviction `ENOENT` reconciles safely
18. other unlink failure does not free bytes
19. write/replace/ENOSPC-style failure rolls back safely
20. load corruption enters INVALID and remains accounted until deletion
21. capacity-skipped key never emits false stored event
22. mixed committed + normal-skip job emits only committed subset
23. real partial I/O failure preserves current no-stored-event contract
24. restart scan rebuilds accounting
25. recognized temp cleanup succeeds/fails correctly
26. corrupt recognized final cleanup succeeds/fails correctly
27. unknown artifact fails without deletion
28. smaller restart max synchronously evicts to bound
29. oversized historical entry is evicted during recovery
30. second bounded namespace owner fails fast
31. ownership releases on shutdown/process exit semantics
32. multiple FS tier metrics do not overwrite gauges
33. shutdown ends with zero active reservations
34. capacity rejection preserves cascade completion/primary-pin lifecycle
35. scheduler-facing submission remains lightweight/non-blocking

## 25. Real-filesystem validation

Focused unit tests are necessary but not sufficient. Run a real filesystem-backed smoke with a deliberately small quota.

The smoke must exercise real final and temp files, including a controlled pause after the temp payload is fully written and before `os.replace()` so the peak state can be inspected while temp bytes exist.

Required smoke sequence:

```text
fill to capacity
-> store another block
-> real LRU unlink
-> concurrent temp writes
-> verify bounded payload peak
-> restart
-> restart with smaller max
-> synchronous startup shrink
-> verify rebuilt accounting
```

Record at least:

- configured `max_bytes`
- block size
- peak accounted bytes
- peak reserved bytes
- peak accounted+reserved
- final payload apparent sizes (`st_size`)
- absence/cleanup of stale temps
- eviction count/bytes
- restart recovered usage
- startup shrink result

`du` may be captured as auxiliary evidence, but filesystem allocation rounding, inode metadata, journals, overlay backing, or unrelated disk consumers are not part of the exact logical payload-byte invariant.

Do not intentionally fill the Pod disk merely to create ENOSPC; test ENOSPC rollback with controlled fault injection.

## 26. Final validation evidence and PR gate

The final Issue #31 evidence should record:

- base/head SHAs
- exact configuration
- capacity and block size
- peak accounted/reserved/combined bytes
- eviction counts/bytes
- capacity skips
- restart recovered bytes
- startup shrink result
- temp/orphan/corrupt-file recovery results
- ownership-lock result
- focused test results
- real-filesystem smoke provenance/results
- CI provenance/status

If the environment is container-local/overlay-backed, call it filesystem evidence, not NVMe evidence.

The implementation PR is Draft by default. Green CI is necessary but not merge authorization. Merge occurs only after explicit user authorization, and the final implementation PR should close Issue #31.

## 27. Safety checklist mapping

This design explicitly resolves the Issue #31 safety checklist:

1. authoritative accounting source: startup scan, then in-memory capacity manager
2. committed vs reserved: defined in Section 4
3. temp files count: yes, through full-size reservations
4. reservation lifecycle: Section 6
5. atomic concurrent reservation: admission serialization + metadata lock
6. capacity locks during real I/O: no payload read/write under capacity locks
7. victim selection/reservation coordination: one admission lock, atomic victim claim
8. overwrite delta: full-new-size reservation; post-replace old/new accounting swap
9. failed/cancelled rollback: idempotent abort; worker-side reservation
10. oversized entry: clean pre-I/O skip
11. restart usage rebuild: namespace scan
12. restart over-limit: synchronous deterministic shrink
13. orphan/temp handling: recognized cleanup; unknown fail-fast
14. LRU recovery: `mtime_ns` + path deterministic cold start
15. lookup/read vs eviction: authoritative state + read pins
16. eviction unlink failure: release bytes only after confirmed absence
17. physical ENOSPC vs logical max: distinct failure classes
18. metric integration: tier-local low-cardinality stats through existing hooks
19. disabled/default behavior: explicit FS requires positive `max_bytes`; no FS tier remains unchanged
20. metadata versioning: unnecessary because scan-based recovery is sufficient

## 28. Implementation process gate

Approved workflow:

```text
approved design
-> checked-in design spec
-> spec self-review
-> user spec review
-> writing-plans
-> TDD
-> implementation
-> focused verification
-> real filesystem smoke
-> Draft PR
-> authoritative GitHub CI
-> explicit user merge authorization
```

No production implementation begins before the design spec has passed review and the implementation plan has been written.