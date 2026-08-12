# Issue #31 Design: Filesystem KV Cache Hard Capacity and Eviction

**Date:** 2026-08-12  
**Issue:** #31 — `[P0] 为 filesystem KV cache 增加硬容量上限与淘汰机制`  
**Base:** `main@c4d9fce61ec5a8eadc24dab8698eca7705d005bf`  
**Status:** Approved design; self-reviewed; implementation not started

## 1. Goal and product contract

V1 follows the product principle:

> 用户决定给多少资源，vLLM 决定这些资源怎么用最划算。

For an explicitly configured local filesystem secondary tier, the configured capacity is a **hard logical ceiling** for cache-owned payload bytes, independent of current free space on the underlying filesystem.

The filesystem tier is an optimization tier. Capacity rejection or filesystem write failure must degrade cache effectiveness without breaking inference correctness.

Issue #31 establishes the bounded safety mechanism only. Later work may improve admission, placement, or eviction economics, but it must build on the accounting and reservation rules defined here.

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
- single-owner bounded namespace protection
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
- positive value smaller than one cache entry: valid; oversized entries are skipped before I/O

A higher-level user setting representing “local cache size = 0” should omit the filesystem tier rather than instantiate a zero-capacity tier.

`max_bytes` is a resource-policy property and **must not participate in the `FileMapper` namespace hash**. Restarting a compatible cache namespace with a smaller `max_bytes` must reuse existing files and synchronously shrink that namespace before READY.

## 4. Capacity contract and authoritative accounting

The core invariant from READY onward is:

```text
accounted_bytes + reserved_bytes <= max_bytes
```

Definitions:

- `accounted_bytes`: logical sizes of committed cache-owned final payload files still managed by this owner.
- `reserved_bytes`: complete payload sizes of admitted writes that have not yet committed or aborted.

Authoritative accounting is lifecycle-specific:

1. **Startup:** scan the owned filesystem data namespace to rebuild committed usage.
2. **Runtime:** the in-memory capacity manager is authoritative. Every owned final-file mutation must flow through its state machine.

Runtime admission must not re-run `du` or scan the full namespace. Correctness comes from atomic in-memory accounting plus exclusive bounded-namespace ownership.

### Temp files count

A reservation covers the full incoming payload size **before the first payload write**. Temp-file peak bytes are therefore part of the hard-cap contract.

For replacement, reserve the **full new size**, not only `new-old`, because old final and new temp may coexist before `os.replace()`.

The exact application contract covers cache-owned payload logical bytes (`st_size`) plus active payload reservations. Directory/inode/control-file/journal/filesystem-allocation overhead is outside the exact byte quota.

## 5. Runtime state model

Self-review clarification: committed-entry state and writer state are separate so replacement can preserve the old committed record while a new temp payload is in flight.

### Committed entry record

Each known final path may have an `EntryRecord` with:

- `state`: `COMMITTED`, `EVICTING`, or `INVALID`
- `size`
- `readers`
- `recency`
- identity/generation token

Semantics:

- `COMMITTED`: valid final payload, accounted, may serve reads; evictable only when `readers == 0`.
- `EVICTING`: admission has exclusively claimed this entry; bytes remain accounted until deletion is confirmed; no new read pin.
- `INVALID`: no longer trusted for serving after load/corruption failure; bytes remain accounted until deletion is confirmed.

### Active writer/reservation record

Separately, a `pending_writes[path] -> Reservation` map records an active writer. At most one active writer may own a path.

For a new path, a reservation may exist without an `EntryRecord`.

For a low-level replacement operation, the old committed `EntryRecord` remains installed and accounted while a distinct reservation owns the replacement temp payload. Normal content-addressed runtime stores do not intentionally rewrite an already-valid committed entry.

All finalize/abort/reconcile operations verify expected entry/reservation identity so stale callbacks cannot double-add or double-subtract bytes.

## 6. Reservation lifecycle

For a normal new entry:

```text
worker starts
  -> inspect committed/pending state
  -> existing: idempotent skip
  -> duplicate writer: normal duplicate-inflight skip
  -> incoming > max: oversized capacity skip
  -> admission
  -> evict if needed
  -> grant reservation
  -> write temp
  -> os.replace(temp, final)
  -> commit reservation
```

Admission must establish:

```text
accounted + reserved + incoming <= max
```

before adding the reservation.

### Commit

After filesystem replacement succeeds:

```text
reserved -= incoming
accounted += final_size
remove pending writer
install COMMITTED entry
```

The short window after `os.replace()` but before logical commit is safe because bytes are still classified as reserved, so accounting is conservative rather than optimistic.

### Abort

On write/replace/cancellation failure after reservation:

```text
cleanup temp if present
reserved -= incoming
remove pending writer
```

Abort must be idempotent.

### Queue cancellation

Reservations are acquired only when a worker task starts, not in `submit_store()`. Queued tasks cancelled before execution therefore cannot leak capacity reservations.

## 7. Existing key and replacement semantics

Normal content-addressed runtime behavior remains idempotent:

- valid committed entry already exists -> skip physical store
- no new reservation
- no temp file
- no extra accounted bytes
- update LRU recency

The lower-level capacity API still supports replacement for deterministic accounting tests and future callers:

- reserve full new size
- old entry remains committed/accounted while the temp exists
- on success: `accounted = accounted - old + new`, `reserved -= new`
- on failure: old entry/accounting unchanged, reservation released

Same-size, larger, and smaller replacement accounting must all be tested.

## 8. Concurrency architecture

Use two synchronization domains.

### `metadata_lock`

Short-held lock protecting:

- `accounted_bytes`
- `reserved_bytes`
- entry index/state
- pending writers/reservation identity
- readers
- LRU recency
- capacity counters/snapshot state

### `admission_lock`

Serializes the capacity decision sequence:

- capacity check
- victim selection/claim
- victim unlink/reconcile
- reservation grant

Global lock ordering:

```text
admission_lock -> metadata_lock
```

Code holding `metadata_lock` must never wait for `admission_lock`.

Large payload read/write/replace operations never hold either capacity lock. Victim `unlink()` runs outside `metadata_lock`, but may remain under admission serialization so a second admission cannot race victim selection/reclaimed capacity.

Result: capacity allocation is serialized; already-admitted payload I/O remains concurrent.

## 9. Concurrent writer rules

### Different keys

Multiple workers may perform payload I/O concurrently after receiving reservations. Because reservations are atomically included in the invariant, two writers cannot both consume the same free capacity.

### Same key

At most one active writer owns a path.

- first writer installs `pending_writes[path]` and reserves bytes
- second writer observes the pending writer and returns a normal duplicate-inflight skip
- no second reservation/temp/eviction
- duplicate-inflight is not a capacity rejection
- if the first writer fails, a later independent store opportunity may retry

## 10. Lookup, load pinning, and promotion race

### Runtime lookup source

In bounded mode, the existing async lookup mechanism may remain, but its membership answer must come from the in-memory capacity state rather than raw `os.path.exists()` as the primary truth.

- `COMMITTED`: HIT candidate
- `EVICTING`: MISS/RETRY, never stable HIT
- `INVALID`: MISS
- no committed entry: MISS, including a new-path pending writer

Keeping the async lookup wrapper preserves existing scheduler/deferred-lookup behavior while removing filesystem existence as authoritative capacity membership.

### Read pin

Before a real load task is queued, `submit_load()` must pin the entry under `metadata_lock`:

```text
require entry is COMMITTED
readers += 1
```

The worker unpins in `finally`.

Eviction eligibility requires:

```text
state == COMMITTED and readers == 0
```

### Lookup-to-submit race

A block may have returned HIT during async lookup but be claimed/removed before the later promotion `submit_load()` call. Therefore `submit_load()` must revalidate and pin.

If revalidation/pin fails:

- do not perform filesystem read I/O
- report the promotion load job as unsuccessful
- let the existing tiering manager call primary `complete_write(..., success=False)` so the reserved/not-ready primary slot is released correctly
- do not treat the missing pin as a successful load

This preserves existing promotion failure semantics and inference fallback while preventing read/eviction races.

## 11. LRU policy

Issue #31 implements deterministic runtime LRU only.

- successful commit receives newest recency
- `touch()` updates committed-entry recency in memory
- no runtime `mtime` updates
- no frequency/TTL/cost/size-aware scoring

Victims are the oldest eligible committed entries.

This is a bounded-safety policy, not #19’s future economic policy.

## 12. Eviction semantics

Victim claim is atomic under `metadata_lock`:

```text
COMMITTED -> EVICTING
```

The victim remains in `accounted_bytes` until the filesystem confirms absence.

### Successful unlink

```text
remove entry
accounted -= victim.size
```

### `ENOENT`

Confirmed absence may be reconciled by removing the tracked entry and reducing accounting, after verifying the same entry/generation is still installed.

### Other unlink failure

- `EVICTING -> COMMITTED`
- `accounted_bytes` unchanged
- runtime admission may try another eligible victim
- if insufficient reclaimable capacity remains, incoming store becomes a capacity skip

Never release capacity merely because unlink was attempted.

## 13. Capacity skip versus I/O failure

Capacity pressure is normal best-effort cache degradation.

Normal skip classes:

- already committed
- duplicate in-flight writer
- oversized entry
- no evictable capacity

Only `oversized` and `no_evictable_capacity` are capacity-rejection metrics; existing/duplicate skips are idempotence/deduplication outcomes.

Oversized entry (`incoming > max_bytes`) is rejected before eviction, reservation, temp creation, or payload write.

True I/O failure includes write/replace/permission/filesystem errors, `ENOSPC`, and `EDQUOT`.

`ENOSPC`/`EDQUOT` may occur below logical `max_bytes`; they roll back temp/reservation state but do not resize or reinterpret the logical quota.

## 14. Store job and KV-event semantics

Internally distinguish at least:

- `COMMITTED`
- `SKIPPED`
- `IO_FAILED`

The public `JobResult` need not expand for #31.

- committed task: normal completion
- normal skip: normal best-effort completion
- real I/O failure: task failure

### Stored events

Capacity-skipped or deduplicated blocks must never be falsely reported as newly stored.

For a job with committed + normal-skipped blocks and no real I/O failure:

- job may complete successfully
- emit `BlockStored` only for the actually committed subset

If all blocks are skipped, emit no stored event.

For any store job containing a real I/O failure, preserve the current conservative behavior:

- `JobResult.success == False`
- no `BlockStored` event for that failed job, even if sibling blocks physically committed

## 15. Load corruption and accounting-aware invalidation

Raw load I/O must not directly delete an owned final file.

On corruption/unreadable failure:

1. mark the entry `INVALID` so no new load pin may be granted
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

On Linux use a primitive such as `fcntl.flock(LOCK_EX | LOCK_NB)`.

- second participating bounded manager for the same data namespace fails fast
- normal close releases lock
- process death releases kernel lock even if the lock file remains
- lock file is control metadata, excluded from payload `max_bytes`

This is **not** cross-process quota coordination. It forbids two independent bounded accounting owners from simultaneously managing the same namespace.

If the required locking operation itself is unavailable/fails, construction fails rather than silently weakening the contract. #31 does not attempt to prove semantics of arbitrary external/non-cooperating filesystem writers.

## 17. Restart recovery and READY gate

Recovery completes synchronously before the tier accepts normal work:

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

No normal store/load/lookup begins while recovery is incomplete.

### Scan boundary

Scan only the current `FileMapper` data namespace, not the whole configured `root_dir` and not other model/config/rank namespaces.

### Artifact classification

1. **Valid final:** expected mapper layout/name and expected size; restore as `COMMITTED`.
2. **Recognized stale temp:** definitely produced by this tier’s temp naming convention; delete before READY. Cleanup failure is fatal.
3. **Recognized corrupt final:** final naming/layout is ours but size invalid; do not serve; delete before READY. Cleanup failure is fatal.
4. **Unknown regular artifact:** do not delete and do not ignore; fail construction with diagnostics.

Empty directories are harmless and need not be aggressively removed.

No reservation survives restart; recovered `reserved_bytes` starts at zero.

## 18. Restart LRU recovery

No persistent LRU DB/WAL/index is introduced.

For valid recovered finals, cold-start ordering is:

```text
(st_mtime_ns, normalized_relative_path)
```

Path tie-break makes equal timestamps deterministic.

After READY, commit/touch use an in-memory monotonic recency sequence. Historical LRU order is approximate across restart, but safety and cold-start determinism are guaranteed.

## 19. Startup over-limit shrink

After scan, if:

```text
accounted_bytes > max_bytes
```

construction synchronously evicts cold-start LRU victims until bounded.

A historical entry larger than the new max is simply a required startup victim.

Startup is stricter than runtime admission:

- success/`ENOENT`: reconcile and continue
- other unlink failure while over limit: fail construction immediately

A tier never announces READY while recovered usage exceeds the configured hard ceiling.

## 20. Logical quota versus physical free space

`max_bytes` remains independent of current filesystem free space.

Startup may inspect/report free space for diagnostics but must not auto-resize the configured quota.

If physical `ENOSPC`/`EDQUOT` occurs below logical max:

- clean temp if possible
- abort reservation
- preserve existing committed accounting
- report real I/O failure
- inference continues through existing cache-failure fallback

## 21. External mutation contract

The ownership lock coordinates participating bounded managers; it cannot prevent arbitrary external processes from modifying the cache directory.

Runtime guarantees cover manager-controlled mutations. Clear external deletion may be reconciled conservatively when observed.

Arbitrary external writes into the owned namespace violate the contract; #31 does not add a filesystem watcher or continuous full scan. Restart recovery detects unknown pollution and fails fast.

## 22. Shutdown

Shutdown should establish:

```text
stop accepting new work
-> finish/cancel queue according to pool contract
-> every active worker reservation commits or aborts
-> reserved_bytes == 0
-> close capacity manager
-> release ownership lock fd
```

A nonzero reservation count after active workers terminate is a correctness bug, not something to silently repair.

## 23. Metrics and observability

Minimum low-cardinality metrics:

- `vllm:kv_offload_fs_capacity_bytes{tier=...}` gauge
- `vllm:kv_offload_fs_accounted_bytes{tier=...}` gauge
- `vllm:kv_offload_fs_reserved_bytes{tier=...}` gauge
- `vllm:kv_offload_fs_evictions{tier=...}` counter
- `vllm:kv_offload_fs_evicted_bytes{tier=...}` counter
- `vllm:kv_offload_fs_capacity_skips{tier=...,reason=...}` counter
- `vllm:kv_offload_fs_eviction_failures{tier=...}` counter

Capacity skip reasons are bounded enums, e.g. `oversized` and `no_evictable_capacity`.

Never label by root path, key/hash, request ID, or temp filename.

### Multiple FS tiers

Existing stats aggregation uses the latest gauge value for an identical metric+label tuple, so multiple unlabeled FS tiers would overwrite each other.

Every FS metric therefore carries deterministic low-cardinality tier identity derived from secondary-tier configuration index, e.g. `fs:0`, `fs:1`.

Do not use `root_dir` and do not overload `cost_model_tier_key` for this identity.

Implementation may add minimal generic tier instance identity plumbing after construction; capacity algorithm/state does not depend on the label value.

A capacity stats snapshot reads capacity/accounted/reserved/counters under one short `metadata_lock` critical section.

### Logs

- normal LRU eviction and existing-key skip: no per-block INFO noise
- capacity pressure: metrics first; warnings rate-limited/aggregated if needed
- real filesystem failures: clear diagnostics
- startup failures: include namespace, configured max, recovered usage, failure category

High-cardinality details belong in logs, not metric labels.

## 24. Code decomposition

### New `vllm/v1/kv_offload/tiering/fs/capacity.py`

Owns:

- `EntryRecord`
- `Reservation`
- committed-entry and pending-writer state
- accounted/reserved bytes
- LRU/recency
- read pin/unpin
- admission and victim selection
- startup scan/recovery/shrink
- ownership lock
- capacity counters and consistent snapshots

The hard-capacity state machine should be directly unit-testable without real KV memory I/O.

### `fs/manager.py`

Remains tier orchestration:

- validate/accept `max_bytes`
- build `FileMapper`
- initialize capacity manager before normal work
- bounded async lookup integration
- `touch()`
- worker-side store/load wrappers
- per-key store outcomes
- job/event aggregation
- stats integration

### `fs/io.py`

Raw filesystem primitives only. Load helpers stop owning final-file deletion/accounting semantics.

### `file_mapper.py`

At most add a small explicit accessor for the data namespace root. Capacity policy does not belong in `FileMapper`.

### Generic tier plumbing

If required for metrics, add only minimal deterministic secondary-tier instance identity based on configured index.

## 25. TDD acceptance matrix

Implementation follows failing-test-first development. Minimum coverage:

1. missing/zero/negative/wrong-type `max_bytes` fails
2. positive `max_bytes < block_size` is valid
3. basic committed accounting
4. existing-key store does not double count
5. reservation is granted before payload write
6. commit converts reserved to committed
7. failure abort releases reservation and cleans temp
8. same/larger/smaller replacement accounting
9. replacement reserves full new size while old remains accounted
10. oversized entry skips without eviction/temp/reservation
11. concurrent writers cannot exceed the invariant
12. same-key concurrent store produces only one real writer
13. deterministic LRU victim order
14. `touch()` updates recency
15. active reader cannot be evicted
16. lookup-HIT-to-submit-load race: failed pin reports unsuccessful promotion without I/O
17. eviction unlink success updates accounting
18. eviction `ENOENT` reconciles safely
19. other unlink failure does not free bytes
20. write/replace/ENOSPC-style failure rolls back safely
21. load corruption enters INVALID and remains accounted until deletion
22. capacity-skipped key never emits false stored event
23. mixed committed + normal-skip job emits only committed subset
24. real partial I/O failure preserves current no-stored-event contract
25. restart scan rebuilds accounting
26. recognized temp cleanup succeeds/fails correctly
27. corrupt recognized final cleanup succeeds/fails correctly
28. unknown artifact fails without deletion
29. smaller restart max synchronously evicts to bound
30. oversized historical entry is evicted during recovery
31. second bounded namespace owner fails fast
32. ownership releases on shutdown/process exit semantics
33. multiple FS tier metrics do not overwrite gauges
34. shutdown ends with zero active reservations
35. capacity rejection preserves cascade completion/primary-pin lifecycle
36. scheduler-facing submission remains lightweight/non-blocking
37. bounded lookup uses authoritative in-memory membership while preserving async/deferred interface behavior

## 26. Real-filesystem validation

Focused unit tests are necessary but not sufficient. Run a real filesystem-backed smoke with a deliberately small quota.

Exercise real final/temp files, including a controlled pause after temp payload completion and before `os.replace()` so the peak state is inspectable while temp bytes exist.

Required sequence:

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

`du` may be auxiliary evidence, but allocation rounding, inode metadata, journals, overlay backing, and unrelated disk consumers are not part of the exact logical payload-byte invariant.

Do not intentionally fill the Pod disk to create ENOSPC; use controlled fault injection for rollback testing.

## 27. Final validation evidence and PR gate

Final Issue #31 evidence records:

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

## 28. Safety checklist mapping

This design resolves the required safety questions:

1. authoritative accounting source: startup scan, then in-memory capacity manager
2. committed vs reserved: Section 4
3. temp files count: full-size reservation before write
4. reservation lifecycle: Section 6
5. atomic concurrent reservation: admission serialization + metadata lock
6. capacity locks during real I/O: no payload I/O under capacity locks
7. victim selection/reservation coordination: one admission lock, atomic victim claim
8. overwrite delta accounting: old remains accounted; full-new-size reservation; atomic old/new swap after replace
9. failed/cancelled rollback: idempotent abort; worker-side reservation
10. oversized single entry: pre-I/O capacity skip
11. restart usage rebuild: owned namespace scan
12. restart over-limit: synchronous deterministic shrink
13. orphan/temp handling: recognized cleanup; unknown fail-fast
14. LRU recovery: `mtime_ns` + path deterministic cold start
15. lookup/read vs eviction: in-memory membership, submit-load revalidation, read pins
16. eviction unlink failure: release bytes only after confirmed absence
17. physical ENOSPC vs logical max: distinct failure classes
18. metric integration: tier-local low-cardinality stats with deterministic tier identity
19. disabled/default behavior: explicit FS requires positive `max_bytes`; no FS tier remains unchanged
20. metadata versioning: scan-based recovery is sufficient; no new persistent schema

## 29. Self-review findings incorporated

The design self-review made two normative clarifications before implementation planning:

1. `WRITING` is represented as a separate pending reservation/writer record, not as a committed-entry state. This is required so low-level replacement can keep the old final accounted while the new temp is in flight.
2. A lookup HIT is not sufficient to protect a later promotion read. `submit_load()` must revalidate/pin; a failed pin reports an unsuccessful promotion without filesystem I/O so existing tiering completion logic can release the reserved primary slot with `success=False`.

No production code was changed during this design phase.

## 30. Implementation process gate

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

No production implementation begins before the design spec has passed user review and the implementation plan has been written.