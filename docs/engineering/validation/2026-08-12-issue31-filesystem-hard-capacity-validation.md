# Issue #31: filesystem KV cache hard-capacity validation

## Executive conclusion

Issue #31 is **locally validated with authoritative GitHub CI still pending**.

The filesystem KV-cache tier now treats configured `max_bytes` as a hard logical
capacity ceiling and accounts both committed bytes and in-flight reservations against
that ceiling. The formal real-filesystem smoke observed a maximum
`accounted + reserved` value of **12288 bytes**
against a configured maximum of **12288 bytes** with
**4096-byte blocks**.

The formal smoke also observed real temp files while reservation bytes were held,
runtime LRU eviction, both capacity-skip reasons, restart accounting rebuild,
synchronous smaller-max startup shrink, and rejection of a second namespace owner.

This is filesystem evidence only. It is **not** evidence of physical NVMe performance.
The implementation has not yet been published from the Pod-local implementation head
to the GitHub Issue #31 branch, no Draft PR exists yet, and authoritative GitHub CI
has not run.

## Scope and provenance

- Issue: #31, `[P0] filesystem KV cache hard capacity and eviction`.
- GitHub repository: `zh19990906/vllm`.
- Live target base observed before validation:
  `main@c4d9fce61ec5a8eadc24dab8698eca7705d005bf`.
- Pod-local implementation head before this validation-document commit:
  `949beed012b57281ae8eadd63cc8a674fb1975e0`.
- GitHub branch `agent/issue31-fs-hard-capacity` still pointed to
  `eea0ff4b16711693b6f9945a4a808916990442ee` before publication.
- Approved design:
  `docs/superpowers/specs/2026-08-12-filesystem-kv-cache-hard-capacity-design.md`.
- Approved implementation plan:
  `docs/superpowers/plans/2026-08-12-issue31-filesystem-kv-cache-hard-capacity.md`.
- Formal smoke evidence:
  `/code/results/cache/issue31-fs-capacity-20260812-run3/smoke.json`.
- Formal filesystem provenance label: `filesystem`.
- Pytest is unavailable in the Pod; pytest-only coverage remains an authoritative
  GitHub-CI gate.

## Fresh focused verification

| Verification | Result |
| --- | --- |
| filesystem capacity unittest | 40/40 passed |
| Issue #31 config unittest | 4/4 passed |
| real-filesystem smoke contract unittest | 1/1 passed |
| compileall | exit 0 |
| targeted Ruff | exit 0 |
| `git diff --check` | exit 0 |
| Pod pytest | unavailable; deferred to GitHub CI |

These commands were rerun from the final Task 10 implementation head rather than
inferred from earlier task-level results.

## Formal real-filesystem hard-cap smoke

| Quantity | Observed |
| --- | ---: |
| configured max | 12288 bytes |
| block size | 4096 bytes |
| peak accounted | 12288 bytes |
| peak reserved | 8192 bytes |
| peak accounted + reserved | 12288 bytes |
| runtime eviction count | 3 |
| runtime evicted bytes | 12288 bytes |
| restart recovered bytes | 12288 bytes |
| startup shrink evictions | 1 |
| startup shrink evicted bytes | 4096 bytes |
| final payload bytes | 8192 bytes |

The hard-cap invariant held at the observed peak:

`accounted_bytes + reserved_bytes <= max_bytes`

with `12288 <= 12288`.

## Temp-file peak and physical/logical distinction

The smoke created real temp files after reservation and before `os.replace`.
At the temp peak, reservation bytes remained included in the logical quota and the
combined logical usage stayed bounded.

Physical filesystem free-space information was recorded only as a diagnostic:

- total: 1056507072512 bytes;
- used: 532254969856 bytes;
- free: 481088946176 bytes.

Those values do not redefine `max_bytes`. Physical `ENOSPC` or `EDQUOT` can therefore
occur before the logical ceiling and remain true I/O failures rather than capacity
admission skips.

## Runtime LRU eviction

Runtime eviction was observed with real files.

The smoke first filled the configured capacity, touched one committed entry to advance
its in-memory recency, then admitted another block. The oldest eligible committed entry
was unlinked while the touched entry was retained.

Observed runtime eviction counters:

- evictions: 3;
- evicted bytes: 12288.

Read-pinned entries are not eligible eviction victims; the focused capacity suite covers
the pin/release state machine and failed-unlink conservative accounting.

## Capacity skips

Both fixed low-cardinality capacity-skip reasons were observed:

- `oversized`: 1;
- `no_evictable_capacity`:
  1.

These are normal cache-write degradation outcomes. They do not reserve bytes, do not
create a successful stored event for the skipped key, and do not change the inference
scheduler into an error path.

True filesystem I/O failures remain failed asynchronous cache jobs rather than being
misclassified as logical-capacity skips.

## Restart recovery and startup shrink

Restart with the same maximum synchronously rebuilt accounting from recognized final
files and recovered **12288 bytes**.

Restart with a smaller maximum synchronously evicted old entries before construction
returned READY:

- startup eviction count: 1;
- startup evicted bytes: 4096;
- final payload after shrink: 8192 bytes.

Startup recovery also has focused unit coverage for recognized temps, wrong-size
recognized finals, unknown artifacts, deterministic `(mtime_ns, relative_path)` ordering,
and fatal cleanup/unlink failures. Unknown artifacts are not silently deleted.

## Ownership and shutdown lifecycle

A second bounded capacity manager for the same namespace was rejected while the first
owner was alive.

Task-level lifecycle coverage additionally proved that:

- only an active worker owns a reservation;
- queued store tasks own none before worker start;
- shutdown stops lookup work, cancels/joins the I/O pool, then releases capacity
  ownership;
- normal shutdown leaves no ordinary pending reservation;
- surviving orphan-temp cleanup failure remains conservatively represented instead of
  being silently subtracted.

The fresh 40-test focused suite includes these lifecycle regressions.

## Observability and multiple filesystem tiers

Filesystem capacity metrics expose:

- configured capacity;
- accounted bytes;
- reserved bytes;
- evictions;
- evicted bytes;
- capacity skips with fixed reason labels;
- eviction failures.

Runtime tier identity is separate from the cost-model tier key. Multiple filesystem
instances receive deterministic identities such as `fs:0` and `fs:1`, preventing metric
label collision without changing Issue #15 cost-model semantics.

Startup shrink counters remain visible on the first stats poll; subsequent polls emit
positive counter deltas only.

## Event, inference, and cascade correctness

Capacity admission failure is intentionally a normal cache degradation path:

- capacity-skipped keys are omitted from stored-event emission;
- mixed commit/skip jobs emit only the committed subset;
- true I/O failure retains failed-job semantics;
- lookup/load revalidate committed capacity state before raw reads;
- cache write inability does not enable or modify active restore/recompute scheduling.

Issue #31 therefore changes bounded filesystem-cache safety, not the Issue #16
restore-vs-recompute decision policy.

No claim is made here that a new end-to-end model-inference benchmark was added specifically
for Issue #31. The correctness claim is limited to the cache-manager/state-machine behavior
covered locally plus the existing inference architecture; authoritative repository CI is
still required before merge.

## Evidence-run history

Two formal evidence directories were intentionally preserved as aborted attempts:

1. `issue31-fs-capacity-20260812`: created before a shell-safe command correction and
   contains no successful `smoke.json`;
2. `issue31-fs-capacity-20260812-run2`: exposed direct-CLI import-path behavior and did
   not produce the accepted evidence.

The accepted formal evidence is `issue31-fs-capacity-20260812-run3/smoke.json`.
No prior formal evidence directory was deleted, overwritten, or reused.

## Remaining non-goals and roadmap gate

Issue #31 does not implement:

- a cross-process quota coordinator;
- Issue #19 cost-aware admission/eviction;
- Issue #16 active restore/recompute enforcement;
- Issue #18 multi-tier placement policy;
- adaptive capacity recommendation from filesystem free space;
- a physical NVMe performance claim.

**Issue #16 remains blocked until Issue #31 is published, passes authoritative GitHub CI,
is merged, and is closed.**

## CI and merge status

Local status: `locally_validated_ci_pending`.

Authoritative GitHub CI has not yet run on the implementation because the Pod-local
implementation commits have not yet been published to the GitHub Issue #31 branch.

The next delivery stage is to publish the implementation branch through the authoritative
GitHub write path, open a Draft PR with `Closes #31`, run authoritative CI, review the
result, and merge only after explicit user authorization.
