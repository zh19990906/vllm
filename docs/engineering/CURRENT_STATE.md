# Current Engineering State

Observed: **2026-08-14**.

This file is intentionally mutable. Verify GitHub issue, PR, branch, and commit metadata
before acting if this file is older than the current session.

## Repository state

- Repository: `zh19990906/vllm`
- Default branch: `main`
- This refresh was prepared from
  `main@ceb895f9abd916524ca7178aac3f42dc230a48af`, after the PR #27-#29 documentation
  consolidation completed.
- Do not treat that snapshot SHA as a permanently current branch head. Updating this file
  creates newer commits; live GitHub metadata remains authoritative for the actual head.
- PR #27 preserved the useful point-in-time history from superseded PR #22 without merging
  its stale mutable roadmap state.
- PR #28 refreshed this file to the Issue #15-era roadmap, and PR #29 corrected the
  snapshot wording so this mutable document does not recursively claim its own merge as a
  permanently current head.
- GitHub is the authoritative remote. The observed development workflow synchronizes
  GitHub branches/tags to a Gitee mirror used by Pod workspaces.
- Pod workspaces are for build/test/hardware validation and are not an authoritative
  GitHub write path.

## Project objective

The long-term objective is an adaptive hierarchical KV cache system that can make runtime
decisions from measured cost instead of treating every reusable KV as an unconditional
restore win.

The core decision chain is:

```text
measure restore vs recompute
  -> calibrate shadow prediction
  -> validate generalization
  -> enable active restore/recompute choice
  -> build multi-tier placement
  -> build adaptive admission/eviction
```

## Roadmap status

Parent roadmap: **Issue #9**.

### Completed work

- **Issue #10**: workload fairness and token-length convergence, completed by PR #3.
- **Issue #11**: real eviction/lower-tier restore benchmark, completed by PR #5.
- **Issue #12**: shadow restore/recompute cost model, completed by PR #7.
- **Issue #13**: systematic restore-vs-recompute crossover measurement, completed by
  PR #25.
- **Issue #14**: calibration of the shadow cost model from real measurements, completed by
  PR #26.
- **Issue #15**: generalization validation of the frozen shadow cost model, completed by
  PR #32.
- **Issue #20**: repository `benchmarks/cache` pre-commit/static baseline cleanup,
  completed.

### Current P0 core work

**Issue #31 is the current P0 safety gate before Issue #16 active enforcement.**

Issue #31 adds a bounded filesystem KV-cache tier with a required positive
`max_bytes`, committed-plus-reserved hard-cap accounting, deterministic LRU eviction,
read pins, restart recovery, exclusive namespace ownership, shutdown lifecycle
invariants, and low-cardinality capacity metrics.

The formal filesystem validation was produced from implementation head
`949beed012b57281ae8eadd63cc8a674fb1975e0`. The latest code-bearing delivery
head observed before this status refresh is
`451a37fadbfc15051ea8438e7d02483b1cc6602d`. It retains the validated behavior,
repository quality-gate remediation, final validation-document changes, and the
PR #34 deterministic ShellCheck CI fix from current `main`. This mutable status
refresh may create a newer documentation-only PR head; live PR metadata remains
authoritative for the current head. Formal filesystem evidence is recorded in:

[`validation/2026-08-12-issue31-filesystem-hard-capacity-validation.md`](validation/2026-08-12-issue31-filesystem-hard-capacity-validation.md).

Before this status refresh, Draft PR #33's latest code-bearing head was
`451a37fadbfc15051ea8438e7d02483b1cc6602d`, based on
`main@5ab0016765bfdbd2d45575908a096fbf675b3a84`. Authoritative GitHub
pre-commit run #208 (`31761549792`) passed on that head. The PR remains Draft
and unmerged. Any newer documentation-only head must also pass authoritative CI
before merge; live GitHub PR metadata is authoritative for that check.

### Next P0 stage

**Issue #16 remains next, but it is blocked until Issue #31 is merged and closed.**

When Issue #16 resumes, its active restore/recompute design must still respect the bounded
eligibility map produced by Issue #15; Issue #31 changes filesystem-cache safety and does
not widen Issue #15 eligibility.

See:
[`handoffs/2026-08-11-issue16-active-decision-handoff.md`](handoffs/2026-08-11-issue16-active-decision-handoff.md).

### Parallel work

- **Issue #17**: NUMA topology discovery and CPU KV placement foundations, open.
- **Issue #21**: permanent fix for the fake `vllm` fixture nested-newline escaping defect,
  open.

Issue #17 can proceed in parallel with Issue #31. Issue #21 is maintenance and should not
block the main restore/recompute research path unless it directly blocks a required test.

### Later work

- **Issue #18**: GPU/CPU/NVMe multi-tier KV placement, open; depends on active
  restore/recompute behavior and NUMA foundations.
- **Issue #19**: adaptive online KV admission/eviction, open; follows multi-tier placement
  and active restore/recompute decisions.

## Recently completed core evidence

### Issue #13 / PR #25: crossover measurement

- PR #25 merge commit:
  `3295d83ff76ec8792942e6ec7faf9adbb4afe39e`
- Baseline environment: one model, one GPU Pod, concurrency=1.
- CPU-primary restore showed a P50 crossover bracket at 192-216 requested prompt tokens.
- Tiered-filesystem restore showed no P50 crossover in the measured 256-4096 requested
  token range; recompute remained faster throughout that measured range.
- Restore provenance and structured calibration input were recorded for the next stage.
- The filesystem evidence is lower-tier/tiered-fs on the container's local
  overlay-backed filesystem; it is not evidence of physical NVMe performance.

See:
[`validation/2026-08-10-issue13-restore-recompute-crossover.md`](validation/2026-08-10-issue13-restore-recompute-crossover.md).

### Issue #14 / PR #26: shadow cost-model calibration

- PR #26 merge commit:
  `ddbe6650778c78ef01dec9ecbb424fa1a4bcf553`
- Primary acceptance metric: P95.
- Before calibration: 13/14 correct decisions and principal P95 macro-MAPE 34.021%.
- After calibration: 14/14 correct decisions and principal P95 macro-MAPE 0.090%.
- Calibration uses runtime actual `external_tokens`, not requested prompt tokens.
- The Phase 1 result remains shadow-only and does not modify the active restore/recompute
  execution path.
- Repository-wide GitHub pre-commit passed on the final PR head before merge.

See:
[`validation/2026-08-10-issue14-shadow-cost-model-calibration.md`](validation/2026-08-10-issue14-shadow-cost-model-calibration.md).

### PR #27-#29: retire stale roadmap state and restore durable documentation

PR #27 preserved only these point-in-time records from superseded PR #22:

- [`history/2026-08-10-pr5-finalization-and-roadmap-transition.md`](history/2026-08-10-pr5-finalization-and-roadmap-transition.md)
- [`handoffs/2026-08-10-post-pr5-roadmap.md`](handoffs/2026-08-10-post-pr5-roadmap.md)

Those files are historical snapshots. They intentionally retain their 2026-08-10 framing
and must not be read as the current roadmap. PR #22 itself was closed without merge.

PR #28 refreshed this mutable status file to the Issue #15-era roadmap. PR #29 corrected
the refresh metadata so it records a preparation snapshot rather than a self-invalidating
claim about the permanently current `main` head.

The consolidated progress record is:

[`history/2026-08-10-to-2026-08-11-issue13-14-and-roadmap-consolidation.md`](history/2026-08-10-to-2026-08-11-issue13-14-and-roadmap-consolidation.md).

## Current objective: finalize Issue #31 filesystem hard-capacity delivery

Issue #31 has completed local implementation, local filesystem validation, repository
publication, and authoritative GitHub pre-commit CI. Remaining work is final review
and explicit merge authorization, not additional local feature expansion.

Current local evidence establishes:

1. configured filesystem size is a hard logical ceiling;
2. committed plus reserved bytes remained bounded at the formal temp-file peak;
3. runtime deterministic LRU eviction occurred with real files;
4. both `oversized` and `no_evictable_capacity` skips were observed;
5. restart rebuilt usage and a smaller maximum synchronously shrank the namespace;
6. a second capacity owner for the same namespace was rejected;
7. shutdown joins workers before releasing namespace ownership;
8. filesystem metrics use deterministic per-instance identities;
9. physical filesystem free space is diagnostic only and does not redefine quota;
10. the environment is described as filesystem/container-local, not physical NVMe.

The accepted formal evidence is:

[`validation/2026-08-12-issue31-filesystem-hard-capacity-validation.md`](validation/2026-08-12-issue31-filesystem-hard-capacity-validation.md)

The machine-readable companion is:

[`validation/2026-08-12-issue31-filesystem-hard-capacity-validation.json`](validation/2026-08-12-issue31-filesystem-hard-capacity-validation.json)

## Current continuation record

Live GitHub `main` was observed at
`5ab0016765bfdbd2d45575908a096fbf675b3a84`, the merge commit for PR #34.

Before this status refresh, Draft PR #33's latest code-bearing head was
`451a37fadbfc15051ea8438e7d02483b1cc6602d`, whose parents are the prior
Issue #31 delivery head and `main@5ab0016765bfdbd2d45575908a096fbf675b3a84`.

Authoritative GitHub pre-commit run #208 (`31761549792`) passed on that
code-bearing head. This mutable file intentionally does not recursively encode
the SHA or CI run of its own documentation-only refresh. Live PR #33 metadata
is authoritative for the current head and the latest authoritative CI result.
Merge remains gated on authoritative CI for whatever head is current at merge
time, final review, and explicit user authorization.

Pytest is unavailable in the Pod, and no pytest CI job was observed for PR #33.
Focused Issue #31 unittest, smoke-contract, mypy, repository policy, compile,
and formal real-filesystem evidence remain recorded in the validation artifact.

Issue #16 remains blocked until Issue #31 is merged and closed.

## Important interpretation rules

- Requested prompt tokens and runtime actual external KV tokens are not interchangeable.
  Cost-model calibration and evaluation should use the runtime quantity used by the model.
- Do not call a local filesystem tier "NVMe" unless the storage provenance actually proves
  physical NVMe behavior.
- Do not treat a shadow decision change as proof that the active runtime path changed.
- Do not repeat the Issue #13 baseline sweep merely to reproduce archived numbers; rerun
  expensive hardware experiments only when Issue #15 introduces a new condition or a new
  hypothesis.
- A current all-files CI failure should be classified against current `main` before being
  attributed to a feature branch. Issue #20's old cache static baseline is completed.
- The fake `vllm` fixture defect is still open as Issue #21 and should be treated as a known
  maintenance item until it is permanently resolved.

## Source-of-truth order

When records disagree, prefer:

1. current GitHub issue, PR, branch, and commit metadata;
2. current repository source and checked-in structured artifacts;
3. raw benchmark/hardware artifacts;
4. validation documents;
5. history and handoff snapshots.

The point-in-time handoffs under `docs/engineering/handoffs/` are useful provenance, but
this file is the mutable repository summary that should be updated as the roadmap advances.
