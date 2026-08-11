# Engineering Documentation

This directory is the durable engineering memory for the KV cache and KV offload work in
this repository. It is intended for maintainers, reviewers, and agentic development
sessions that need to continue work without relying on chat history.

## Read this first

A new engineering session should read, in order:

1. [`CURRENT_STATE.md`](CURRENT_STATE.md) for the latest known remote state and active
   work.
2. The handoff linked from `CURRENT_STATE.md` for exact continuation steps.
3. Any incident documents that match the current symptom.
4. Validation documents before repeating expensive GPU or integration work.
5. Historical records when the reason behind a design or branch boundary is unclear.

## Document types

### `CURRENT_STATE.md`

Mutable summary of what is true now. Keep it compact. It should contain active PRs,
recently merged work, known blockers, source-of-truth rules, and the next workstream.

When state changes, update this file rather than editing old historical snapshots to make
them look current.

### `history/`

Chronological engineering record. These documents explain how a change evolved, including
requirements that changed, failed approaches, implementation details, validation, and the
final or currently observed outcome.

History answers: **what happened and why?**

### `incidents/`

Symptom-oriented debugging memory. A recurring or expensive failure gets one focused
record with root cause, reproduction evidence, resolution or workaround, false conclusions
to avoid, and regression protection.

Incidents answer: **have we seen this failure before?**

### `validation/`

Evidence from benchmarks, real hardware, native-wheel integration, or other expensive
checks. Large raw result directories stay outside Git; these documents record commands,
important measurements, conclusions, and artifact locations.

Validation answers: **what have we actually proved?**

### `handoffs/`

Point-in-time continuation context for another session. Handoffs can contain transient
branch heads, local paths, and execution constraints, but every transient fact must be
dated and re-verified before use.

Handoffs answer: **what should the next session do next?**

## Relationship to Superpowers specs and plans

Files under `docs/superpowers/specs/` and `docs/superpowers/plans/` are design intent and
implementation plans. Do not rewrite an old plan to make it match later discoveries.

This directory records observed reality. A production implementation or hardware result
may legitimately differ from the original plan; that difference is useful engineering
history and should remain visible.

## Source-of-truth order

When records disagree, prefer:

1. current GitHub PR, branch, and commit metadata;
2. current repository files and diffs;
3. raw hardware or benchmark artifacts;
4. validation and history documents here;
5. handoff snapshots;
6. chat logs only as a recovery source before missing facts are committed here.

A historical statement is not automatically stale or wrong just because later state
changed. It should be read with its date.

## Writing rules

- Use ISO dates in filenames: `YYYY-MM-DD` or an explicit date range.
- Prefer one incident per independently searchable root cause.
- Preserve exact commit SHAs, PR numbers, test counts, and benchmark numbers when known.
- Clearly label expected behavior versus observed behavior.
- Clearly label validation-only profiles and machine-specific paths.
- Never claim a test passed if it was not executed.
- Do not commit large generated logs, result directories, KV cache files, or secrets.
- Link from a handoff to stable incident/history/validation documents instead of copying
  thousands of lines into one ever-growing file.
- When a problem is disproved, record the negative conclusion so future sessions do not
  repeat the same investigation.

## Repository synchronization workflow

Some validation work uses a Gitee mirror and Pod-side workspaces. Pod workspaces are a
read/test environment, not an authoritative GitHub write path. GitHub branch, PR, and
merge mutations should use an authorized GitHub-side mechanism; the resulting GitHub state
can then be synchronized to Gitee and fetched by the Pod.

This distinction matters because a valid local commit is not useful as authoritative
history if it cannot be delivered to the GitHub branch that owns the PR.

## Current index

### History

- [`2026-08-07-pr3-cache-workload-fairness-convergence.md`](history/2026-08-07-pr3-cache-workload-fairness-convergence.md)
- [`2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md`](history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md)
- [`2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md`](history/2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md)
- [`2026-08-10-pr5-finalization-and-roadmap-transition.md`](history/2026-08-10-pr5-finalization-and-roadmap-transition.md)
- [`2026-08-10-to-2026-08-11-issue13-14-and-roadmap-consolidation.md`](history/2026-08-10-to-2026-08-11-issue13-14-and-roadmap-consolidation.md)

### Incidents

- [`cache-workload-tokenizer-fairness-and-convergence.md`](incidents/cache-workload-tokenizer-fairness-and-convergence.md)
- [`fake-vllm-newline-fixture.md`](incidents/fake-vllm-newline-fixture.md)
- [`native-wheel-exact-overlay.md`](incidents/native-wheel-exact-overlay.md)
- [`transient-metrics-before-connection-refused.md`](incidents/transient-metrics-before-connection-refused.md)
- [`cache-benchmark-ci-baseline.md`](incidents/cache-benchmark-ci-baseline.md)

### Validation

- [`cache-crossover-baseline.md`](validation/cache-crossover-baseline.md)
- [`pr7-shadow-cost-model-hardware-validation.md`](validation/pr7-shadow-cost-model-hardware-validation.md)
- [`2026-08-10-issue13-restore-recompute-crossover.md`](validation/2026-08-10-issue13-restore-recompute-crossover.md)
- [`2026-08-10-issue14-shadow-cost-model-calibration.md`](validation/2026-08-10-issue14-shadow-cost-model-calibration.md)

### Current handoff

- [`2026-08-11-issue15-generalization-handoff.md`](handoffs/2026-08-11-issue15-generalization-handoff.md)

### Historical handoffs

- [`2026-08-10-pr5-current-handoff.md`](handoffs/2026-08-10-pr5-current-handoff.md)
- [`2026-08-10-post-pr5-roadmap.md`](handoffs/2026-08-10-post-pr5-roadmap.md)
