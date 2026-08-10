# Engineering History System Design

## Purpose

The repository needs a durable record of engineering decisions, bug investigations,
hardware validation, and handoff state that survives individual chat sessions and local
worktrees. This information does not belong in the top-level README because it is
operational history rather than user-facing project introduction.

The system must let a future developer answer four different questions without reading
chat logs:

1. What is true now?
2. What happened historically, and why?
3. Has this symptom happened before, and what was the root cause?
4. What has already been validated, especially on real hardware?

## Design principles

### Repository state is distinct from history

`docs/engineering/CURRENT_STATE.md` is mutable and represents the latest known project
state. It should stay short enough to read at the start of every engineering session.

Historical documents are snapshots. Once an event is accurately recorded, later work
should add a new history or handoff entry instead of silently rewriting the old event to
match current knowledge.

### Plans and specs are not history

Existing files under `docs/superpowers/specs/` and `docs/superpowers/plans/` describe
intent: what was designed and what was planned.

Files under `docs/engineering/` describe observed reality: what actually happened,
which hypotheses failed, what hardware measured, and what state a later session should
inherit.

A completed implementation can legitimately differ from its original plan. The history
must preserve that difference instead of editing the plan retroactively.

### Incidents are searchable regression memory

A bug or failure with meaningful recurrence risk gets one focused document under
`docs/engineering/incidents/`. Each incident records symptoms, root cause, reproduction,
correct fix or workaround, false conclusions to avoid, regression protection, and last
verification.

This is intentionally redundant with chronological history. History explains sequence;
incident documents optimize for symptom-driven search.

### Validation records preserve expensive evidence

Hardware and integration evidence belongs under `docs/engineering/validation/`.
Validation documents distinguish seed profiles, measured data, runtime observations,
execution-path invariants, and artifact locations. They must never imply that a test ran
when it did not.

Raw benchmark logs and large result directories remain external artifacts. The repository
records the commands, important measurements, conclusions, and artifact paths instead of
checking large generated data into Git.

### Handoffs are bounded snapshots

`docs/engineering/handoffs/` records enough operational context for another session to
continue work immediately. A handoff may contain transient paths and branch heads, but it
must label them with the observation date and tell the next session to verify them before
acting.

Handoffs should not grow indefinitely. Stable knowledge moves into history, incidents,
and validation documents; the handoff contains only current continuation context.

## Directory structure

```text
docs/engineering/
├── README.md
├── CURRENT_STATE.md
├── history/
│   ├── 2026-08-07-pr3-cache-workload-fairness-convergence.md
│   ├── 2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md
│   └── 2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md
├── incidents/
│   ├── cache-workload-tokenizer-fairness-and-convergence.md
│   ├── fake-vllm-newline-fixture.md
│   ├── native-wheel-exact-overlay.md
│   ├── transient-metrics-before-connection-refused.md
│   └── cache-benchmark-ci-baseline.md
├── validation/
│   ├── cache-crossover-baseline.md
│   └── pr7-shadow-cost-model-hardware-validation.md
└── handoffs/
    └── 2026-08-10-pr5-current-handoff.md
```

## Document contracts

### `README.md`

Defines the taxonomy, update rules, naming conventions, and recommended reading order for
humans and agents.

### `CURRENT_STATE.md`

Contains repository head, active PRs, completed PRs, known baseline problems, current
objective, do-not-repeat guidance, and the next exact workstream. It should link to the
detailed documents rather than duplicate them.

### `history/*.md`

Records chronology, design changes, implementation outcomes, failed approaches, final
state, and related PR/commit identifiers.

### `incidents/*.md`

Uses a stable shape:

- status and dates;
- symptom;
- impact;
- root cause;
- reproduction or diagnostic evidence;
- correct resolution or workaround;
- false conclusions to avoid;
- regression protection;
- related files, PRs, and commits.

### `validation/*.md`

Uses a stable shape:

- validation question;
- environment and code identity;
- method;
- measured evidence;
- pass/fail criteria;
- result;
- limitations and artifact locations.

### `handoffs/*.md`

Uses a stable shape:

- read-first warning;
- current goal;
- authoritative remote state;
- observed local workspace state;
- validated facts;
- known blockers;
- do-not-do constraints;
- exact next actions.

## Source-of-truth rules

When sources disagree, use this order:

1. Current GitHub branch and PR metadata for remote state.
2. Current repository files and diffs for code state.
3. Hardware result artifacts for measured behavior.
4. Engineering history and validation documents for summarized evidence.
5. Handoff snapshots for operational context.
6. Chat logs only to recover missing historical detail before it is committed.

Older history must not override newer remote state. A historical document may say that a
PR was draft on a given date; `CURRENT_STATE.md` must reflect whether it later merged.

## Repository synchronization constraint

The development workflow may use a Gitee mirror for Pod-side fetch and execution. Pod
workspaces must not be treated as a GitHub write path. GitHub branch, PR, and merge writes
should be performed through an authorized GitHub-side mechanism, followed by mirror sync
and Pod fetch for validation.

This constraint belongs in handoff documentation because violating it can create local
commits that cannot become the authoritative remote history.

## Initial content scope

The first version records the recent KV cache / KV offload development sequence:

- PR #3 workload fairness and tokenizer convergence fix;
- PR #5 eviction/restore pressure benchmark development and current draft state;
- filesystem versus recompute crossover measurements;
- PR #7 shadow cost model design, integration, hardware validation, and merge;
- exact source-over-wheel runtime validation;
- transient metrics collection investigation;
- fake `vllm` newline fixture failure;
- repository-wide `benchmarks/cache` Ruff/format/SPDX/markdownlint baseline;
- the current PR #5 continuation handoff.

## Success criteria

The system is complete when a fresh session can start from
`docs/engineering/README.md` and `docs/engineering/CURRENT_STATE.md`, locate detailed
root-cause and validation evidence without chat history, and continue PR #5 without
repeating already completed hardware or debugging work.
