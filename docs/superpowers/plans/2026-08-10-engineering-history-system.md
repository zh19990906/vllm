# Engineering History System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Add a durable engineering-history system that records current state, project
history, recurring incidents, hardware validation, and session handoffs for the KV cache
and KV offload work.

**Architecture:** Keep mutable current state separate from append-oriented historical
records. Preserve intent in existing Superpowers specs/plans, while storing observed
engineering reality under `docs/engineering/` with focused history, incident, validation,
and handoff documents.

**Tech Stack:** Markdown, Git, GitHub pull requests, existing repository markdownlint and
pre-commit tooling.

## Global Constraints

- Do not put detailed engineering chronology into the top-level README.
- Do not check large generated benchmark results or machine-local caches into Git.
- Do not rewrite old plans to make them look consistent with later discoveries.
- Do not claim a test or hardware validation passed unless evidence exists.
- Current GitHub metadata overrides stale handoff or chat state.
- Pod-side workspaces are not an authoritative GitHub write path.
- The initial documentation must be useful without access to the originating chat.

---

### Task 1: Establish the engineering-documentation contract

**Files:**

- Create: `docs/engineering/README.md`
- Create: `docs/engineering/CURRENT_STATE.md`

**Interfaces:**

- Consumes: the approved design in
  `docs/superpowers/specs/2026-08-10-engineering-history-system-design.md`.
- Produces: the entry point and mutable state document used by every future handoff.

- [ ] **Step 1: Write the taxonomy and reading order**

Document the purpose of `history/`, `incidents/`, `validation/`, and `handoffs/`, plus the
rule that specs/plans describe intent while engineering documents describe observed
reality.

- [ ] **Step 2: Write current remote state**

Record current `main`, PR #3, PR #5, and PR #7 state using GitHub metadata. Include the
known cache benchmark static-check baseline and the next PR #5 workstream.

- [ ] **Step 3: Review for stale-state hazards**

Every transient path or branch head must include an observation date and an instruction
to verify it before acting.

---

### Task 2: Record PR chronology

**Files:**

- Create: `docs/engineering/history/2026-08-07-pr3-cache-workload-fairness-convergence.md`
- Create:
  `docs/engineering/history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md`
- Create: `docs/engineering/history/2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md`

**Interfaces:**

- Consumes: PR metadata, existing specs/plans, hardware notes, and recovered development
  history.
- Produces: chronological records that explain why each PR exists and how its final or
  current state differs from early plans.

- [ ] **Step 1: Record PR #3**

Include cross-cache-mode workload identity, tokenizer convergence, TDD evidence, merge
commit, and the unrelated fake-executable fixture caveat.

- [ ] **Step 2: Record PR #5**

Include victim/filler pressure semantics, population-result validation, token-budget
search, crossover measurements, current 13-file scope, current draft state, local 71-test
result with the fixture correction, and the cache static-check baseline blocker.

- [ ] **Step 3: Record PR #7**

Include the move from possible enforcement to shadow-only behavior, cost-curve semantics,
provenance, promotion EWMA, metrics, exact source-over-wheel validation, hardware anchors,
p256 adaptive crossover, final scoped checks, cleanup, and merge commit.

---

### Task 3: Convert recurring failures into incident memory

**Files:**

- Create: `docs/engineering/incidents/cache-workload-tokenizer-fairness-and-convergence.md`
- Create: `docs/engineering/incidents/fake-vllm-newline-fixture.md`
- Create: `docs/engineering/incidents/native-wheel-exact-overlay.md`
- Create: `docs/engineering/incidents/transient-metrics-before-connection-refused.md`
- Create: `docs/engineering/incidents/cache-benchmark-ci-baseline.md`

**Interfaces:**

- Consumes: detailed chronological history and direct diagnostic evidence.
- Produces: symptom-oriented records that future sessions can search before debugging.

- [ ] **Step 1: Write root-cause-focused incident documents**

Use consistent headings for symptom, impact, root cause, evidence, resolution, false
conclusions, regression protection, and related work.

- [ ] **Step 2: Preserve negative knowledge**

Explicitly record approaches that produced misleading validation, such as broad source
overlay and interpreting the fake-executable syntax failure as a `run_suite` regression.

---

### Task 4: Preserve expensive validation evidence

**Files:**

- Create: `docs/engineering/validation/cache-crossover-baseline.md`
- Create: `docs/engineering/validation/pr7-shadow-cost-model-hardware-validation.md`

**Interfaces:**

- Consumes: benchmark measurements, hardware environment evidence, PR #7 validation
  results, and artifact locations.
- Produces: reusable proof for future design and regression decisions.

- [ ] **Step 1: Record crossover seeds**

Store recompute, filesystem restore, filesystem promotion, and CPU-primary anchor values,
clearly labeling them as validation data rather than production defaults.

- [ ] **Step 2: Record shadow invariance evidence**

Document CPU-primary 1024, filesystem 1024, the five-point sweep, p256 adaptive behavior,
actual-path invariance, focused tests, scheduler integration, and final PR-scoped checks.

---

### Task 5: Write the active PR #5 handoff

**Files:**

- Create: `docs/engineering/handoffs/2026-08-10-pr5-current-handoff.md`

**Interfaces:**

- Consumes: current GitHub PR #5 metadata and the latest local validation observations.
- Produces: an actionable next-session entry point.

- [ ] **Step 1: Record authoritative remote state**

Include PR #5 head, draft/open state, 13 changed files, and its relationship to merged
PR `#3` and PR `#7`.

- [ ] **Step 2: Record the clean continuation path**

Explain the GitHub-to-Gitee-to-Pod constraint, the clean PR #5 workspace, the separate
cache-hygiene workstream, the 71/71 functional baseline with temporary fixture correction,
and the next static-check cleanup steps.

- [ ] **Step 3: Add do-not-repeat guidance**

List expensive or misleading work that is already settled, including re-running PR #7
hardware sweeps without a new hypothesis and committing a local merge that cannot reach
GitHub.

---

### Task 6: Verify and publish

**Files:** all files created by this plan.

- [ ] **Step 1: Verify repository diff**

Compare the documentation branch against `main` and confirm that only documentation files
were added.

- [ ] **Step 2: Verify markdown structure**

Check for placeholders, broken relative links, duplicate sibling headings, and obvious
markdownlint violations. Run repository CI when available.

- [ ] **Step 3: Open a PR**

Create a PR against `main` describing the taxonomy, initial historical coverage, and
handoff benefit.

- [ ] **Step 4: Merge after validation**

Use the repository-supported merge strategy after the PR is mergeable and checks are in an
acceptable state. Record the resulting merge commit in the final completion report.
