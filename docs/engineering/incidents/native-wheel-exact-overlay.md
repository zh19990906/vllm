# Incident: Source Checkout Does Not Prove Runtime Under Test

Status: **resolved validation method**.

First identified during PR #7 scheduler and hardware validation.

## Symptom

A disposable integration worktree contained the PR #7 Python source and PR #5 benchmark
code, but running the benchmark did not automatically mean the `vllm serve` subprocess was
executing PR #7 runtime modules.

The benchmark runner launches the `vllm` executable from `PATH`. On the validation machine,
that executable belonged to the installed vLLM wheel.

A second problem appeared when attempting broad source-first imports: the source tree did
not contain all native extension artifacts supplied by the installed wheel.

## Impact

Without runtime provenance checks, a hardware run could appear to validate feature code
while actually exercising the unmodified installed package.

Conversely, forcing the entire source tree ahead of the wheel could break imports because
native `.so` modules and packaged dependencies no longer came from the environment that
built them.

Both failure modes can produce misleading validation conclusions.

## Root cause

There were two independent layers:

1. Python source location;
2. native/runtime packaging from the installed wheel.

A worktree merge changes files on disk but does not change which executable a subprocess
uses. A global `PYTHONPATH` or full-tree source overlay changes too much and can detach
Python code from its compatible native modules.

## Rejected validation approaches

### Merely merge PR #7 and PR #5 in a worktree

Insufficient. The benchmark still invokes the installed `vllm` executable unless `PATH`
and import behavior are deliberately changed.

### `pip install -e` or full rebuild

Unnecessarily invasive for validation, expensive, and likely to disturb the known-good
wheel/native environment.

### Broad source-first `PYTHONPATH`

Insufficiently precise. It can shadow package modules that should remain from the installed
wheel and expose missing or incompatible native extension imports.

## Correct validation method: exact source-over-wheel

The working solution was an exact module overlay on top of the installed wheel:

1. import and retain the installed `vllm` package/runtime base;
2. register an exact source finder for only the feature modules under validation;
3. load those six Python modules from the PR #7 validation worktree;
4. leave native and unrelated package modules in the installed wheel;
5. use a temporary `vllm` PATH shim so benchmark-spawned server processes install the same
   exact overlay before entering the real CLI.

The six overlaid runtime modules were:

```text
vllm/v1/kv_offload/cost_model.py
vllm/v1/kv_offload/base.py
vllm/v1/kv_offload/tiering/spec.py
vllm/v1/kv_offload/tiering/manager.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
```

Native modules such as FlashAttention and `_C_stable_libtorch` remained from the installed
wheel.

## Required provenance check

A validation run is not credible merely because the overlay mechanism was configured. It
must verify module origins.

The successful PR #7 validation checked that:

```text
cost_model.py        -> feature/integration worktree
base.py              -> feature/integration worktree
tiering/spec.py      -> feature/integration worktree
tiering/manager.py   -> feature/integration worktree
offloading/metrics.py   -> feature/integration worktree
offloading/scheduler.py -> feature/integration worktree

vllm_flash_attn      -> installed wheel
_C_stable_libtorch  -> installed wheel
```

The same guarantee was required inside the benchmark server subprocess, not only in the
parent test interpreter.

## Evidence that the method worked

Using the installed wheel/native environment with the exact six-module overlay, the real
scheduler integration suite completed:

```text
93 passed in 6.27s
```

The same method was then used for PR #7 CPU-primary and filesystem hardware anchors and the
filesystem sweep.

## False conclusions to avoid

- Do not claim feature hardware validation because the feature files exist in the current
  worktree.
- Do not assume a subprocess inherits source imports in the same way as the parent process.
- Do not use a broad source overlay when only a small number of Python modules changed.
- Do not interpret native import failures from a source-first environment as evidence that
  the feature logic itself is broken.
- Do not replace a known-good installed wheel just to run a Python-only feature validation
  unless the feature actually changes native code.

## Regression checklist

For future Python-only runtime validation against an installed vLLM wheel:

1. list the exact changed runtime Python modules;
2. keep the wheel as package/native base;
3. overlay only those modules;
4. assert each changed module's `__file__` or loader origin;
5. assert key native modules still originate from the wheel;
6. repeat the provenance assertion in any spawned CLI/server process;
7. record the provenance evidence with the benchmark result.

## Related work

- PR #7 final head: `96de0c823721c374527dbb0b3a49fdc7eccba341`
- PR #7 merge commit: `37f65141108e112a317fe4a5d8215a4c21c3c00e`
- [`../history/2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md`](../history/2026-08-07-to-2026-08-10-pr7-shadow-cost-model.md)
- [`../validation/pr7-shadow-cost-model-hardware-validation.md`](../validation/pr7-shadow-cost-model-hardware-validation.md)
