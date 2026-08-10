# Incident: Transient `metrics_before` Connection Refused

Status: **non-reproduced transient; no code change made**.

Observed during a 4096-token filesystem cache benchmark.

## Symptom

One benchmark run failed before measurement while collecting pre-run metrics with a
connection-refused error:

```text
MetricsCollectionError: connection refused
```

Because the failure occurred around a long, high-pressure case, possible explanations
included server crash, OOM, startup/readiness race, or a genuine metrics endpoint
reliability problem.

## Investigation

The failure was investigated before adding retry behavior.

Observed evidence included:

- OOM counter: 0;
- OOM-kill counter: 0;
- server stderr contained no fatal exception or traceback explaining a crash;
- the later server shutdown was consistent with benchmark-runner cleanup after the runner
  itself had failed;
- the same p4096 case was rerun independently;
- the rerun completed with `rc=0`;
- a watcher observed the server reach ready state and `/metrics` remain available until the
  benchmark's normal shutdown.

## Conclusion

The event was classified as:

```text
non-reproduced transient metrics collection failure
```

There was not enough evidence to justify a production or benchmark retry policy change.

## Why no retry was added

A retry can hide real lifecycle errors as easily as it can smooth a harmless transient.
Adding one after a single non-reproduced failure would change benchmark semantics without a
reliable model of the underlying fault.

The decision was therefore to preserve the failure as historical evidence and wait for a
reproducible pattern before changing behavior.

## False conclusions to avoid

- Do not describe the single failure as an OOM; the inspected counters did not support
  that conclusion.
- Do not describe later process termination as the original cause without separating
  runner cleanup from the pre-cleanup failure.
- Do not add broad retry loops merely because one run failed to scrape `/metrics`.
- Do not discard a later successful independent reproduction attempt; the non-reproduction
  is part of the diagnosis.

## What to do if it recurs

If the same error appears again, preserve enough evidence to determine whether it is the
same incident:

1. record server readiness time;
2. timestamp the failing metrics request;
3. preserve server stdout/stderr before cleanup;
4. inspect cgroup/system OOM counters;
5. independently probe `/metrics` during the run;
6. determine whether failure clusters around startup, shutdown, pressure level, or case
   length;
7. only design retry behavior after a repeatable failure window is identified.

## Related work

- PR #5 cache crossover / eviction-restore benchmark work.
- [`../history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md`](../history/2026-08-07-to-2026-08-10-pr5-cache-eviction-restore-benchmark.md)
- [`../validation/cache-crossover-baseline.md`](../validation/cache-crossover-baseline.md)
