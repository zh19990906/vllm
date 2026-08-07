# Cache Restore-vs-Recompute Crossover Sweep Design

## Goal

Measure the prompt-length crossover where restoring KV from the filesystem tier becomes faster or slower than recomputing the same prefix locally.

The sweep must preserve comparable cache pressure across prompt lengths so the result reflects restore-vs-recompute cost rather than changing eviction severity or filesystem footprint.

## Scope

The experiment covers prompt lengths:

- 256
- 512
- 1024
- 2048
- 4096

For each prompt length, run only:

- `tiered-fs` with 2 GiB GPU KV and 2 GiB CPU primary
- `no-cache` with the same workload content

The existing 1024-token CPU-hit result remains a separate reference point and is not part of this sweep.

## Pressure Model

Add an optional workload configuration field:

```yaml
pressure_fill_tokens: 65536
```

For `eviction-restore`, the filler request count is derived per case as:

```text
ceil(pressure_fill_tokens / case.prompt_tokens)
```

The victim count remains `requests_per_case`.

For the five selected prompt lengths, 65,536 filler tokens yields approximately:

| Prompt tokens | Filler requests | Filler tokens |
| ---: | ---: | ---: |
| 256 | 256 | 65,536 |
| 512 | 128 | 65,536 |
| 1024 | 64 | 65,536 |
| 2048 | 32 | 65,536 |
| 4096 | 16 | 65,536 |

This keeps pressure and expected filesystem data volume approximately constant while preserving the same eviction ordering: victims are generated first, then fillers, and measurement replays only the victims.

## Configuration Semantics

`pressure_fill_requests` remains supported for compatibility with existing pressure experiments.

`pressure_fill_tokens` is optional and defaults to `0`.

Exactly one pressure mechanism may be active:

- `pressure_fill_requests > 0`, or
- `pressure_fill_tokens > 0`.

A configuration with both non-zero is invalid and must fail strict validation.

When `pressure_fill_tokens > 0`, an `eviction-restore` scenario is generated even when `pressure_fill_requests == 0`.

Metadata records both the configured token budget and the derived filler count so runs are auditable.

## Workload Fairness

The generator seed identity must remain independent of cache mode and execution controls.

For the same prompt length, `tiered-fs` and `no-cache` must produce byte-identical population and measurement datasets.

Changing the pressure mechanism must not perturb the victim prefix stream unnecessarily. The population remains one deterministic unique sequence containing victims followed by fillers.

## Sweep Configuration

Add a dedicated local experiment configuration with:

```yaml
prompt_tokens:
  - 256
  - 512
  - 1024
  - 2048
  - 4096
requests_per_case: 8
pressure_fill_requests: 0
pressure_fill_tokens: 65536
output_tokens: 1
concurrency:
  - 1
request_rate:
  - inf
```

Server and tiering settings remain the verified hardware baseline:

- GPU KV: 2 GiB via `--kv-cache-memory-bytes 2147483648`
- CPU primary: 2 GiB
- offload chunk: 64 tokens
- LRU eviction
- filesystem secondary under `/tmp/vllm-kv-cache`

The first hardware sweep should select only `no-cache__eviction-restore` and `tiered-fs__eviction-restore` cases for the five prompt lengths.

## Evidence Collected Per Point

For every completed case record:

- P95 TTFT
- request throughput
- prompt token source counts
- external prefix cache hit/query counts
- CPU-to-GPU transfer count, bytes, and time
- tiering sync/async lookup count and delay
- filesystem footprint for tiered-fs runs

Expected proof patterns:

### Recompute

- all prompt tokens reported as `local_compute`
- no external KV hits
- no CPU-to-GPU restore traffic
- no tiering lookup metrics

### Filesystem restore

- victim tokens reported as `external_kv_transfer`
- external prefix cache hits equal the restored victim prefix tokens
- one CPU-to-GPU transfer per measured victim
- one tiering async lookup per measured victim
- filesystem secondary contains materialized block files

## Crossover Output

Produce a table with one row per prompt length:

```text
prompt_tokens | recompute_p95_ms | fs_restore_p95_ms | delta_ms | preferred_path
```

Where:

```text
delta_ms = fs_restore_p95_ms - recompute_p95_ms
```

Interpretation:

- negative delta: filesystem restore is faster
- positive delta: recompute is faster
- a sign change between adjacent prompt lengths brackets the crossover region

Do not fit a complicated model from five points. The first result should be used as an empirical bracket and as input to the initial AdaptiveTiering cost model.

## Error Handling

A sweep point is invalid for crossover analysis if evidence does not prove the intended path. In particular:

- a tiered-fs point without secondary lookup evidence is not counted as an FS restore point
- a no-cache point with external KV transfer evidence is not counted as a recompute point
- workload generation failures are reported rather than silently resampled with a different seed

Partial sweep results remain useful and should be reported without interpolating missing measurements.

## Tests

Add tests for:

1. strict parsing of `pressure_fill_tokens`
2. rejection when both pressure fields are non-zero
3. scenario generation when token pressure is enabled
4. exact derived filler counts for the five sweep lengths
5. victim-first population order and victim measurement replay
6. metadata recording configured token budget and derived filler count
7. byte-identical workloads across cache modes for the same prompt length
8. existing `pressure_fill_requests` behavior remaining unchanged

Run the complete `benchmarks/cache/tests` suite and `python -m compileall -q benchmarks/cache` before hardware execution.

## Non-Goals

This change does not implement the AdaptiveTieringManager, production restore-vs-recompute scheduler policy, CPU-tier sweep, NVMe-specific benchmarking, remote cache, or a fitted analytical cost model.
