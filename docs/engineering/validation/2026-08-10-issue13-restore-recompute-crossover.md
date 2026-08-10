# Issue #13: restore vs recompute crossover hardware validation

- Date: 2026-08-10
- Issue: #13 `[P0] 系统化测量 restore vs recompute crossover`
- Baseline commit: `b68dcd4f85b2ffe6299e75172218c1d44bff4040`
- Scope: one model / one known GPU Pod / concurrency=1 baseline
- Structured calibration artifact: `2026-08-10-issue13-restore-recompute-crossover.json`

## Executive conclusion

The measurement establishes a tier-dependent restore/recompute boundary:

- **CPU-primary restore:** P50 crossover is reliably bracketed between **192 and 216 requested prompt tokens**.
- **tiered-fs configuration:** no P50 crossover was observed from **256 through 4096 requested tokens**; recompute was faster at every point.
- A configured cache mode is not sufficient provenance: the initial 2 GiB `cpu-offload` sweep actually recomputed after CPU eviction.
- Therefore `source tier` and actual runtime restore/recompute evidence must be first-class inputs to #14 cost-model calibration.

The filesystem result is intentionally described as **tiered-fs / lower-tier external restore**, not physical NVMe restore. The configured path is container filesystem backed in this Pod, and physical NVMe provenance was not established.

## Environment

### Repository

- worktree branch: `local/issue13-crossover`
- HEAD: `b68dcd4f85b2ffe6299e75172218c1d44bff4040`
- Python: `Python 3.11.11`
- vLLM CLI: `0.26.0`

### GPU

```text
NVIDIA RTX PRO 5000 72GB Blackwell, GPU-5516e45d-3e50-69ef-f0f2-8ecff465beea, 73415 MiB, 580.126.09
NVIDIA RTX PRO 5000 72GB Blackwell, GPU-74f68875-4f31-d1fb-f276-b2bb9cc7c80d, 73415 MiB, 580.126.09
```

Benchmark execution was pinned with:

```bash
CUDA_VISIBLE_DEVICES=0
```

GPU topology:

```text
GPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	NODE	0-63,128-191	0		N/A
GPU1	NODE	 X 	0-63,128-191	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

### CPU

- Model name: `AMD EPYC 9A75 64-Core Processor`
- GPU-visible NUMA affinity: `0`
- GPU-visible CPU affinity: `0-63,128-191`
- Provenance note: the generator's localized `lscpu` field parser produced an empty structured CPU map; the model name was captured separately in the same Pod validation session, and affinity is corroborated by `nvidia-smi topo -m`.

Memory snapshot:

```text
total        used        free      shared  buff/cache   available
内存：      1.5Ti       128Gi       791Gi        53Gi       590Gi       1.3Ti
交换：         0B          0B          0B
```

Host load snapshot:

```text
17:50:31 up 25 days,  7:17,  0 users,  load average: 1.12, 1.31, 2.33
```

### Filesystem provenance

`/tmp/vllm-kv-cache`:

```text
文件系统       类型     大小  已用  可用 已用% 挂载点
overlay        overlay  984G  510G  434G   55% /
```

Model mount:

```text
文件系统                                                        类型         大小  已用  可用 已用% 挂载点
mtp-temp.oss-cn-beijing-internal.aliyuncs.com:/GW00357265/model fuse.ossfs2   16E     0   16E    0% /mnt/model
```

## Benchmark configuration

Core parameters:

```yaml
model: /mnt/model/Qwen2.5-7B-Instruct
served_model: qwen2.5-7b
tensor_parallel_size: 1
pipeline_parallel_size: 1
gpu_kv_bytes: 2147483648       # 2 GiB
cpu_bytes_baseline: 2147483648 # 2 GiB
cpu_bytes_valid_restore: 8589934592 # 8 GiB
offload_block_size: 64
eviction_policy: lru
pressure_fill_tokens: 65536
requests_per_case: 8
concurrency: 1
request_rate: inf
output_tokens: 1
seed: 1
token_length_tolerance: 2
filesystem_root: /tmp/vllm-kv-cache
```

The 8 GiB CPU variant changes only CPU offload capacity relative to the crossover baseline; GPU KV remains 2 GiB so the victim is still evicted from GPU while remaining recoverable from CPU.

## Raw run directories

- `wide_smoke`: `/code/results/cache/20260810T083254Z-d78c7e79`
- `wide_sweep`: `/code/results/cache/20260810T084238Z-d78c7e79`
- `cpu_2g_invalid`: `/code/results/cache/20260810T085151Z-d78c7e79`
- `cpu_8g_1024`: `/code/results/cache/20260810T090011Z-2b4f76bf`
- `cpu_8g_wide`: `/code/results/cache/20260810T090219Z-2b4f76bf`
- `short_128_192`: `/code/results/cache/20260810T090835Z-45ca6bec`
- `point_224`: `/code/results/cache/20260810T091358Z-d50cae9e`
- `point_208_failure`: `/code/results/cache/20260810T091653Z-581858ca`
- `point_216`: `/code/results/cache/20260810T092229Z-43e7c684`
- `repeat_192_1`: `/code/results/cache/20260810T092546Z-d5811de8`
- `repeat_192_2`: `/code/results/cache/20260810T092803Z-d5811de8`
- `repeat_192_3`: `/code/results/cache/20260810T092948Z-d5811de8`
- `repeat_216_1`: `/code/results/cache/20260810T093135Z-43e7c684`
- `repeat_216_2`: `/code/results/cache/20260810T093320Z-43e7c684`
- `repeat_216_3`: `/code/results/cache/20260810T093507Z-43e7c684`

Each run is grounded in its `scenario-results.jsonl` plus per-case `raw/<case-id>/` artifacts.

## Wide cost curves

`delta = restore TTFT - recompute TTFT`; negative means restore is faster.

| requested | recompute P50 | CPU P50 | CPU ΔP50 | tiered-fs P50 | fs ΔP50 | CPU P95 | fs P95 | CPU ext tokens | fs ext tokens | CPU MiB | fs MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 24.784 | 20.150 | -4.634 | 30.305 | +5.521 | 21.872 | 36.007 | 1856 | 1856 | 101.5 | 101.5 |
| 512 | 44.338 | 19.742 | -24.596 | 50.514 | +6.176 | 23.057 | 59.159 | 4096 | 4096 | 224.0 | 224.0 |
| 1024 | 77.679 | 20.913 | -56.766 | 98.799 | +21.120 | 24.687 | 101.799 | 8192 | 8192 | 448.0 | 448.0 |
| 2048 | 146.365 | 26.096 | -120.269 | 184.965 | +38.600 | 29.173 | 320.793 | 16128 | 16128 | 882.0 | 882.0 |
| 4096 | 297.150 | 33.190 | -263.960 | 639.955 | +342.805 | 35.213 | 648.235 | 32704 | 32704 | 1788.5 | 1788.5 |

### Wide-curve interpretation

- CPU restore is faster than recompute at every measured wide point from 256 through 4096.
- tiered-fs restore is slower than recompute at every measured point from 256 through 4096.
- The tiered-fs penalty grows sharply at 4096 requested tokens.
- Every valid restore point has 8 CPU→GPU transfers and positive external-KV tokens.
- Every tiered-fs wide point also has 8 async tiering lookups.

## CPU crossover refinement

| requested | recompute P50 | CPU P50 | ΔP50 | recompute P95 | CPU P95 | ΔP95 | external tokens | xfers |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 17.663 | 19.546 | +1.883 | 19.660 | 21.220 | +1.560 | 832 | 8 |
| 192 | 18.691 | 19.162 | +0.471 | 22.186 | 21.830 | -0.356 | 1344 | 8 |
| 216 | 23.010 | 19.953 | -3.057 | 24.872 | 22.130 | -2.742 | 1536 | 8 |
| 224 | 23.244 | 19.930 | -3.315 | 25.291 | 22.295 | -2.996 | 1536 | 8 |

### 192 / 216 repeat validation

| run | role | requested | ΔP50 | ΔP95 | ΔP99 |
|---|---|---:|---:|---:|---:|
| `20260810T090835Z-45ca6bec` | anchor | 192 | +0.471 | -0.356 | +0.062 |
| `20260810T092229Z-43e7c684` | anchor | 216 | -3.057 | -2.742 | -2.483 |
| `20260810T092546Z-d5811de8` | repeat | 192 | +0.512 | -0.342 | +0.187 |
| `20260810T092803Z-d5811de8` | repeat | 192 | +0.614 | -0.304 | +0.207 |
| `20260810T092948Z-d5811de8` | repeat | 192 | +0.708 | -0.159 | +0.391 |
| `20260810T093135Z-43e7c684` | repeat | 216 | -3.375 | -3.243 | -3.052 |
| `20260810T093320Z-43e7c684` | repeat | 216 | -3.324 | -3.360 | -3.305 |
| `20260810T093507Z-43e7c684` | repeat | 216 | -2.973 | -2.267 | -2.030 |

Repeat-only summary:

- 192 P50: mean `+0.611 ms`, range `+0.512` to `+0.708`; all 3 repeats favor recompute.
- 216 P50: mean `-3.224 ms`, range `-3.375` to `-2.973`; all 3 repeats favor restore.

**Result:** CPU-primary P50 crossover is conservatively reported as **192–216 requested prompt tokens**.

For P95, 128 favors recompute while the 192 anchor and all three 192 repeats favor restore. The current P95 bracket is therefore 128–192 requested tokens; it was not further refined in #13.

## Restore provenance evidence

A restore sample is accepted only when runtime metrics show external KV use and actual transfer behavior, rather than inferring provenance from the configured cache mode.

For valid CPU-primary restore points:

- `prompt_tokens_by_source{source="external_kv_transfer"} > 0`
- `kv_offload_size_count{transfer_type="CPU_to_GPU"} = 8`
- `kv_offload_total_bytes{transfer_type="CPU_to_GPU"} > 0`
- local compute is only a small residual where present.

For the tiered-fs wide curve the same external/transfer evidence is present, plus 8 async tiering lookups per case.

This proves lower-tier/external restore and rules out a warm GPU-only hit. It does **not** prove that the physical backing device is NVMe.

## Requested tokens are not restored-KV tokens

The model must not treat requested prompt length as identical to actual external KV hit tokens. Examples from the valid CPU curve:

| requested | external tokens total | external tokens/request |
|---:|---:|---:|
| 256 | 1856 | 232.0 |
| 512 | 4096 | 512.0 |
| 1024 | 8192 | 1024.0 |
| 2048 | 16128 | 2016.0 |
| 4096 | 32704 | 4088.0 |
| 192 | 1344 | 168.0 |
| 216 | 1536 | 192.0 |
| 224 | 1536 | 192.0 |

The reported crossover interval is therefore explicitly on the **requested-prompt axis**, not an exact restored-KV-token threshold.

## Invalid 2 GiB CPU sweep: capacity-pressure diagnostic

The first 2 GiB `cpu-offload` sweep must not be used as an actual CPU restore curve.

| requested | recompute P50 | configured cpu-offload P50 | ΔP50 | external tokens | CPU→GPU xfers | local compute |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 24.784 | 25.426 | +0.642 | 0 | 0 | 2049 |
| 512 | 44.338 | 44.517 | +0.179 | 0 | 0 | 4105 |
| 1024 | 77.679 | 78.221 | +0.542 | 0 | 0 | 8198 |
| 2048 | 146.365 | 146.883 | +0.517 | 0 | 0 | 16379 |
| 4096 | 297.150 | 298.053 | +0.903 | 0 | 0 | 32769 |

All five cases show zero external-KV tokens and zero CPU→GPU restores, while local compute is approximately 8 × prompt length. The victims were evicted from the 2 GiB CPU tier by the pressure population and recomputed.

Increasing only CPU capacity to 8 GiB restored expected provenance. At 1024 requested tokens the validating case showed:

```text
external_hits       = 8192
external_tokens     = 8192
local_compute       = 6
CPU_to_GPU bytes    = 469762048
CPU_to_GPU transfers= 8
```

This diagnostic is important input to #14: configured cache mode must never be used as a substitute for actual source/action provenance.

## 208 requested-token workload failure

- `no-cache`: status `benchmark_error`, `WorkloadGenerationError` at stage `workload` — unable to generate prompt with requested length 208; last observed length was 190 after 32 attempts
- `cpu-offload`: status `benchmark_error`, `WorkloadGenerationError` at stage `workload` — unable to generate prompt with requested length 208; last observed length was 190 after 32 attempts

The deterministic generator was not reseeded or repeatedly resampled to force this point to succeed. 200 and 216 were separately preflighted as generatable; 216 was used as the next measured boundary point.

## Workload fairness

For every wide token point, the measurement and population JSONL SHA-256 hashes match across no-cache, valid CPU restore, and tiered-fs cases.
Short CPU/no-cache pairs likewise have byte-identical workloads.

This ensures cost comparisons are paired against the same deterministic prompt population rather than different sampled requests.

## Reproduction command pattern

Wide baseline cases were executed as selected `eviction-restore` cases:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/cache/run_suite.py \
  --config benchmarks/cache/configs/local-crossover.yaml \
  --case-id <selected-no-cache-or-tiered-fs-case-id>
```

Valid CPU-primary cases used the same config with only:

```yaml
cache:
  cpu_bytes_to_use: 8589934592
```

and were executed with:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/cache/run_suite.py \
  --config /tmp/issue13-cpu-crossover.yaml \
  --case-id <selected-cpu-offload-eviction-restore-case-id>
```

Short crossover configs changed only `workload.prompt_tokens` while keeping the deterministic seed and all other benchmark semantics fixed.

Exact case IDs, run directories, TTFT data, workload hashes, transfer metrics, and evidence are retained in the companion JSON artifact.

## Completion against #13

- [x] Dense enough measurement to locate CPU P50 crossover and show no tiered-fs P50 crossover in the measured 256–4096 range.
- [x] Interpretable recompute, CPU-primary restore, and tiered-fs curves.
- [x] Per-restore-case lower-tier evidence.
- [x] Structured JSON result suitable for #14 calibration.
- [x] Commands, raw run directories, environment, failures, and caveats recorded here.

## Limitations and handoff

- This is a **concurrency=1 baseline** on one model and one machine.
- Model/concurrency/hardware generalization belongs to #15.
- Physical NVMe provenance is not established; use filesystem/local-tier terminology.
- P95 CPU crossover is bracketed but not repeated/refined at the lower 128-token endpoint.
- 208 requested tokens are unavailable under the deterministic generator for this seed/tolerance.
- No active restore/recompute execution decision was enabled in #13.

The resulting calibration implication for #14 is: **source tier, actual external KV tokens, transfer bytes, fixed restore overhead, and recompute cost must be modeled explicitly; `cache hit => restore` is not valid.**
