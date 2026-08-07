# Cache Workload Fairness and Convergence Design

## Context

The cache benchmark suite currently derives its workload RNG seed from `case.case_id`. A case ID includes `cache_mode`, so otherwise matching no-cache, GPU APC, CPU offload, and tiered-filesystem cases generate different prompts. This invalidates direct cache-mode performance comparisons because the compared cases do not receive identical request content.

The existing Qwen2.5 token-length convergence fix also resamples a fresh random suffix on every adjustment attempt. For tokenizers whose `decode -> encode` round trip expands or contracts depending on token content, the observed-length feedback is therefore applied to a different random sample on the next attempt. Hardware testing showed that some cache-mode-specific seeds still exhaust the 32-attempt bound with observed lengths such as 998 or 1039 for a requested 1024 tokens.

## Goals

1. Make workload content identical across cache modes for otherwise matching workload shapes.
2. Keep workload content independent of execution-only controls such as concurrency and request rate so comparisons across those axes are also controlled.
3. Make prompt-length adjustment operate on a stable token sequence so observed length is meaningful feedback for the next candidate length.
4. Preserve configured token-length tolerance, encoded shared-prefix guarantees, uniqueness checks, deterministic generation, the 32-attempt bound, case IDs, configuration schemas, result schemas, and benchmark commands.
5. Keep all changes inside `benchmarks/cache`; do not modify vLLM inference-core behavior.

## Non-goals

- Changing cache-mode case IDs or result-directory naming.
- Changing Scheduler, Attention, PagedAttention, KVCacheManager, BlockPool, OffloadingConnector, or CUDA kernels.
- Relaxing `token_length_tolerance`.
- Turning approximate prefix matching into semantic matching.
- Making benchmark datasets persistent across unrelated suite configurations.

## Workload Content Identity

Introduce an explicit workload-content identity used only for RNG seeding. It contains:

- `workload_kind`
- `prompt_tokens`
- `prefix_ratio`
- `repetition`

The configured `workload.seed` remains the user-controlled root seed.

The identity deliberately excludes:

- `cache_mode`: cache policy must not change request content.
- `concurrency`: scheduling pressure must not change request content.
- `request_rate`: arrival rate must not change request content.
- `case_id`, result paths, and filesystem cache paths: storage/execution metadata must not change request content.

This means matching workload shapes produce byte-identical `measure.jsonl` and, when applicable, `populate.jsonl` across cache modes and execution controls. Case IDs remain unchanged and continue to distinguish result rows.

## Stable Token-stream Length Search

For each generated prompt, `_sample_prompt` will use a stable suffix token stream during its bounded length-search loop.

1. Start with the requested source suffix length.
2. Lazily extend a local suffix token pool from the RNG when a candidate length requires more tokens than are already available.
3. Build each candidate from `fixed_prefix + token_pool[:candidate_suffix_length]`.
4. Decode and re-encode the candidate.
5. If observed encoded length is outside tolerance, compute the next suffix length using the existing proportional adjustment rule.
6. If the next length would repeat a previously tried candidate, move one token in the direction indicated by the observed error, choosing the nearest untried non-negative length.
7. Apply encoded-prefix and uniqueness checks only after length is within tolerance, as today.
8. Keep the existing maximum of 32 attempts and the existing failure type.

The important property is that changing candidate length no longer changes all previously sampled suffix tokens. The function being searched is therefore the encoded length of prefixes of one deterministic token stream rather than 32 unrelated random samples.

## Shared-prefix Behavior

Shared-prefix workloads retain their existing fixed source prefix and encoded-prefix validation. Only suffix sampling changes. A stable suffix stream is local to each generated row; it does not alter the shared prefix. The first accepted row still establishes the encoded prefix used to validate subsequent rows.

## Tests

Add regression tests before implementation for the following invariants:

1. Matching warm-exact cases from every cache mode produce byte-identical measurement and population JSONL and matching file hashes.
2. Matching shared-prefix cases from every cache mode produce byte-identical measurement JSONL.
3. Changing concurrency or request rate does not change workload content for the same workload shape.
4. Case IDs remain different across cache modes even though generated workload content is identical.
5. A tokenizer double whose round-trip expansion depends on token values exposes the current resampling failure but converges when candidate lengths reuse a stable token stream.
6. Existing expanding-tokenizer, deterministic generation, exact-warm identity, uniqueness, shared-prefix, metadata, and command-construction tests remain green.

## Metadata

`metadata.json` keeps `case_id` and `cache_mode` as execution metadata. `generator_seed` becomes the cache-mode-independent workload seed derived from the explicit workload-content identity. Existing schema keys are unchanged.

## Failure Semantics

If the stable search still cannot produce a prompt within tolerance after 32 attempts, generation fails with `WorkloadGenerationError` as before. The benchmark will not silently widen tolerance or substitute a different workload shape.

## Acceptance Criteria

- Cross-mode matching workload artifacts are byte-identical.
- Concurrency/request-rate changes do not alter prompt content for an otherwise identical workload shape.
- Case IDs and benchmark command behavior are unchanged.
- Qwen2.5-like non-stable round trips converge in regression coverage without widening tolerance.
- Full `benchmarks/cache/tests` passes.
- `python -m compileall -q benchmarks/cache` passes.
