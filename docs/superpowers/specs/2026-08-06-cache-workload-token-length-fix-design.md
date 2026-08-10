# Cache workload token-length generation fix

## Problem

`benchmarks/cache/workload.py` currently samples exactly `requested_length` token IDs, decodes them to text, and re-encodes the text. Some real tokenizers, including Qwen2.5, do not preserve arbitrary token sequences through `decode -> encode`. A requested length of 1024 can re-encode to roughly 1068-1081 tokens, so all cache modes fail before the server starts.

The benchmark must retain deterministic prompt generation and must not weaken `token_length_tolerance`, because large prompt-length differences would invalidate TTFT comparisons.

## Selected approach

Keep the existing token-ID sampling and round-trip validation, but adapt the number of sampled suffix token IDs after each failed attempt.

For each attempt:

1. Sample `candidate_suffix_length` token IDs, preserving any fixed prefix.
2. Decode and re-encode the complete prompt.
3. Accept only when the observed length is within the configured tolerance and all prefix/uniqueness constraints pass.
4. Otherwise estimate the next suffix length from the observed round-trip ratio:

   `next_suffix = round(candidate_suffix * target_suffix / observed_suffix)`

5. Clamp the next suffix length to a valid non-negative range and ensure an adjustment always makes progress when the estimate equals the current value.

The fixed-prefix path uses the same adjustment logic while keeping the source prefix unchanged. Exact-prefix and restart-persistence workloads continue to reuse identical generated rows. Case IDs and configuration schemas remain unchanged.

## Alternatives rejected

### Increase token-length tolerance

Rejected because the observed error is tens of tokens. Accepting that variance would make otherwise matching cache-mode measurements incomparable.

### Generate repeated natural-language filler

Rejected because it changes workload entropy and shared-prefix characteristics, and can create model- or language-specific behavior.

### Truncate encoded IDs after round-trip

Rejected because benchmark requests are text, not token-ID arrays; truncating encoded IDs without regenerating matching text would not control the actual server-side prompt length.

## Error handling

The generator keeps the existing bounded retry behavior. Failure messages will include the requested length and final observed length. Invalid fixed-prefix lengths continue to fail immediately.

## Testing

Add a deterministic tokenizer test double whose `decode -> encode` round trip expands token counts. Verify that:

- generation converges within the existing tolerance;
- deterministic seeds still produce identical artifacts;
- uniqueness checks remain active;
- exact-warm population and measurement rows remain identical;
- shared-prefix generation preserves the encoded prefix;
- existing stable-tokenizer tests continue to pass.

Run the complete `benchmarks/cache/tests` suite and compile the cache benchmark package.

## Scope

This change only affects workload generation and its tests. It does not modify cache-mode commands, Scheduler, attention code, KV cache internals, case IDs, result schemas, or reporting.
