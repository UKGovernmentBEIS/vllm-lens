## Summary

Adds a generic hook system inspired by [Garçon](https://transformer-circuits.pub/2021/garcon/index.html) that lets users run arbitrary Python functions per-request, per-layer during inference. Hooks can both **modify** hidden states (ablation, clamping, patching) and **capture** data (probes, activation analysis) via a persistent `ctx.saved` dict.

Two modes:

- **Per-request hooks** — passed inline via `extra_args={"apply_hooks": [hook]}`. Results returned in the response. Cleaned up automatically.
- **Persistent hooks (Garçon-style)** — registered once via `POST /v1/hooks/register` (or `llm.register_hooks()`). Fire on every subsequent request. Results accumulate server-side until explicitly collected via `/v1/hooks/collect`. Supports append semantics — multiple register calls stack hooks.

### New API surfaces

**Types:**
- `Hook(fn=callable, layer_indices=[...], pre=False)` — user-defined hook with cloudpickle serialization for HTTP transport. Set `pre=True` to run before the layer (for corrupting inputs).
- `HookContext` — mutable context with `layer_idx`, `seq_len`, `saved` dict

**HTTP endpoints (no dev mode required):**
- `POST /v1/hooks/register` — register persistent hooks (appends)
- `POST /v1/hooks/collect` — retrieve accumulated results (non-destructive)
- `POST /v1/hooks/clear` — remove hooks and all contexts

**LLM methods:**
- `llm.register_hooks([hook])` / `llm.collect_hook_results()` / `llm.clear_hooks()`

### Pre-hooks and post-hooks

Hooks support both `pre=False` (default, post-hook — runs after the layer, receives output hidden states) and `pre=True` (pre-hook — runs before the layer, receives input hidden states). Pre-hooks enable corrupting embeddings for causal tracing / activation patching experiments.

Execution order per layer:
1. Pre-hooks (persistent first, then per-request)
2. Layer forward pass
3. Post-hooks (persistent first, then per-request)
4. Steering vectors
5. Activation capture

### ctx.model access

`HookContext.model` exposes the full model inside hooks, enabling operations like logit lens (projecting hidden states through `lm_head`) without transferring large tensors to the client.

### Design decisions
- Persistent hooks fire **before** per-request hooks (base layer → override)
- Per-request and persistent hooks use separate context storage to avoid cleanup conflicts
- `cloudpickle` used for HTTP transport of callables (security: arbitrary code execution, same trust model as existing `pickle.loads` in steering)
- Hooks run on all TP ranks (needed for modification); capture collected from rank 0 only
- `collect` is non-destructive; `clear` is the explicit cleanup

### Bug fix

Fixed `output_residual_stream` layer filtering via `vllm_xargs` — string values like `"[15]"` were treated as truthy (capturing all 32 layers) instead of being parsed as JSON lists. Now correctly filters to requested layers only.

### Examples

- **`causal_tracing.py`** — ROME-style causal tracing (activation patching). Corrupts subject token embeddings via a pre-hook, then systematically restores clean hidden states at each (layer, position) to produce a heatmap of causal importance. Reproduces the classic two-site pattern: early layers at subject tokens + late layers at last token.
- **`logit_lens.py`** — Projects hidden states at each layer through the unembedding matrix to show how the model's predictions evolve. Uses `ctx.model` to access `lm_head` weights on-GPU, avoiding large activation transfers.
- **`bench_session_vs_per_request.py`** — Benchmarks persistent vs per-request activation extraction.

## Test plan
- [x] Activation extraction returns correct shape and dtype
- [x] Steering changes output; scale=0 preserves output
- [x] Per-request hook capture returns `ctx.saved` data
- [x] Per-request hook modification changes output; `None` return preserves output
- [x] Hook capture matches native `output_residual_stream` (zero diff, single and multi-token)
- [x] Persistent lifecycle: register → run requests → collect → clear
- [x] Persistent + per-request coexistence (separate contexts, both fire)
- [x] Persistent modification affects all requests; clear restores baseline
- [x] Firing order: per-request hook sees persistent modification
- [x] Collect is non-destructive (two collects return same data)
- [x] Collect with no requests returns empty
- [x] Append across multiple register calls
- [x] Persistent capture matches native `output_residual_stream`
- [x] Pre-hook modification changes output
- [x] Pre-hook returning None preserves output
- [x] Pre-hook ctx.saved data is returned in hook_results (regression)
- [x] output_residual_stream layer filtering parses JSON list correctly (regression)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
