## Summary

Adds a generic hook system inspired by [Garçon](https://transformer-circuits.pub/2021/garcon/index.html) that lets users run arbitrary Python functions per-request, per-layer during inference. Hooks can both **modify** hidden states (ablation, clamping, patching) and **capture** data (probes, activation analysis) via a persistent `ctx.saved` dict.

Two modes:

- **Per-request hooks** — passed inline via `extra_args={"apply_hooks": [hook]}`. Results returned in the response. Cleaned up automatically.
- **Persistent hooks (Garçon-style)** — registered once via `POST /v1/hooks/register` (or `llm.register_hooks()`). Fire on every subsequent request. Results accumulate server-side until explicitly collected via `/v1/hooks/collect`. Supports append semantics — multiple register calls stack hooks.

### New API surfaces

**Types:**
- `Hook(fn=callable, layer_indices=[...])` — user-defined hook with cloudpickle serialization for HTTP transport
- `HookContext` — mutable context with `layer_idx`, `seq_len`, `saved` dict

**HTTP endpoints (no dev mode required):**
- `POST /v1/hooks/register` — register persistent hooks (appends)
- `POST /v1/hooks/collect` — retrieve accumulated results (non-destructive)
- `POST /v1/hooks/clear` — remove hooks and all contexts

**LLM methods:**
- `llm.register_hooks([hook])` / `llm.collect_hook_results()` / `llm.clear_hooks()`

### Design decisions
- Persistent hooks fire **before** per-request hooks (base layer → override)
- Per-request and persistent hooks use separate context storage to avoid cleanup conflicts
- `cloudpickle` used for HTTP transport of callables (security: arbitrary code execution, same trust model as existing `pickle.loads` in steering)
- Hooks run on all TP ranks (needed for modification); capture collected from rank 0 only
- `collect` is non-destructive; `clear` is the explicit cleanup

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
