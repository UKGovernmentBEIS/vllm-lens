## Unreleased

## v1.2.0 (14 July 2026)

- Hooks: Added a generic, Garçon-style hook system — run arbitrary Python functions per-request and per-layer to capture data (via `ctx.saved`) and/or modify hidden states (by returning a tensor). Supports per-request hooks (`apply_hooks` in `extra_args`), persistent register-once hooks (`register_hooks` / `collect_hook_results` / `clear_hooks`), and pre-hooks (run before the layer forward pass). Exposed over HTTP at `/v1/hooks/*` and through a `VLLMLensClient`; `ctx.get_parameter()` gathers parameters across tensor- and pipeline-parallel ranks. (#3)
- Steering: Fixed `norm_match` on fused-residual architectures (Qwen, Gemma, Llama, …). It now scales the steering vector to the full residual stream rather than the MLP-delta half, so the injected magnitude equals `‖residual‖ · scale` as intended. **Behavior change:** existing `norm_match=True` steering becomes correspondingly stronger. (#7)
- Performance: On the offline `LLM.generate` path, a batch's captured activations are now fetched in a single RPC instead of one per request. (#6)
- Plugin: Added the `VLLM_LENS_DISABLE=1` environment variable to make the plugin a no-op, so vllm-lens can be installed alongside another inference server without perturbing it. (#9)
- Plugin: Default to the V1 model runner and raise a clear error if the V2 runner is active (the capture/steering hooks read V1 model-runner internals). (#12)
- Examples: Moved examples to a top-level `examples/` directory and added causal-tracing, logit-lens, deception-probe, and emotion-concepts examples. (#4, #11)
- Docs: Documented the `Hook`/`HookContext` interface and added first-class README sections for activation capture and steering. (#13, #16)

## v1.1.0

- Prior releases.
