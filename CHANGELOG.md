## Unreleased

- Activations: Added an opt-in **binary transport** for activation payloads over the HTTP API (`activations_transport="binary"` in `vllm_xargs` / the client). Instead of base64-inlining the (already zstd-compressed) tensors in the JSON body — which inflates ~33% and forces the whole blob to be materialized and parsed in one piece on both ends — the raw bytes are parked in a bounded, TTL'd, process-local store under an unguessable handle, and the client fetches them from `GET /v1/activations/{handle}` as `application/octet-stream`. Default responses are byte-identical (base64); `decode_activations` and `VLLMLensClient` handle both transports transparently. The store is bounded by `VLLM_LENS_ACT_MAX_BYTES` (default 8 GiB) and `VLLM_LENS_ACT_TTL_S` (default 300 s), and is process-local (pin fetches to the producing replica in multi-replica deployments). (#31)
- Plugin: Decoder-layer discovery now uses vLLM's `static_forward_context` attention-layer registry instead of hardcoded attribute paths (`model.model.layers`, …), so capture, steering, and hooks work on any decoder architecture vLLM can serve — including MLA (DeepSeek/GLM-5), hybrid (Mamba/linear-attention), and multimodal layouts. Only known decoder mixers (`Attention`, `MLAAttention`, `MambaBase`) are treated as layer markers; the discovered map is checked against the config's `num_hidden_layers` and discovery **fails loudly** (`LayerDiscoveryError`) on any mismatch rather than falling back to a guess. New `VLLM_LENS_LAYER_PATH` env var (a `get_submodule` path template, e.g. `model.layers.{i}`) overrides discovery for layouts the registry can't resolve. (#33)

## v1.2.1 (22 July 2026)

- Steering: Fixed offline steering via `LLM.chat` — the plugin now patches `LLM.chat` (which submits requests to the engine directly rather than routing through `LLM.generate`). Previously, live `SteeringVector` objects raised a msgpack `TypeError`, and JSON-serialized vectors ran **silently unsteered**. Activation capture (`output_residual_stream`) and per-request hooks (`apply_hooks`) now also work through `LLM.chat`. (#28)
- Steering: The offline `LLM.generate` / `LLM.chat` path now decodes the JSON-string wire format for `apply_steering_vectors` and `apply_hooks` (the form the HTTP API accepts via `vllm_xargs`), instead of failing inside `collective_rpc`. Both entry points accept both wire forms. (#28)

## v1.2.0 (17 July 2026)

- Hooks: Added a generic, Garçon-style hook system — run arbitrary Python functions per-request and per-layer to capture data (via `ctx.saved`) and/or modify hidden states (by returning a tensor). Supports per-request hooks (`apply_hooks` in `extra_args`), persistent register-once hooks (`register_hooks` / `collect_hook_results` / `clear_hooks`), and pre-hooks (run before the layer forward pass). Exposed over HTTP at `/v1/hooks/*` and through a `VLLMLensClient`; `ctx.get_parameter()` gathers parameters across tensor- and pipeline-parallel ranks. (#3)
- Hooks: Added `POST /v1/hooks/clear_results` (`client.clear_hook_results()`) — drain accumulated persistent-hook results while keeping the hooks registered, so long-lived clients can bound accumulation without re-uploading hook state. (#25)
- Steering: Fixed `norm_match` on fused-residual architectures (Qwen, Gemma, Llama, …). It now scales the steering vector to the full residual stream rather than the MLP-delta half, so the injected magnitude equals `‖residual‖ · scale` as intended. **Behavior change:** existing `norm_match=True` steering becomes correspondingly stronger. (#7, #15)
- Performance: On the offline `LLM.generate` path, a batch's captured activations are now fetched in a single RPC instead of one per request. (#6)
- Plugin: Added the `VLLM_LENS_DISABLE=1` environment variable to make the plugin a no-op, so vllm-lens can be installed alongside another inference server without perturbing it. (#9, #14)
- Plugin: Default to the V1 model runner and raise a clear error if the V2 runner is active (the capture/steering hooks read V1 model-runner internals). (#12)
- Examples: Moved examples to a top-level `examples/` directory and added causal-tracing, logit-lens, deception-probe, and emotion-concepts examples. (#4, #11)
- Examples: Added a Jacobian-lens / J-space example — fit a per-model average Jacobian `J_l = E[∂h_final/∂h_l]` offline (HuggingFace backward pass), then read it out live on vLLM-captured residuals to see what the model is "disposed to say" at each layer. (#19)
- Examples: Hardened the Jacobian-lens run path for tensor / pipeline / expert parallelism, so it works on larger multi-GPU models. (#21)
- Examples: Added a live J-space chat visualizer — chat with a served model in a generated HTML page where every token is hoverable, showing the streaming top-k Jacobian-lens readout across the captured layers. (#25)
- Examples: The Jacobian-lens fitter stamps provenance (model, corpus, sample count, estimator settings) into the saved lens `.pt`. (#27)
- Docs: Documented the `Hook`/`HookContext` interface and added first-class README sections for activation capture and steering. (#13, #16)

## v1.1.0

- Prior releases.
