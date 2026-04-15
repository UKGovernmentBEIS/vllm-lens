# vLLM-lens

vLLM-Lens enables top-down interpretability (e.g., probes, steering, activation oracles, causal tracing). It offers high performance, supporting tensor parallelism & pipeline parallelism (across GPUs and nodes) out of the box. You can also apply all these techniques concurrently (in the same dynamic batch) - removing the need to switch between model instances.

Features:
- **Activation extraction** — capture residual stream hidden states from specific layers
- **Steering vectors** — add activation vectors to modify the residual stream in-flight
- **Generic hooks** — run arbitrary Python functions per-request, per-layer during inference (inspired by [Garçon](https://transformer-circuits.pub/2021/garcon/index.html))
- **Persistent hooks** — register hooks once, run many requests, collect results in bulk
- **Pre-hooks** — modify layer inputs (e.g., corrupt embeddings for causal tracing)

The module auto-registers as a [vLLM general plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html) and an [Inspect](https://inspect.aisi.org.uk/) model provider on install. Interact with model internals per-call via `SamplingParams.extra_args` (vLLM) or `GenerateConfig.extra_body` (Inspect).

## Install

```bash
uv add vllm-lens
```

## Quickstart

vllm-lens auto-registers as a vLLM plugin — just start a server normally:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

All vllm-lens features (activation extraction, steering, hooks) are available immediately via `SamplingParams.extra_args` (offline) or `vllm_xargs` (HTTP API). The plugin also registers custom HTTP endpoints for persistent hook management at `/v1/hooks/*`.

For the HTTP API, a Python client is provided:

```python
from vllm_lens.client import VLLMLensClient
from vllm_lens import Hook

client = VLLMLensClient("http://localhost:8000")

# Per-request hook
output = client.generate("Hello", max_tokens=10, hooks=[hook])
print(output.hook_results)

# Activation capture
output = client.generate("Hello", capture_layers=[15])
print(output.activations)

# Persistent hooks
client.register_hooks([hook])
client.generate("prompt 1", max_tokens=10)
client.generate("prompt 2", max_tokens=10)
results = client.collect_hook_results()
client.clear_hooks()
```

## Examples

### Generic hooks

Hooks let you run arbitrary Python functions on hidden states at specific layers during inference. They can capture data (via `ctx.saved`) and/or modify hidden states (by returning a tensor).

```python
from vllm import LLM, SamplingParams
from vllm_lens import Hook

def ablate_neuron(ctx, h):
    ctx.saved[f"pre_L{ctx.layer_idx}"] = h[:, 42].cpu()
    h = h.clone()
    h[:, 42] = 0
    return h  # return None to skip modification

hook = Hook(fn=ablate_neuron, layer_indices=[15, 16])
sp = SamplingParams(
    temperature=0.0,
    max_tokens=10,
    extra_args={"apply_hooks": [hook]},
)
outputs = llm.generate(["Hello world"], sp)
print(outputs[0].hook_results)  # {"0": {"pre_L15": tensor, "pre_L16": tensor}}
```

Pre-hooks run before the layer forward pass (useful for corrupting inputs):

```python
def corrupt_embeddings(ctx, h):
    noise = torch.randn_like(h) * 3.0
    return h + noise

hook = Hook(fn=corrupt_embeddings, layer_indices=[0], pre=True)
```

### Persistent hooks (Garçon-style)

Register hooks once, run many requests, collect all results in one bulk transfer:

```python
llm.register_hooks([hook])

for prompt in prompts:
    llm.generate([prompt], sp)  # hooks fire, results stay server-side

results = llm.collect_hook_results()  # bulk retrieval
llm.clear_hooks()
```

Over HTTP (no dev mode required):

```
POST /v1/hooks/register          {"hooks": [...], "prefetch_params": [...]}
POST /v1/completions             (hooks fire automatically)
POST /v1/hooks/collect           → {"results": {<req_id>: ...}}
POST /v1/hooks/clear
POST /v1/hooks/prefetch          {"params": ["lm_head.weight", ...]}
POST /v1/hooks/clear_prefetched
```

Multiple `register` calls append hooks. `collect` is non-destructive. `clear` removes hooks and all accumulated results. Pre-fetched parameters persist independently.

### Accessing model parameters from hooks

Hooks can access model parameters (e.g. `lm_head.weight` for logit lens) via `ctx.get_parameter()`. This auto-gathers across TP ranks:

```python
def logit_lens(ctx, h):
    weight = ctx.get_parameter("lm_head.weight")  # full unsharded weight
    logits = h.float() @ weight.float().T
    ctx.saved["top_ids"] = logits.topk(5).indices.cpu()
    return None
```

With pipeline parallelism, parameters may live on a different PP stage. Pre-fetch them so they're available on all ranks:

```python
# Standalone — works with both per-request and persistent hooks
client.prefetch_params(["lm_head.weight"])
output = client.generate(prompt, hooks=[hook])  # hook can use lm_head.weight

# Or at registration time for persistent hooks
client.register_hooks([hook], prefetch_params=["lm_head.weight"])

# Clean up when done
client.clear_prefetched()
```

Pre-fetched parameters persist until explicitly cleared. When the parameter already exists locally (TP=1, same PP stage), no copy is made.

### Causal tracing (activation patching)

The [`causal_tracing.py`](vllm_lens/_examples/causal_tracing.py) example implements ROME-style causal tracing using pre-hooks for embedding corruption and post-hooks for clean-state restoration:

```bash
python -m vllm_lens._examples.causal_tracing \
    --base-url http://localhost:8000 \
    --prompt "The Eiffel Tower is in the city of" \
    --subject "Eiffel Tower" \
    --answer " Paris"
```

This produces a (layers × tokens) heatmap showing which hidden states are causally important for factual recall.

### Inspect AI provider

An [Inspect AI](https://inspect.ai-safety-institute.org.uk/) model provider is auto-registered as `vllm-lens`, when you install this package. This model provider extends the built-in vLLM provider to serialize `torch.Tensor` steering vectors for HTTP transport and decode base64-encoded activations from responses into `ModelOutput.metadata["activations"]`. It also supports LoRA adaptors.

Usage is the same as the [default vLLM provider](https://inspect.aisi.org.uk/providers.html#vllm) but with the `vllm-lens` prefix (e.g. `vllm-lens/meta-llama/Llama-3.1-1B`).

#### Extracting activations

```python
capture_config = GenerateConfig(
    temperature=0.0,
    max_tokens=1,
    extra_body={
        "extra_args": {"output_residual_stream": extraction_activation_layers},
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
output = await model.generate(state.messages, config=capture_config)
residual_stream = output.metadata["activations"]["residual_stream"]
```

#### Steering with an Activation Oracle

```python
from vllm_lens import SteeringVector

messages = [ChatMessageUser(content=oracle_content)]
oracle_config = GenerateConfig(
    temperature=0.0,
    max_tokens=50,
    extra_body={
        "extra_args": {
            "apply_steering_vectors": [
                SteeringVector(
                    activations=act_vector,
                    layer_indices=[injection_layer],
                    scale=steering_coefficient,
                    norm_match=True,
                    position_indices=[special_pos],
                )
            ],
        },
        "lora_request": {
            "lora_name": "oracle",
            "lora_int_id": 1,
            "lora_path": lora_path,
        },
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
response = await model.generate(messages, config=oracle_config)
```

## Theory

vllm-lens registers as a [vLLM plugin](https://docs.vllm.ai/en/stable/design/plugin_system) and injects itself into vLLM's processing pipeline at broadly 3 stages:

1. **Intercepting generate calls.** To utilise the plugin, you can pass [extra args](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams.extra_args) such as `output_residual_stream` or `apply_steering_vectors` in the sampling parameters. The plugin extracts these, initialises relevant [PyTorch hooks](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html) if they're not already setup (by adding a [worker extension](https://docs.vllm.ai/en/stable/cli/run-batch/?h=worker+extension#-worker-extension-cls)) and sends steering vectors directly to workers (vLLM typically has one worker per GPU).
2. **Per-sample hook operations**. vLLM dynamically batches tokens from multiple concurrent requests into a single forward pass, so a core challenge is "book-keeping" - working out which operations (e.g., activation extraction) should be applied to which parts of the request. To do this we read the `forward_context` metadata, utilising the `query_start_loc` (a tensor of token boundaries per request) and `req_ids` (mapping batch index to request ID). We then, for example, apply hooks to just the slices that correspond to the request. Any extracted activations are moved to CPU ram and compressed (lossless), ready to be requested by the vLLM scheduler process. Steering runs on all tensor-parallel ranks (since it modifies the forward pass), but capture only runs on TP rank 0 (residual streams are identical across TP replicas after all-reduce).
3. **Response collation.** The plugin intercepts the response before it is sent to the client, at which point it queries the relevant vLLM processes for any requested activations. If trims surplus activations, as vLLM does under the hook with tokens (the scheduler often gets ahead of the number of tokens it needs to generate, before stopping). Activations are then returned to the client.

## Running tests

Integration tests in `tests/` run against a live vLLM server. You can either let the fixture start one automatically, or point at an existing server:

```bash
# Option 1: Let pytest start a server (requires GPU, takes a few minutes to boot)
pytest tests/ -v

# Option 2: Start a server yourself, then run tests against it
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
VLLM_TEST_PORT=8000 pytest tests/ -v
```

Environment variables:
- `VLLM_TEST_PORT` — server port (default: `8100`)
- `VLLM_TEST_MODEL` — model to serve (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `VLLM_TEST_STARTUP_TIMEOUT` — seconds to wait for server startup (default: `300`)

Unit tests in `vllm_lens/tests/` use a small model and don't require a running server:

```bash
pytest vllm_lens/tests/ -v
```

## Credits

Developed by Alan Cooney, with credit going to Sid Black for the original vLLM worker extension idea.
