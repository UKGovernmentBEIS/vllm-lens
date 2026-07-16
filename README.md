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

Runnable examples live in [`examples/`](examples/) — each is standalone; run any
with `python examples/<name>.py --help`. Most (`causal_tracing`, `logit_lens`,
`deception_probe`, `emotion_tracker`) drive a **separately running** vllm-lens
server over HTTP via [`VLLMLensClient`](vllm_lens/client.py) (start one with
`vllm serve <model>`, then pass `--base-url`); `activation_oracle` instead spins
up its own in-process engine. The notebooks are self-contained.

| Example | What it demonstrates |
| --- | --- |
| [`causal_tracing.py`](examples/causal_tracing.py) | ROME-style causal tracing — pre-hook embedding corruption + post-hook clean-state patching, rendered as a (layer × token) heatmap. |
| [`logit_lens.py`](examples/logit_lens.py) | Logit lens — project each layer's hidden states through the final norm + unembedding via `ctx.get_parameter("lm_head.weight")`. |
| [`jacobian_lens.py`](examples/jacobian_lens.py) | Jacobian lens / J-space ([Anthropic global-workspace](https://transformer-circuits.pub/2026/workspace)) — like the logit lens, but transports each layer's residual through a pre-fitted average Jacobian `J_l` before unembedding, reading out what the model is "disposed to say". Readout + visualization; the lens is fit by [`jacobian_lens_fit.py`](examples/jacobian_lens_fit.py). |
| [`deception_probe.py`](examples/deception_probe.py) | Apollo-style linear deception probe — contrastive activation extraction with persistent hooks, then a pure-torch logistic-regression probe. |
| [`emotion_tracker.py`](examples/emotion_tracker.py) | Anthropic emotion-concepts replication — emotion direction vectors + per-token projection tracking, with an interactive HTML visualization. |
| [`activation_oracle.py`](examples/activation_oracle.py) | Activation Oracle steering (arXiv:2512.15674) — capture activations and steer a LoRA oracle to describe them. |
| [`extract_residual_stream.ipynb`](examples/extract_residual_stream.ipynb) | Notebook — per-request and persistent activation capture, offline and over HTTP. |
| [`inspect-demo.ipynb`](examples/inspect-demo.ipynb) | Notebook — Activation Oracle via the vllm-lens Inspect provider. |
| [`benchmark.ipynb`](examples/benchmark.ipynb) | Notebook — overhead of activation capture and steering. |

### Activation capture

Capture the residual stream at chosen layers by passing `output_residual_stream` (a list of layer indices) in `extra_args`:

```python
from vllm import LLM, SamplingParams

llm = LLM("meta-llama/Llama-3.1-8B-Instruct")
sp = SamplingParams(max_tokens=1, extra_args={"output_residual_stream": [15, 20]})
out = llm.generate(["Hello world"], sp)
acts = out[0].activations["residual_stream"]  # (n_layers, n_positions, hidden_dim)
```

Over the HTTP server, pass it in `vllm_xargs` — or use the client's `capture_layers`:

```python
from vllm_lens.client import VLLMLensClient

client = VLLMLensClient("http://localhost:8000")
out = client.generate("Hello world", capture_layers=[15, 20])
print(out.activations["residual_stream"].shape)
```

Layers are stacked in ascending order along dim 0. Capture runs on TP rank 0 only (residual streams are identical across TP ranks after all-reduce).

### Steering vectors

Add activation vectors to the residual stream in-flight with `apply_steering_vectors`. A `SteeringVector` carries the activations plus how to apply them:

```python
import torch
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector

sv = SteeringVector(
    activations=torch.randn(1, 4096),  # (n_layers, hidden) or (n_layers, n_positions, hidden)
    layer_indices=[15],
    scale=4.0,
    norm_match=True,        # scale the added vector so its magnitude is ‖residual‖ · scale
    position_indices=None,  # None = all positions (2D) or sequential 0..n-1 (3D)
)
sp = SamplingParams(max_tokens=20, extra_args={"apply_steering_vectors": [sv]})
llm.generate(["I think the best dessert is"], sp)
```

Via the client (or `vllm_xargs` over HTTP):

```python
client.generate("I think the best dessert is", steering_vectors=[sv])
```

### Generic hooks

Hooks let you run arbitrary Python functions on hidden states at specific layers during inference. They can capture data (via `ctx.saved`) and/or modify hidden states (by returning a tensor).

**Interface.** A hook is `Hook(fn, layer_indices, pre=False)`:

| Field | Meaning |
| --- | --- |
| `fn(ctx, hidden_states) -> Tensor \| None` | Called once per hooked layer, per request. `hidden_states` is that request's slice, shape `(seq_len, hidden_dim)`. Return a tensor to **replace** it, or `None` to leave it unchanged. |
| `layer_indices: list[int]` | Which layers to run on. |
| `pre: bool` | If `True`, run as a *pre-hook* — before the layer forward pass, on its input — instead of after. |

The `ctx` (a `HookContext`) passed to `fn` exposes:

| Attribute | Meaning |
| --- | --- |
| `ctx.saved: dict[str, Any]` | Scratch dict that persists across layers **and** forward passes for this hook. Returned to the client as `hook_results["<hook_index>"]` (index = the hook's position in the list you passed). |
| `ctx.layer_idx: int` | The layer index currently firing. |
| `ctx.seq_len: int` | Number of tokens in this request's slice. |
| `ctx.model` | The underlying model (for architecture-specific access). |
| `ctx.get_parameter(name) -> Tensor` | Fetch a full model parameter, gathered across TP/PP ranks — e.g. `ctx.get_parameter("lm_head.weight")`. With pipeline parallelism, `prefetch_params` the name first (see below). |

`fn` runs on **every** tensor-parallel rank, so it must be deterministic across ranks (seed any randomness). For HTTP transport `fn` is serialized with cloudpickle — i.e. **arbitrary code execution** on the server, so only use with trusted clients.

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

The [`causal_tracing.py`](examples/causal_tracing.py) example implements ROME-style causal tracing using pre-hooks for embedding corruption and post-hooks for clean-state restoration:

```bash
python examples/causal_tracing.py \
    --base-url http://localhost:8000 \
    --prompt "The Eiffel Tower is in the city of" \
    --subject "Eiffel Tower" \
    --answer " Paris"
```

This produces a (layers × tokens) heatmap showing which hidden states are causally important for factual recall.

### Jacobian lens / J-space

The Jacobian lens extends the logit lens: it transports each layer's residual through a pre-fitted average Jacobian `J_l = E[∂h_final/∂h_l]` before the norm + unembedding, reading out the tokens a model is *disposed toward*. The full flow is **fit → serve → read out + visualize**, and the fit and readout run in **two different environments** connected only by the fitted `lens.pt` (format: `{J, source_layers, d_model}`, identical in both):

1. **Fit** `J_l` with [`jacobian_lens_fit.py`](examples/jacobian_lens_fit.py) — the single source of truth for fitting. It needs the backward pass and runs in the **prime-rl fit env** (torch 2.11/cu128 + torchtitan; not the vllm-lens env), on prime-rl's FSDP2 + expert-parallel stack. This scales from small dense models on one GPU up to large MoE across many GPUs / nodes — GLM-4.5-Air (110B) is validated (cosine 1.0 vs a single-GPU reference), and the same path serves GLM-5.2. Build the env once with [`jacobian_lens_fit_env.sh`](examples/jacobian_lens_fit_env.sh).
2. **Serve** the model under vllm-lens (the readout env).
3. **Read out** with [`jacobian_lens.py`](examples/jacobian_lens.py) — forward-only, applied in a hook on the vLLM worker, correct under TP/PP/EP.
4. **Chat interactively** with [`jacobian_lens_chat.py`](examples/jacobian_lens_chat.py) — generates a self-contained HTML page where you chat with the served model and watch the top-k J-space readout at a layer you pick live.

```bash
# 1. fit J_l (once; cached to lens.pt). In the prime-rl fit env — see
#    examples/jacobian_lens_fit_env.sh to build it. Single node:
cd /path/to/prime-rl && unset VIRTUAL_ENV
uv run --no-sync torchrun --nproc-per-node=8 \
    /path/to/vllm-lens/examples/jacobian_lens_fit.py \
    --model zai-org/GLM-4.5-Air --layers 25,33,40 --ep 8 --out lens.pt
#    Multi-node (torchrun; run on each node, differing only in --node-rank):
#    torchrun --nnodes=2 --node-rank=0 --nproc-per-node=8 \
#        --rdzv-backend=c10d --rdzv-endpoint=$HEAD_IP:29500 \
#        examples/jacobian_lens_fit.py --model ... --ep 16 --out lens.pt
#    (SLURM: set --nnodes/--node-rank/--rdzv-endpoint from srun env — see the
#    jacobian_lens_fit.py module docstring.)

# 2. serve it under vllm-lens (V1 runner so hooks work)
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve zai-org/GLM-4.5-Air

# 3. read the lens out live against the server (vllm-lens env)
python examples/jacobian_lens.py run --lens lens.pt \
    --prompt "The Eiffel Tower is located in the city of" \
    --layers 25,33,40

# 4. or build an interactive HTML: chat + live J-space readout at a chosen layer
python examples/jacobian_lens_chat.py --lens lens.pt \
    --base-url http://localhost:8000 --out jacobian_lens_chat
```

`run` prints the top-1 J-lens token at each (layer, position); with `--grid-out FILE` it also writes a static top-k grid, one subplot per layer (needs matplotlib). `jacobian_lens_chat.py` writes a self-contained HTML page: send a message and the response's top-k J-space tokens are shown per position, with a dropdown to switch the readout layer instantly.

`--layers` picks which layers to read out (the lens is shipped to the worker, so keep it modest on large models — and fit only the layers you'll read out); `--k` sets how many tokens per position; `--baseline` drops the `J_l` transport for a logit-lens comparison; `--norm-weight` overrides the auto-detected final-norm weight name. `run --lens` also loads a pre-fitted Hub lens (e.g. [Neuronpedia](https://huggingface.co/neuronpedia/jacobian-lens)).

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

1. **Intercepting generate calls.** To utilise the plugin, you can pass [extra args](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams.extra_args) such as `output_residual_stream`, `apply_steering_vectors`, or `apply_hooks` in the sampling parameters. The plugin extracts these, initialises relevant [PyTorch hooks](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html) if they're not already setup (by adding a [worker extension](https://docs.vllm.ai/en/stable/cli/run-batch/?h=worker+extension#-worker-extension-cls)) and sends steering vectors and hook definitions directly to workers (vLLM typically has one worker per GPU).
2. **Per-sample hook operations**. vLLM dynamically batches tokens from multiple concurrent requests into a single forward pass, so a core challenge is "book-keeping" - working out which operations (e.g., activation extraction) should be applied to which parts of the request. To do this we read the `forward_context` metadata, utilising the `query_start_loc` (a tensor of token boundaries per request) and `req_ids` (mapping batch index to request ID). We then, for example, apply hooks to just the slices that correspond to the request. Any extracted activations are moved to CPU ram and compressed (lossless), ready to be requested by the vLLM scheduler process. Steering runs on all tensor-parallel ranks (since it modifies the forward pass), but capture only runs on TP rank 0 (residual streams are identical across TP replicas after all-reduce).
3. **Response collation.** The plugin intercepts the response before it is sent to the client, at which point it queries the relevant vLLM processes for any requested activations. It trims surplus activations, since vLLM can run an extra forward pass under the hood (the scheduler often gets ahead of the number of tokens it needs to generate, before stopping). Activations are then returned to the client.

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
- `VLLM_TEST_TP_SIZE` — tensor parallel size (default: `1`)
- `VLLM_TEST_PP_SIZE` — pipeline parallel size (default: `1`)
- `VLLM_TEST_STARTUP_TIMEOUT` — seconds to wait for server startup (default: `900`)

Unit tests in `vllm_lens/tests/` use a small model and manage their own in-process vLLM engine, so they don't need a separately-running server — but they still require a GPU:

```bash
pytest vllm_lens/tests/ -v
```

## Credits

Developed by Alan Cooney, with credit going to Sid Black for the original vLLM worker extension idea.
