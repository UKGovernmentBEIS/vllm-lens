# vLLM-lens

vLLM-Lens enables top-down interpretability (e.g., probes, steering, activation oracles). It offers high performance, supporting tensor parallelism & pipeline parallelism (across GPUs and nodes) out of the box. You can also apply all these techniques concurrently (in the same dynamic batch) - removing the need to switch between model instances.

Note this performance comes at the expense of flexibility - for example, you would need to edit the source to add additional custom hooks (though it should be easy enough for coding agents to do that). For more flexibility out of the box, consider [nnsight](https://nnsight.net/) or [TransformerLens](https://transformerlensorg.github.io/TransformerLens/).

The module auto-registers as a [vLLM general plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html) and an [Inspect](https://inspect.aisi.org.uk/) model provider on install. Interact with model internals per-call via `SamplingParams.extra_args` (vLLM) or `GenerateConfig.extra_body` (Inspect).

## Install

```bash
uv add vllm-lens
```

## Examples

These examples use the Inspect integration. See the [`examples/`](examples/) folder for offline and online direct vLLM usage.

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

## Credits

Developed by Alan Cooney, with credit going to Sid Black for the original vLLM worker extension idea.
