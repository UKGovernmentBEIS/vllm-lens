# vLLM-lens

vLLM-Lens is designed to enable top-down interpretability (e.g., probes, steering, activation oracles). It is designed to offer high performance, supporting tensor parallelism & pipeline parallelism (across GPUs and nodes) out of the box. It's also designed so you can apply all these techniques concurrently (in the same dynamic batch) - removing the need to switch between model instances.

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

Ships a vLLM worker extension that adds hooks to the model in the underlying workers (credit to Sid Black for pioneering this), plus a plugin to track which activations belong to which prompts across vLLM's dynamic batching with paged attention (similar to the nnsight approach).

## Credits

Developed by Alan Cooney, with credit going to Sid Black for the original vLLM worker extension idea.
