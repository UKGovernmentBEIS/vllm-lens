"""Registry-based decoder-layer discovery (``_discover_layer_modules``).

Asserts that layer discovery via vLLM's ``static_forward_context``
registry finds every decoder layer with its correct global index, and
that hooks installed from the discovered map still capture activations
that match HuggingFace transformers.
"""

import gc

import torch
from vllm import LLM, SamplingParams

from .conftest import LAYER_IDX, MODEL_NAME, NUM_LAYERS, PROMPT


def test_registry_discovery_and_capture(hf_model):
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=0.3,
    )
    try:
        # Registry discovery finds all layers with global indices.
        per_rank = llm.collective_rpc("_debug_layer_discovery")
        assert per_rank, "no ranks responded"
        for indices in per_rank:
            assert indices == list(range(NUM_LAYERS)), (
                f"registry discovery returned {indices}"
            )

        # Capture through registry-installed hooks still matches HF.
        model, tokenizer = hf_model
        inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
        num_tokens = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        hf_acts = out.hidden_states[LAYER_IDX + 1][0, :num_tokens].float()

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_residual_stream": [LAYER_IDX]},
        )
        outputs = llm.generate([PROMPT], sampling_params)
        stream = outputs[0].activations["residual_stream"]  # type: ignore[attr-defined]
        vllm_acts = stream[0, :num_tokens].float()

        assert vllm_acts.shape == hf_acts.shape, (
            f"Shape mismatch: vLLM {vllm_acts.shape} vs HF {hf_acts.shape}"
        )
        mean_abs_diff = (vllm_acts - hf_acts.to(vllm_acts.device)).abs().mean().item()
        assert mean_abs_diff < 1e-2, f"Mean abs diff too large: {mean_abs_diff:.6f}"
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
