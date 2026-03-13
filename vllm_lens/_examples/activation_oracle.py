"""Activation Oracle demo using vllm-lens.

Runs a target prompt through the base model, captures residual-stream activations,
then queries the oracle LoRA adapter token-by-token to see what the model is
"thinking about" at each position.

Activation capture notes
~~~~~~~~~~~~~~~~~~~~~~~~
Verified correct:
  - vLLM's Qwen3DecoderLayer returns ``(hidden_states, residual)``; the
    true residual stream is ``hidden_states + residual``.  This is
    mathematically equivalent to HF Transformers' single-tensor return
    where the residual additions happen inside the layer.
  - Layer indexing (0-based) matches between vLLM and HF.
  - Token IDs produced by ``get_target_prompt_input_ids`` are identical
    across frameworks.

Known cross-framework differences (Qwen3-8B-FP8, layer 18):
  Comparing vLLM activations against an HF reference gives ~0.03–0.27
  mean-abs-diff per position (cosine similarity ~0.997+).  Three sources
  compound over 18 layers:
    1. **FP8 matmul kernels** — vLLM custom CUDA kernels vs HF/PyTorch.
    2. **Attention backends** — FLASH_ATTN (vLLM) vs SDPA (HF).
    3. **LoRA implementations** — Punica GPU (vLLM) vs PEFT (HF).
  These are inherent cross-framework differences, not capture bugs.

LoRA during collection:
  The original ``ao_original_demo.py`` happens to have the oracle LoRA
  adapter active when it captures target activations (PEFT's
  ``inject_adapter`` re-sets whichever adapter was last loaded).  However,
  the Activation Oracles paper (arXiv:2512.15674) describes collecting
  activations from the **base model** — the oracle LoRA is only used when
  *querying*, not when capturing.  We follow the paper here and do NOT
  apply the LoRA during collection.
"""

import asyncio
import json
import os

os.environ.setdefault("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True")
import random
import string
from typing import Annotated

import pandas as pd
import torch
import typer
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from transformers import PreTrainedTokenizerBase
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.tokenizers import TokenizerLike

from vllm_lens import SteeringVector

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen3-8B"
LORA_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
LAYER = 18

TARGET_MESSAGES = [
    {
        "role": "user",
        "content": (
            "The philosopher who drank hemlock taught a student who founded "
            "an academy. That student's most famous pupil was"
        ),
    },
]
ORACLE_QUESTION = "Can you name all people the model is currently thinking about?"


# ============================================================
# AOConfig
# ============================================================


class AOConfig(BaseModel, extra="allow"):
    hook_onto_layer: int
    special_token: str
    prefix_template: str = "Layer: {layer}\n{special_tokens} \n"
    steering_coefficient: float
    act_layer_combinations: list[list[int]]

    @property
    def act_layers(self) -> list[int]:
        return [layer for group in self.act_layer_combinations for layer in group]


def load_ao_config(lora_path: str) -> AOConfig:
    config_path = hf_hub_download(repo_id=lora_path, filename="ao_config.json")
    with open(config_path) as f:
        return AOConfig(**json.load(f))


# ============================================================
# Prompt construction
# ============================================================


def get_target_prompt_input_ids(
    target_messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """Tokenize target chat messages into prompt token IDs."""
    result = tokenizer.apply_chat_template(
        target_messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    # Some tokenizers return a BatchEncoding; extract input_ids.
    if not isinstance(result, list):
        return result["input_ids"]
    return result


def create_oracle_prompt(
    config: AOConfig,
    layer: int,
    num_positions: int,
    oracle_question: str,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    template = config.prefix_template.replace(
        "{special_token} * {num_positions}", "{special_tokens}"
    ).replace("\\n", "\n")
    prefix = template.format(
        special_tokens=config.special_token * num_positions,
        layer=layer,
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prefix + oracle_question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ============================================================
# Activation collection
# ============================================================


async def collect_activations(
    engine: AsyncLLMEngine,
    input_ids: list[int],
    layer: int,
) -> torch.Tensor:
    """Capture residual-stream activations from the **base model** (no LoRA).

    The oracle LoRA is only used later when querying — the paper
    (arXiv:2512.15674) collects activations from the unmodified model.
    The original ``ao_original_demo.py`` inadvertently has the oracle LoRA
    active during collection (PEFT's ``inject_adapter`` re-sets whichever
    adapter was last loaded), but we follow the paper here.
    """
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={"output_residual_stream": [layer]},
    )

    final = None
    async for output in engine.generate(
        {"prompt_token_ids": input_ids},
        sampling_params,
        "target-0",
    ):
        final = output

    return final.activations["residual_stream"]  # type: ignore[union-attr]


# ============================================================
# Oracle query
# ============================================================


async def query_oracle(
    activation_vector: torch.Tensor,
    config: AOConfig,
    engine: AsyncLLMEngine,
    tokenizer: PreTrainedTokenizerBase,
    lora_path: str,
    layer: int,
    question: str,
) -> str:
    num_positions = activation_vector.shape[1]
    oracle_prompt = create_oracle_prompt(
        config, layer, num_positions, question, tokenizer
    )
    oracle_token_ids: list[int] = tokenizer.encode(
        oracle_prompt, add_special_tokens=True
    )

    special_token_id: int = tokenizer.encode(
        config.special_token, add_special_tokens=False
    )[0]
    special_token_start_pos = oracle_token_ids.index(special_token_id)

    steering_vectors = [
        SteeringVector(
            activations=activation_vector,
            layer_indices=[config.hook_onto_layer],
            scale=config.steering_coefficient,
            norm_match=True,
            position_indices=[
                special_token_start_pos + i for i in range(num_positions)
            ],
        )
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
        extra_args={"apply_steering_vectors": steering_vectors},
    )

    final = None
    rand_id = "".join(random.choices(string.ascii_letters, k=5))
    lora_request = LoRARequest("oracle", 1, lora_path)
    async for output in engine.generate(
        {"prompt_token_ids": oracle_token_ids},
        sampling_params,
        rand_id,
        lora_request=lora_request,
    ):
        final = output
    return final.outputs[0].text.strip()  # type: ignore[union-attr]


# ============================================================
# Oracle sweep
# ============================================================


async def run_oracle_sweep(
    engine: AsyncLLMEngine,
    tokenizer: TokenizerLike,
    ao_config: AOConfig,
    target_prompt_ids: list[int],
    residual_stream: torch.Tensor,
    layer: int = LAYER,
    oracle_question: str = ORACLE_QUESTION,
) -> pd.DataFrame:
    """Run the oracle on every token position and return a DataFrame of results."""
    tasks = []
    decoded_tokens = []

    for token_idx, token in enumerate(target_prompt_ids):
        token_decoded = tokenizer.decode([token])  # type: ignore[arg-type]
        decoded_tokens.append(token_decoded)

        act_vector = residual_stream[:, token_idx].unsqueeze(0)
        tasks.append(
            query_oracle(
                act_vector,
                ao_config,
                engine,
                tokenizer,
                LORA_PATH,
                layer,
                oracle_question,
            )
        )

    results = await asyncio.gather(*tasks)
    return pd.DataFrame(
        {
            "token": decoded_tokens,
            "oracle_response": results,
        }
    )


# ============================================================
# Main
# ============================================================


async def main(device: int = 0) -> pd.DataFrame:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    ao_config = load_ao_config(LORA_PATH)

    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=4096,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer: TokenizerLike = engine.tokenizer  # type: ignore[assignment]

    target_prompt_ids = get_target_prompt_input_ids(TARGET_MESSAGES, tokenizer)  # type: ignore[arg-type]

    residual_stream = await collect_activations(engine, target_prompt_ids, LAYER)

    return await run_oracle_sweep(
        engine, tokenizer, ao_config, target_prompt_ids, residual_stream
    )


app = typer.Typer()


@app.command()
def cli(
    device: Annotated[int, typer.Option(help="CUDA device index to use.")] = 0,
) -> None:
    df = asyncio.run(main(device=device))

    console = Console()
    table = Table(title="Activation Oracle — Token-by-Token Sweep")
    table.add_column("Token", style="cyan")
    table.add_column("Oracle Response", style="green")
    for _, row in df.iterrows():
        table.add_row(row["token"], row["oracle_response"])
    console.print(table)


if __name__ == "__main__":
    app()
