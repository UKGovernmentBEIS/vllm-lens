"""Native HuggingFace Transformers activation extraction benchmark."""

import time
from typing import Annotated

import torch
import typer
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.types import BenchmarkConfig, BenchmarkResult

app = typer.Typer()


def _resolve_layer(
    model: torch.nn.Module, layer_prefix: str, layer_idx: int
) -> torch.nn.Module:
    """Traverse dotted layer_prefix to reach the target decoder layer."""
    module = model
    for attr in layer_prefix.split("."):
        module = getattr(module, attr)
    return module[layer_idx]


def load_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    layer_prefix: str,
    layer: int,
    batch_size: int,
    max_new_tokens: int,
) -> list[torch.Tensor]:
    target_layer = _resolve_layer(model, layer_prefix, layer)
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    model.eval()
    all_acts: list[torch.Tensor] = []

    for i in tqdm(
        range(0, len(prompts), batch_size), total=n_batches, desc="hf-transformers"
    ):
        batch = prompts[i : i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            # Pass 1: generate full sequences
            full_tokens = model.generate(
                **tokens, max_new_tokens=max_new_tokens, do_sample=False
            )

            # Pass 2: extract activations over the complete sequences
            captured: list[torch.Tensor] = []

            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured.append(hidden.detach())

            handle = target_layer.register_forward_hook(hook_fn)
            model(full_tokens)
            handle.remove()
            all_acts.extend(c.cpu() for c in captured)
    return all_acts


@app.command()
def main(
    config_file: Annotated[str, typer.Option(help="Path to JSON BenchmarkConfig file")],
) -> None:
    """Benchmark native HuggingFace Transformers activation extraction."""
    from pathlib import Path

    load_dotenv()
    cfg = BenchmarkConfig.model_validate_json(Path(config_file).read_text())

    if cfg.tensor_parallelism > 1 or cfg.pipeline_parallelism > 1:
        raise typer.BadParameter(
            f"HF Transformers does not support parallelism (got tp={cfg.tensor_parallelism}, pp={cfg.pipeline_parallelism})"
        )

    ds = load_dataset(cfg.dataset, split="train")
    prompts = [row["instruction"] for row in ds.select(range(cfg.samples))]

    t0 = time.perf_counter()
    model, tokenizer = load_model(cfg.model)
    startup_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    for batch_size in (512, 256, 128, 64, 32, 16, 8, 4, 2):
        try:
            torch.cuda.empty_cache()
            all_acts = extract_activations(
                model,
                tokenizer,
                prompts,
                cfg.layer_prefix,
                cfg.layer,
                batch_size,
                cfg.max_new_tokens,
            )
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size == 2:
                raise
            typer.echo(
                f"OOM at batch_size={batch_size}, retrying with {batch_size // 2}"
            )
            torch.cuda.empty_cache()
            t1 = time.perf_counter()
    run_time = time.perf_counter() - t1

    # Compute activation shape metadata (after timing).
    # Each tensor has shape (batch, seq_len, d_model).
    n_activation_vectors = sum(a.shape[0] for a in all_acts)
    total_len = sum(a.shape[0] * a.shape[1] for a in all_acts)
    average_len = total_len / n_activation_vectors
    d_model = all_acts[0].shape[-1]

    result = BenchmarkResult(
        lib_name="HF-Transformers",
        model=cfg.model,
        n_samples=cfg.samples,
        startup_time=startup_time,
        run_time=run_time,
        batch_size=batch_size,
        n_activation_vectors=n_activation_vectors,
        average_len=average_len,
        d_model=d_model,
    )
    out_path = result.save()
    typer.echo(f"{result}  → {out_path}")


if __name__ == "__main__":
    app()
