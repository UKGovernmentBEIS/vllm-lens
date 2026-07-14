"""Shared utilities for vllm-lens examples."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from transformers import AutoConfig


@lru_cache(maxsize=None)
def get_num_layers(model_name: str) -> int:
    """Return the number of transformer layers for a model.

    Reads the HuggingFace config so the examples work on any model, not
    just the default 32-layer Llama-3.1-8B.  Pass the served model id
    (e.g. ``client.model``).
    """
    config = AutoConfig.from_pretrained(model_name)
    # Multimodal models nest the language-model config under text_config.
    text_config = getattr(config, "text_config", config)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        n = getattr(text_config, attr, None)
        if n is not None:
            return int(n)
    raise ValueError(f"Could not determine layer count for {model_name!r}")


def find_norm(model: Any) -> Any:
    """Find the final layer norm from a vLLM model."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "norm")
    ):
        return model.language_model.model.norm
    return None
