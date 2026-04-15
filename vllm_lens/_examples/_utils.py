"""Shared utilities for vllm-lens examples."""

from __future__ import annotations

from typing import Any

import requests

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
N_LAYERS = 32


def completions(
    base_url: str,
    prompt: str,
    max_tokens: int = 1,
    vllm_xargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict:
    """Send a completion request to the vLLM server."""
    body: dict[str, Any] = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        **kwargs,
    }
    if vllm_xargs:
        body["vllm_xargs"] = vllm_xargs
    return requests.post(f"{base_url}/v1/completions", json=body).json()


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
