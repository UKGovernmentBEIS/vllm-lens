"""HTTP client for vllm-lens, mirroring the LLM interface.

Wraps the vLLM OpenAI-compatible API and vllm-lens hook endpoints,
handling serialization of hooks, steering vectors, and activations.

Usage::

    from vllm_lens.client import VLLMLensClient
    from vllm_lens import Hook

    client = VLLMLensClient("http://localhost:8000")

    # Per-request hooks
    output = client.generate("Hello", max_tokens=10, hooks=[hook])
    print(output.hook_results)

    # Persistent hooks
    client.register_hooks([hook])
    client.generate("Hello", max_tokens=10)
    client.generate("World", max_tokens=10)
    results = client.collect_hook_results()
    client.clear_hooks()

    # Activation capture
    output = client.generate("Hello", max_tokens=10, capture_layers=[15])
    print(output.activations)

    # Steering
    output = client.generate("Hello", max_tokens=10, steering_vectors=[sv])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import requests
import torch

from vllm_lens._helpers._serialize import (
    deserialize_hook_results,
    deserialize_tensor,
)
from vllm_lens._helpers.types import Hook, SteeringVector


@dataclass
class GenerateOutput:
    """Output from a generate call, mirroring relevant fields of vLLM's RequestOutput."""

    text: str
    """Generated text."""

    activations: dict[str, torch.Tensor] | None = None
    """Captured residual stream activations, if requested."""

    hook_results: dict[str, dict[str, Any]] | None = None
    """Per-request hook results (ctx.saved dicts), if hooks were passed."""

    logprobs: dict[str, Any] | None = None
    """Log-probabilities, if requested."""

    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    """The full raw JSON response from the server."""


class VLLMLensClient:
    """HTTP client for a vLLM server with vllm-lens installed.

    Provides the same interface as the patched ``LLM`` class but
    communicates over HTTP.
    """

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._model = model
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def model(self) -> str:
        """The model name served by the server."""
        if self._model is None:
            resp = self._session.get(f"{self.base_url}/v1/models").json()
            self._model = resp["data"][0]["id"]
        return self._model

    def _build_xargs(
        self,
        hooks: list[Hook] | None,
        capture_layers: list[int] | None,
        steering_vectors: list[SteeringVector] | None,
    ) -> dict[str, str]:
        xargs: dict[str, str] = {}
        if capture_layers is not None:
            xargs["output_residual_stream"] = json.dumps(capture_layers)
        if hooks is not None:
            xargs["apply_hooks"] = json.dumps([h.model_dump() for h in hooks])
        if steering_vectors is not None:
            xargs["apply_steering_vectors"] = json.dumps(
                [sv.model_dump() for sv in steering_vectors]
            )
        return xargs

    def _parse_response(self, resp: dict[str, Any]) -> GenerateOutput:
        if "error" in resp:
            raise RuntimeError(resp["error"].get("message", resp["error"]))

        choice = resp["choices"][0]
        text = choice.get("message", {}).get("content") or choice.get("text", "")

        activations = None
        if "activations" in resp:
            activations = {
                name: deserialize_tensor(encoded)
                for name, encoded in resp["activations"].items()
            }

        hook_results = None
        if "hook_results" in resp:
            hook_results = deserialize_hook_results(resp["hook_results"])

        lp = choice.get("logprobs")

        return GenerateOutput(
            text=text,
            activations=activations,
            hook_results=hook_results,
            logprobs=lp,
            raw=resp,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
        hooks: list[Hook] | None = None,
        capture_layers: list[int] | None = None,
        steering_vectors: list[SteeringVector] | None = None,
        logprobs: int | None = None,
        echo: bool = False,
        **kwargs: Any,
    ) -> GenerateOutput:
        """Generate a completion from a raw prompt.

        Uses ``/v1/completions``. For chat messages, use :meth:`chat`.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        if logprobs is not None:
            body["logprobs"] = logprobs
        if echo:
            body["echo"] = True

        xargs = self._build_xargs(hooks, capture_layers, steering_vectors)
        if xargs:
            body["vllm_xargs"] = xargs

        resp = self._session.post(f"{self.base_url}/v1/completions", json=body).json()
        return self._parse_response(resp)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
        hooks: list[Hook] | None = None,
        capture_layers: list[int] | None = None,
        steering_vectors: list[SteeringVector] | None = None,
        **kwargs: Any,
    ) -> GenerateOutput:
        """Generate a chat completion from a list of messages.

        Uses ``/v1/chat/completions``.

        Example::

            output = client.chat([
                {"role": "user", "content": "What is the capital of France?"}
            ], hooks=[hook])
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        xargs = self._build_xargs(hooks, capture_layers, steering_vectors)
        if xargs:
            body["vllm_xargs"] = xargs

        resp = self._session.post(
            f"{self.base_url}/v1/chat/completions", json=body
        ).json()
        return self._parse_response(resp)

    # ------------------------------------------------------------------
    # Persistent hooks
    # ------------------------------------------------------------------

    def register_hooks(
        self,
        hooks: list[Hook],
        prefetch_params: list[str] | None = None,
    ) -> None:
        """Register persistent hooks (appends to existing).

        Args:
            hooks: Hooks to register.
            prefetch_params: Parameter names to pre-fetch across all ranks
                (TP + PP).  Needed for ``ctx.get_parameter()`` with PP.
        """
        body: dict[str, Any] = {"hooks": [h.model_dump() for h in hooks]}
        if prefetch_params:
            body["prefetch_params"] = prefetch_params
        resp = self._session.post(
            f"{self.base_url}/v1/hooks/register",
            json=body,
        ).json()
        if resp.get("status") != "ok":
            raise RuntimeError(f"Failed to register hooks: {resp}")

    def collect_hook_results(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Collect all accumulated persistent hook results.

        Returns ``{request_id: {hook_index: ctx.saved}}``.
        Non-destructive — call :meth:`clear_hooks` to clean up.
        """
        resp = self._session.post(f"{self.base_url}/v1/hooks/collect").json()
        results = resp.get("results", {})
        return {
            req_id: deserialize_hook_results(hook_data)
            for req_id, hook_data in results.items()
        }

    def clear_hooks(self) -> None:
        """Remove all persistent hooks and accumulated results."""
        self._session.post(f"{self.base_url}/v1/hooks/clear")

    # ------------------------------------------------------------------
    # Parameter prefetch
    # ------------------------------------------------------------------

    def prefetch_params(self, names: list[str]) -> None:
        """Pre-fetch model parameters across all TP/PP ranks.

        Makes the parameters available to ``ctx.get_parameter()`` in
        hooks on any layer, even with pipeline parallelism.  Parameters
        persist until :meth:`clear_prefetched` is called.
        """
        resp = self._session.post(
            f"{self.base_url}/v1/hooks/prefetch",
            json={"params": names},
        ).json()
        if resp.get("status") != "ok":
            raise RuntimeError(f"Failed to prefetch parameters: {resp}")

    def clear_prefetched(self) -> None:
        """Remove all pre-fetched parameters."""
        self._session.post(f"{self.base_url}/v1/hooks/clear_prefetched")
