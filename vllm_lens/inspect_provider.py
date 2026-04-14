"""Custom Inspect AI model provider for vllm-lens."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Any

from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.model._providers.vllm import VLLMAPI
from inspect_ai.model._registry import modelapi
from inspect_ai.tool._tool_choice import ToolChoice
from inspect_ai.tool._tool_info import ToolInfo
from typing_extensions import override

from vllm_lens._helpers._serialize import (
    deserialize_hook_results,
    deserialize_tensor,
)
from vllm_lens._helpers.types import Hook, SteeringVector

logger = logging.getLogger(__name__)

# Bridge between on_response() and generate() within the same async call.
_pending_activations: ContextVar[dict[str, Any] | None] = ContextVar(
    "_pending_activations", default=None
)
_pending_token_ids: ContextVar[dict[str, list[int]] | None] = ContextVar(
    "_pending_token_ids", default=None
)
_pending_hook_results: ContextVar[dict[str, Any] | None] = ContextVar(
    "_pending_hook_results", default=None
)

DEFAULT_ATTEMPT_TIMEOUT = 3600  # 1 hour


@modelapi(name="vllm-lens")
class VLLMLensAPI(VLLMAPI):
    """Inspect AI model provider for vllm-lens.

    Registered as ``"vllm-lens"`` so that
    ``get_model("vllm-lens/model-name")`` automatically handles
    activation capture and steering vector serialization.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **server_args: Any,
    ) -> None:
        if config.attempt_timeout is None:
            config = config.merge(
                GenerateConfig(attempt_timeout=DEFAULT_ATTEMPT_TIMEOUT)
            )

        # Inspect's start_local_server does ``--key value`` for every arg,
        # but vLLM boolean flags (e.g. --enable-lora) are argparse
        # store_true/store_false and don't accept a value.  Fix this by
        # converting True → None (bare flag) and dropping False/None entries.
        server_args = {
            k: (None if v is True else v)
            for k, v in server_args.items()
            if v is not False and v is not None
        }

        super().__init__(
            model_name=model_name,
            base_url=base_url,
            port=port,
            api_key=api_key,
            config=config,
            **server_args,
        )

    @override
    def on_response(self, response: dict[str, Any]) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Capture serialized activations and token IDs from the HTTP response.

        ``OpenAICompatibleAPI.generate()`` calls this with
        ``completion.model_dump()`` which, thanks to pydantic's
        ``extra="allow"``, includes extra keys like ``"activations"``
        (injected by the vllm-lens server plugin) and ``"prompt_token_ids"``
        / ``"token_ids"`` (returned by vLLM when ``return_token_ids`` is set).
        """
        _pending_activations.set(response.get("activations"))
        _pending_hook_results.set(response.get("hook_results"))

        token_id_data: dict[str, list[int]] = {}
        prompt_token_ids = response.get("prompt_token_ids")
        if prompt_token_ids is not None:
            token_id_data["prompt_token_ids"] = prompt_token_ids
        choices = response.get("choices")
        if choices:
            choice_token_ids = choices[0].get("token_ids")
            if choice_token_ids is not None:
                token_id_data["token_ids"] = choice_token_ids
        _pending_token_ids.set(token_id_data or None)

    @override
    async def generate(  # pyright: ignore[reportGeneralTypeIssues]
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        config = self._transform_config(config)

        # Always request token IDs from vLLM.
        if config.extra_body is None:
            config = config.model_copy()
            config.extra_body = {"return_token_ids": True}
        elif "return_token_ids" not in config.extra_body:
            config = config.model_copy(deep=True)
            config.extra_body["return_token_ids"] = True  # type: ignore

        token_act = _pending_activations.set(None)
        token_tid = _pending_token_ids.set(None)
        token_hr = _pending_hook_results.set(None)
        try:
            result = await super().generate(input, tools, tool_choice, config)

            metadata: dict[str, Any] = {}

            raw = _pending_activations.get()
            if raw is not None:
                metadata["activations"] = {
                    name: deserialize_tensor(encoded) for name, encoded in raw.items()
                }

            raw_hr = _pending_hook_results.get()
            if raw_hr is not None:
                metadata["hook_results"] = deserialize_hook_results(raw_hr)

            tid = _pending_token_ids.get()
            if tid is not None:
                metadata.update(tid)

            if metadata:
                result = self._inject_metadata(result, metadata)

            # Strip large fields from the ModelCall response dict so they
            # don't bloat eval logs.  (on_response() .pop() is too late —
            # Inspect deep-copies the response into ModelCall before calling
            # on_response().)
            if isinstance(result, tuple):
                self._strip_model_call_response(result[1])

            return result
        finally:
            _pending_activations.reset(token_act)
            _pending_token_ids.reset(token_tid)
            _pending_hook_results.reset(token_hr)

    @staticmethod
    def _transform_config(config: GenerateConfig) -> GenerateConfig:
        """Translate the user-friendly extra_body layout to vLLM server format.

        Transformations:

        - ``extra_args`` → ``vllm_xargs`` (serialize any Tensor steering vectors)
        - ``lora_request`` → ``model`` override (uses ``lora_name``)
        """
        if config.extra_body is None:
            return config

        extra_args = config.extra_body.get("extra_args")
        lora_request = config.extra_body.get("lora_request")
        if extra_args is None and lora_request is None:
            return config

        config = config.model_copy(deep=True)

        # extra_args → vllm_xargs with serialized steering vectors and hooks
        if extra_args is not None:
            del config.extra_body["extra_args"]  # type: ignore[reportOptionalSubscript]
            vectors: list[SteeringVector] | None = extra_args.get(
                "apply_steering_vectors"
            )
            if vectors is not None:
                # Client-side validation + serialization via Pydantic.
                # model_dump() invokes @field_serializer to base64-encode tensors.
                extra_args = dict(extra_args)
                extra_args["apply_steering_vectors"] = json.dumps(
                    [sv.model_dump() for sv in vectors]
                )
            hooks: list[Hook] | None = extra_args.get("apply_hooks")
            if hooks is not None:
                # model_dump() invokes @field_serializer to cloudpickle fn.
                if extra_args is vectors:  # not yet copied
                    extra_args = dict(extra_args)
                extra_args["apply_hooks"] = json.dumps(
                    [h.model_dump() for h in hooks]
                )
            config.extra_body["vllm_xargs"] = extra_args  # type: ignore[reportOptionalSubscript]

        # lora_request → model name override
        if lora_request is not None:
            del config.extra_body["lora_request"]  # type: ignore[reportOptionalSubscript]
            config.extra_body["model"] = lora_request["lora_name"]  # type: ignore[reportOptionalSubscript]

        return config

    @staticmethod
    def _inject_metadata(
        result: ModelOutput | tuple[ModelOutput | Exception, ModelCall],
        metadata: dict[str, Any],
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        """Merge additional keys into ``ModelOutput.metadata``."""
        if isinstance(result, tuple):
            output, call = result
            if isinstance(output, ModelOutput):
                if output.metadata is None:
                    output.metadata = {}
                output.metadata.update(metadata)
            return output, call

        if result.metadata is None:
            result.metadata = {}
        result.metadata.update(metadata)
        return result

    @staticmethod
    def _strip_model_call_response(call: ModelCall) -> None:
        """Remove large vllm-lens fields from the logged API response."""
        resp = call.response
        if isinstance(resp, dict):
            resp.pop("activations", None)
            resp.pop("hook_results", None)
            resp.pop("prompt_token_ids", None)
            for choice in resp.get("choices") or []:  # type: ignore
                if isinstance(choice, dict):
                    choice.pop("token_ids", None)
