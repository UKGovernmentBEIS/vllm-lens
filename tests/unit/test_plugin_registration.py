"""Registration diagnostics without importing vLLM or initializing CUDA."""

import sys
from types import ModuleType

import pytest

from vllm_lens import _activations_plugin as plugin


@pytest.fixture
def serving_surface(monkeypatch):
    """Supply the upstream API surface; remove individual methods to model drift."""

    def original(*args, **kwargs):
        pass

    surfaces = {
        "vllm": {"LLM": type("LLM", (), {"generate": original, "chat": original})},
        "vllm.engine.arg_utils": {
            "EngineArgs": type("EngineArgs", (), {"create_engine_config": original})
        },
        "vllm.v1.engine.async_llm": {
            "AsyncLLM": type("AsyncLLM", (), {"generate": original})
        },
        "vllm.entrypoints.openai.completion.serving": {
            "OpenAIServingCompletion": type(
                "Completion", (), {"request_output_to_completion_response": original}
            )
        },
        "vllm.entrypoints.openai.chat_completion.serving": {
            "OpenAIServingChat": type(
                "Chat",
                (),
                {
                    "chat_completion_full_generator": original,
                    "chat_completion_stream_generator": original,
                },
            )
        },
        "vllm.entrypoints.serve": {"register_vllm_serve_api_routers": original},
    }
    modules = {}
    for name, attributes in surfaces.items():
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            path = ".".join(parts[:end])
            if path not in modules:
                modules[path] = ModuleType(path)
                monkeypatch.setitem(sys.modules, path, modules[path])
                if end > 1:
                    setattr(
                        modules[".".join(parts[: end - 1])],
                        parts[end - 1],
                        modules[path],
                    )
        for key, value in attributes.items():
            setattr(modules[name], key, value)
    for name in vars(plugin):
        if name.startswith("_original_"):
            monkeypatch.setattr(plugin, name, None)
    monkeypatch.delenv("VLLM_LENS_DISABLE", raising=False)
    monkeypatch.delenv("VLLM_LENS_STRICT_COMPATIBILITY", raising=False)
    monkeypatch.setattr(plugin, "version", lambda _: "test-version")
    return modules


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "module,owner,method,component",
    [
        (
            "completion.serving",
            "OpenAIServingCompletion",
            "request_output_to_completion_response",
            "completion response patch",
        ),
        (
            "chat_completion.serving",
            "OpenAIServingChat",
            "chat_completion_full_generator",
            "chat response patches",
        ),
        (
            "chat_completion.serving",
            "OpenAIServingChat",
            "chat_completion_stream_generator",
            "chat response patches",
        ),
        (
            "routes",
            None,
            "register_vllm_serve_api_routers",
            "hook and activation HTTP routes",
        ),
    ],
)
def test_missing_integration_is_visible(
    serving_surface, monkeypatch, caplog, strict, module, owner, method, component
):
    path = (
        "vllm.entrypoints.serve"
        if module == "routes"
        else f"vllm.entrypoints.openai.{module}"
    )
    target = serving_surface[path]
    if owner:
        target = getattr(target, owner)
    delattr(target, method)
    if strict:
        monkeypatch.setenv("VLLM_LENS_STRICT_COMPATIBILITY", "1")
        with pytest.raises(RuntimeError, match=component) as caught:
            plugin.register()
        assert isinstance(caught.value.__cause__, AttributeError)
        message = str(caught.value)
    else:
        plugin.register()
        message = caplog.text
    assert component in message
    assert "test-version" in message
    assert method in message
    assert "VLLM_LENS_STRICT_COMPATIBILITY=1" in message


def test_registration_installs_all_http_integrations(serving_surface, monkeypatch):
    monkeypatch.setenv("VLLM_LENS_STRICT_COMPATIBILITY", "1")
    plugin.register()
    completion = serving_surface[
        "vllm.entrypoints.openai.completion.serving"
    ].OpenAIServingCompletion
    chat = serving_surface[
        "vllm.entrypoints.openai.chat_completion.serving"
    ].OpenAIServingChat
    assert (
        completion.request_output_to_completion_response
        is plugin._patched_completion_response
    )
    assert chat.chat_completion_full_generator is plugin._patched_chat_full_generator
    assert (
        chat.chat_completion_stream_generator is plugin._patched_chat_stream_generator
    )
    assert (
        serving_surface["vllm.entrypoints.serve"].register_vllm_serve_api_routers
        is plugin._patched_register_routers
    )


def test_disable_still_bypasses_strict_registration(serving_surface, monkeypatch):
    monkeypatch.setenv("VLLM_LENS_DISABLE", "1")
    monkeypatch.setenv("VLLM_LENS_STRICT_COMPATIBILITY", "1")
    original = serving_surface["vllm"].LLM.generate
    plugin.register()
    assert serving_surface["vllm"].LLM.generate is original
