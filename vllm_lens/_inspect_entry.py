from inspect_ai.model import modelapi


@modelapi(name="vllm-lens")
def vllm_lens():
    from .inspect_provider import VLLMLensAPI

    return VLLMLensAPI
