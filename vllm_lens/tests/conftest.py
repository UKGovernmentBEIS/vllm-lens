import gc
import multiprocessing
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

import pytest
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER_IDX = 2
NUM_LAYERS = 24
PROMPT = "The future of AI is"
PROMPTS = [
    "The future of AI is",
    "Once upon a time in a land far away",
    "The quick brown fox jumps over",
    "In the beginning there was nothing but",
    "Scientists recently discovered that the universe",
    "The best way to learn programming is",
    "Deep learning models have revolutionized",
    "A long time ago in a galaxy",
    "The weather forecast for tomorrow shows",
    "Robots and humans will eventually",
]


@pytest.fixture(scope="module")
def hf_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(  # type: ignore[reportCallIssue]
        MODEL_NAME, dtype="auto", device_map="cuda"
    ).eval()
    yield model, tokenizer
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True, scope="module")
def _cleanup_vllm():
    yield
    # Terminate leftover vLLM EngineCore subprocesses that outlive engine.shutdown().
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout=5)
        if child.is_alive():
            child.kill()
    gc.collect()
    torch.cuda.empty_cache()
    if ray.is_initialized():
        ray.shutdown()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run cross-API tests last.

    The cross-API module creates and destroys multiple vLLM engines whose
    EngineCore subprocesses may not fully terminate, holding CUDA resources.
    Running it last avoids blocking subsequent test modules; ``os._exit()``
    in ``pytest_unconfigure`` handles the final cleanup.
    """
    last: list[pytest.Item] = []
    rest: list[pytest.Item] = []
    for item in items:
        if "test_activations_cross_api" in str(item.fspath):
            last.append(item)
        else:
            rest.append(item)
    items[:] = rest + last


_exit_status = 0


def pytest_sessionfinish(session, exitstatus):
    global _exit_status
    _exit_status = exitstatus


def pytest_unconfigure(config):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_status)
