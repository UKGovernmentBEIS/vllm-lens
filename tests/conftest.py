"""Fixtures for integration tests against a running vLLM server."""

import os
import subprocess
import sys
import time

import pytest
import requests

SERVER_PORT = int(os.environ.get("VLLM_TEST_PORT", "8100"))
MODEL = os.environ.get("VLLM_TEST_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
TP_SIZE = int(os.environ.get("VLLM_TEST_TP_SIZE", "1"))
PP_SIZE = int(os.environ.get("VLLM_TEST_PP_SIZE", "1"))
BASE_URL = f"http://localhost:{SERVER_PORT}"

# How long to wait for the server to start (seconds).  Large models
# (weight download + load + CUDA-graph capture) can take well over five
# minutes on a cold cache, so default generously and allow overriding.
_STARTUP_TIMEOUT = int(os.environ.get("VLLM_TEST_STARTUP_TIMEOUT", "900"))


def _server_healthy(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture(scope="session")
def vllm_server(tmp_path_factory):
    """Start a vLLM server for the test session, or reuse an existing one.

    If a server is already listening on ``SERVER_PORT``, it is reused
    (useful when you start the server manually in tmux). Otherwise a new
    subprocess is spawned and torn down after the session.
    """
    if _server_healthy(BASE_URL):
        if os.environ.get("VLLM_TEST_REUSE_SERVER") == "0":
            raise RuntimeError(
                f"Compatibility tests require a fresh server; {BASE_URL} is already in use"
            )
        # Reuse existing server — don't manage its lifecycle.
        yield BASE_URL
        return

    vllm_bin = os.path.join(os.path.dirname(sys.executable), "vllm")
    command = [
        vllm_bin,
        "serve",
        MODEL,
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        "0.9",
        "--port",
        str(SERVER_PORT),
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--pipeline-parallel-size",
        str(PP_SIZE),
    ]
    if max_len := os.environ.get("VLLM_TEST_MAX_MODEL_LEN"):
        command.extend(["--max-model-len", max_len])
    # A PIPE that nobody drains can fill during model startup and deadlock.
    log_path = tmp_path_factory.mktemp("vllm-server") / "server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(_STARTUP_TIMEOUT):
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"vLLM server exited early (code {proc.returncode}):\n"
                        f"{log_path.read_text()[-8000:]}"
                    )
                if _server_healthy(BASE_URL):
                    break
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"vLLM server did not become healthy within {_STARTUP_TIMEOUT}s:\n"
                    f"{log_path.read_text()[-8000:]}"
                )
            yield BASE_URL
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
