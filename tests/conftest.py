"""Fixtures for integration tests against a running vLLM server."""

import os
import subprocess
import sys
import time

import pytest
import requests

SERVER_PORT = int(os.environ.get("VLLM_TEST_PORT", "8100"))
MODEL = os.environ.get("VLLM_TEST_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
BASE_URL = f"http://localhost:{SERVER_PORT}"

# How long to wait for the server to start (seconds).
_STARTUP_TIMEOUT = int(os.environ.get("VLLM_TEST_STARTUP_TIMEOUT", "300"))


def _server_healthy(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture(scope="session")
def vllm_server():
    """Start a vLLM server for the test session, or reuse an existing one.

    If a server is already listening on ``SERVER_PORT``, it is reused
    (useful when you start the server manually in tmux). Otherwise a new
    subprocess is spawned and torn down after the session.
    """
    if _server_healthy(BASE_URL):
        # Reuse existing server — don't manage its lifecycle.
        yield BASE_URL
        return

    vllm_bin = os.path.join(os.path.dirname(sys.executable), "vllm")
    proc = subprocess.Popen(
        [
            vllm_bin, "serve", MODEL,
            "--dtype", "auto",
            "--gpu-memory-utilization", "0.9",
            "--port", str(SERVER_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    for _ in range(_STARTUP_TIMEOUT):
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(
                f"vLLM server exited early (code {proc.returncode}):\n{stdout[-2000:]}"
            )
        if _server_healthy(BASE_URL):
            break
        time.sleep(1)
    else:
        proc.kill()
        raise RuntimeError(
            f"vLLM server did not become healthy within {_STARTUP_TIMEOUT}s"
        )

    yield BASE_URL

    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
