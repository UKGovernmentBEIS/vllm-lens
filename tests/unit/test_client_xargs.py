"""``vllm_xargs`` the HTTP client sends for each capture option (no server)."""

from typing import Any

import pytest

from vllm_lens.client import VLLMLensClient


class _FakeResponse:
    def json(self) -> dict[str, Any]:
        return {"choices": [{"text": "ok", "message": {"content": "ok"}}]}


@pytest.fixture
def client_and_bodies(monkeypatch):
    client = VLLMLensClient("http://server", model="m")
    bodies: list[dict[str, Any]] = []

    def fake_post(url, json, timeout):
        bodies.append(json)
        return _FakeResponse()

    monkeypatch.setattr(client._session, "post", fake_post)
    return client, bodies


def _call(client: VLLMLensClient, method: str, **kwargs: Any) -> None:
    if method == "generate":
        client.generate("hi", **kwargs)
    else:
        client.chat([{"role": "user", "content": "hi"}], **kwargs)


@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("capture_qk", [None, False])
def test_capture_qk_off_does_not_request_qk(client_and_bodies, method, capture_qk):
    client, bodies = client_and_bodies
    _call(client, method, capture_qk=capture_qk)
    assert "vllm_xargs" not in bodies[-1]


@pytest.mark.parametrize("method", ["generate", "chat"])
def test_capture_qk_false_keeps_other_xargs(client_and_bodies, method):
    client, bodies = client_and_bodies
    _call(client, method, capture_layers=[2], capture_qk=False)
    assert bodies[-1]["vllm_xargs"] == {"output_residual_stream": "[2]"}


@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize(
    ("capture_qk", "expected"), [(True, "true"), ([3], "[3]"), ([0, 5], "[0, 5]")]
)
def test_capture_qk_on_requests_qk(client_and_bodies, method, capture_qk, expected):
    client, bodies = client_and_bodies
    _call(client, method, capture_qk=capture_qk)
    assert bodies[-1]["vllm_xargs"] == {"output_qk": expected}
