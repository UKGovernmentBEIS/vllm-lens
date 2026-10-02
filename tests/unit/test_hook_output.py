"""Post-hook replacements must survive fused bfloat16 residual arithmetic."""

import pytest
import torch

from vllm_lens._helpers._hook_output import _replace_hook_output


@pytest.mark.parametrize("value", [0.0, 0.25, -0.25])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_hook_replaces_stream_despite_residual_outliers(value, dtype):
    update = torch.tensor([[10.0], [3.0]], dtype=dtype)
    residual = torch.tensor([[512.0], [4.0]], dtype=dtype)
    original_update, original_residual = update.clone(), residual.clone()
    stream = update + residual
    result = torch.full((1, 1), value, dtype=dtype)

    modified = _replace_hook_output((update, residual), None, stream, 0, 1, result)

    torch.testing.assert_close((modified[0] + modified[1])[:1], result, rtol=0, atol=0)
    torch.testing.assert_close(stream[:1], result, rtol=0, atol=0)
    # A different request in the batch and the original output are untouched.
    torch.testing.assert_close(modified[0][1:], original_update[1:], rtol=0, atol=0)
    torch.testing.assert_close(modified[1][1:], original_residual[1:], rtol=0, atol=0)
    assert torch.equal(update, original_update)
    assert torch.equal(residual, original_residual)


def test_post_hooks_compose_after_steering_and_previous_hook():
    output = (
        torch.tensor([[10.0]], dtype=torch.bfloat16),
        torch.tensor([[512.0]], dtype=torch.bfloat16),
    )
    modified = (output[0] + 4, output[1])  # A preceding steering modification.
    stream = modified[0] + modified[1]
    first = torch.full_like(stream, 0.25)
    modified = _replace_hook_output(output, modified, stream, 0, 1, first)
    second = stream + 0.5
    modified = _replace_hook_output(output, modified, stream, 0, 1, second)
    torch.testing.assert_close(modified[0] + modified[1], second, rtol=0, atol=0)
    torch.testing.assert_close(stream, second, rtol=0, atol=0)
    assert output[0].item() == 10
    assert output[1].item() == 512


def test_plain_hook_replacement_preserves_other_requests_and_input():
    output = torch.tensor([[512.0], [3.0]], dtype=torch.bfloat16)
    stream = output.clone()
    result = torch.full((1, 1), 0.25, dtype=torch.bfloat16)
    modified = _replace_hook_output(output, None, stream, 0, 1, result)
    torch.testing.assert_close(modified[:1], result, rtol=0, atol=0)
    assert modified[1].item() == output[1].item() == 3
    assert output[0].item() == 512
