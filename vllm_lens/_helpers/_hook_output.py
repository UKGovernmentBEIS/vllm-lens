"""Write post-hook values into plain or fused decoder-layer outputs."""

import torch


def _replace_hook_output(
    output: torch.Tensor | tuple[torch.Tensor, ...],
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None,
    hook_hidden: torch.Tensor,
    start: int,
    end: int,
    result: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Replace a request's residual stream without low-precision cancellation.

    Fused layers return an MLP update and a residual whose sum is the stream.
    Adding ``result - stream`` to just the update can leave large rounding
    errors when the residual has outliers. Store the requested stream in the
    update and zero the corresponding residual slice instead. The following
    layer consumes their sum, while other requests retain both original parts.
    Clone both parts on the first write to avoid mutating the layer's output.
    """
    if modified_output is None:
        if isinstance(output, tuple):
            modified_output = (
                output[0].clone(),
                output[1].clone() if output[1] is not None else output[1],
            )
        else:
            modified_output = output.clone()
    assert modified_output is not None
    if isinstance(modified_output, tuple):
        # A preceding steering write clones only the MLP update. Avoid
        # modifying its still-shared residual when the first post-hook runs.
        if modified_output[1] is output[1] and modified_output[1] is not None:
            modified_output = (modified_output[0], modified_output[1].clone())
        modified_output[0][start:end] = result
        if modified_output[1] is not None:
            modified_output[1][start:end] = 0
    else:
        modified_output[start:end] = result
    hook_hidden[start:end] = result
    return modified_output
