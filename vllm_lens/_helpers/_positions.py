"""Position selection for residual-stream capture (issue #40).

By default the worker captures every processed token position. The optional
``output_residual_stream_positions`` entry in ``extra_args`` narrows this to
the last prompt token or to explicit absolute positions. The helpers here hold
the request-side validation, the per-forward row selection and the ordering
rules; they import no vLLM so the plugin, the client and the CPU unit tests can
share them.

Positions index the tokenized model input. Prompt tokens occupy
``0 .. n_prompt - 1`` and ``n_prompt + k`` is the state computed when generated
token ``k`` is fed back for the next step. The final generated token is never
processed, so it has no row.
"""

from __future__ import annotations

import json
from typing import Any

import torch

POSITIONS_KEY = "output_residual_stream_positions"
"""``extra_args`` / ``vllm_xargs`` key carrying the selector."""

LAST_PROMPT_TOKEN = "last_prompt_token"
"""Selector value for the final token of the full prompt."""

POSITIONS_META_KEY = "residual_stream_positions"
"""Activations entry listing the absolute position of each captured row."""


def parse_capture_positions(value: Any) -> str | list[int] | None:
    """Normalise a selector value from any entry point.

    Accepts ``None`` (capture every position), the literal
    ``"last_prompt_token"``, a list of non-negative absolute positions, or the
    JSON-encoded string form of either (how ``vllm_xargs`` delivers values).
    Explicit positions come back sorted with duplicates removed. Anything else
    raises ``ValueError`` so a bad request fails before it is scheduled.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value == LAST_PROMPT_TOKEN:
            return LAST_PROMPT_TOKEN
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            raise ValueError(
                f"{POSITIONS_KEY} must be {LAST_PROMPT_TOKEN!r} or a list of "
                f"absolute token positions, got {value!r}"
            ) from None
        if decoded is None:
            return None
        if decoded == LAST_PROMPT_TOKEN:
            return LAST_PROMPT_TOKEN
        value = decoded
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{POSITIONS_KEY} must be {LAST_PROMPT_TOKEN!r} or a list of "
            f"absolute token positions, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{POSITIONS_KEY} must name at least one position")
    positions: set[int] = set()
    for position in value:
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(
                f"{POSITIONS_KEY} entries must be integers, got {position!r}"
            )
        if position < 0:
            raise ValueError(
                f"{POSITIONS_KEY} entries must be non-negative absolute "
                f"positions, got {position}"
            )
        positions.add(position)
    return sorted(positions)


def normalize_capture_positions(extra_args: dict[str, Any] | None) -> None:
    """Validate the selector at a request entry point and store its normal form.

    Rewrites ``extra_args[POSITIONS_KEY]`` in place so the workers receive the
    parsed value. A selector without ``output_residual_stream`` is rejected,
    since there would be nothing to select from.
    """
    if not extra_args or POSITIONS_KEY not in extra_args:
        return
    selector = parse_capture_positions(extra_args[POSITIONS_KEY])
    if selector is None:
        del extra_args[POSITIONS_KEY]
        return
    if extra_args.get("output_residual_stream") is None:
        raise ValueError(f"{POSITIONS_KEY} requires output_residual_stream")
    extra_args[POSITIONS_KEY] = selector


def positions_in_slice(
    selector: str | list[int], abs_start: int, n_tokens: int, n_prompt: int
) -> list[int]:
    """Absolute positions the selector wants from one request's slice of a forward.

    The slice covers ``[abs_start, abs_start + n_tokens)``: one prefill chunk
    or one decode token. ``"last_prompt_token"`` resolves to ``n_prompt - 1``,
    the end of the full prompt, which only falls inside the final prefill
    chunk. Explicit positions are already sorted and unique.
    """
    abs_end = abs_start + n_tokens
    if selector == LAST_PROMPT_TOKEN:
        wanted: list[int] = [n_prompt - 1]
    else:
        wanted = selector  # type: ignore[assignment]
    return [p for p in wanted if abs_start <= p < abs_end]


def select_rows(
    hidden_states: torch.Tensor, start: int, abs_start: int, positions: list[int]
) -> torch.Tensor:
    """Rows of ``hidden_states`` at the given absolute positions, still on device.

    ``start`` is the batch row holding absolute position ``abs_start``. A
    contiguous run is returned as a view so the following ``.cpu()`` copies only
    the selected rows; an empty selection is a ``(0, hidden_dim)`` view.
    """
    if not positions:
        return hidden_states[start:start]
    first = start + positions[0] - abs_start
    if positions[-1] - positions[0] == len(positions) - 1:
        return hidden_states[first : first + len(positions)]
    index = torch.tensor(
        [start + p - abs_start for p in positions],
        dtype=torch.long,
        device=hidden_states.device,
    )
    return hidden_states.index_select(0, index)


def order_positions(positions: list[int]) -> tuple[list[int], list[int]]:
    """Ascending unique positions and, for each, the captured row to keep.

    Rows arrive in computation order, which is ascending except when vLLM
    preempts a request and recomputes it from the start. A position computed
    twice keeps its latest row, so the returned row indices select one row per
    position in ascending position order.
    """
    latest: dict[int, int] = {}
    for row, position in enumerate(positions):
        latest[position] = row
    ordered = sorted(latest)
    return ordered, [latest[p] for p in ordered]
