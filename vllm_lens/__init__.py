from importlib.metadata import PackageNotFoundError, version

from vllm_lens._helpers._serialize import (
    decode_activations,
    deserialize_tensor,
    serialize_activations,
    serialize_tensor,
)
from vllm_lens._helpers.types import SteeringVector

try:
    __version__ = version("vllm-lens")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "decode_activations",
    "deserialize_tensor",
    "serialize_activations",
    "serialize_tensor",
    "SteeringVector",
    "__version__",
]
