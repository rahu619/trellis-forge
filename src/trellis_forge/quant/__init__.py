"""Quantization library: make TRELLIS.2 fit Apple Silicon Macs with <= 16 GB.

The ``quantize-mlx`` CLI command is a thin wrapper over this package; the
library is importable on its own so other tooling (including, eventually, the
port's serving hook) can build on it.

Importing this package never requires MLX or the trellis2-mlx port — every
heavy import happens inside a function.
"""

from .markers import DEFAULT_WEIGHTS_REL, QUANT_MARKER, WEIGHTS_ENV, read_marker
from .mlx_weights import (
    Component,
    ComponentResult,
    QuantSummary,
    discover_components,
    find_quant_marker,
    human_bytes,
    quantize_weights,
    resolve_repo,
    resolve_weights_dir,
)

__all__ = [
    "DEFAULT_WEIGHTS_REL",
    "QUANT_MARKER",
    "WEIGHTS_ENV",
    "Component",
    "ComponentResult",
    "QuantSummary",
    "discover_components",
    "find_quant_marker",
    "human_bytes",
    "quantize_weights",
    "read_marker",
    "resolve_repo",
    "resolve_weights_dir",
]
