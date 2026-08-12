"""The quantized-weights marker: precision metadata next to the checkpoints.

``quantized.json`` records what was quantized, at what precision, and when.
Status displays and backends probe for it, so a machine that has gone through
quantization gets different (better) guidance than one that has not.

Stdlib only and no internal imports: this module must stay importable from
anywhere, including lazily from backend probes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

QUANT_MARKER = "quantized.json"

# The same env var / default path trellis2-mlx's api_server.py uses for its
# weights directory — quantized weights must land where the server looks.
WEIGHTS_ENV = "TRELLIS2_WEIGHTS"
DEFAULT_WEIGHTS_REL = Path("weights") / "TRELLIS.2-4B"


def read_marker(weights_dir: Path) -> dict | None:
    """Parse the marker, or None when absent/corrupt — probing never crashes."""
    try:
        return json.loads((weights_dir / QUANT_MARKER).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_marker(
    weights_dir: Path,
    *,
    bits: int,
    group_size: int,
    components: dict[str, str],
    tool_version: str,
) -> Path:
    """Write the marker, merging over any earlier one.

    A second run (new components, or ``--force`` on a subset) must not lose
    the record of components quantized previously.
    """
    merged: dict[str, str] = {}
    previous = read_marker(weights_dir)
    if previous and isinstance(previous.get("components"), dict):
        merged.update(previous["components"])
    merged.update(components)
    marker = {
        "format": "trellis-forge-quantized",
        "version": 1,
        "bits": bits,
        "group_size": group_size,
        "components": merged,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "forge_version": tool_version,
    }
    path = weights_dir / QUANT_MARKER
    path.write_text(json.dumps(marker, indent=2) + "\n")
    return path
