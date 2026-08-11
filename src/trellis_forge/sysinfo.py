"""Machine introspection used to set expectations before a run.

TRELLIS.2's feasibility is dominated by memory (a 4B-parameter model), and on
Apple Silicon that memory is unified and shared with the OS. Rather than let a
16 GB MacBook walk into an OOM, we surface the machine profile up front and let
backends tailor their guidance to it.

Stdlib only, and every probe degrades to `None`/`False` instead of raising —
detection must never sink a command.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

# At or below this many GB, full-precision local generation is best-effort and
# the honest recommendation is 512^3 or the hosted backend.
LOW_MEMORY_GB = 24


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001 — probing is best-effort by design
        return None


def total_memory_gb() -> float | None:
    """Total physical memory in GB, or None if it can't be determined."""
    try:
        if sys.platform == "darwin":
            raw = _sysctl("hw.memsize")
            return int(raw) / (1024**3) if raw else None
        if hasattr(os, "sysconf"):
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except Exception:  # noqa: BLE001
        return None
    return None


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def chip_name() -> str | None:
    if sys.platform == "darwin":
        return _sysctl("machdep.cpu.brand_string")
    return platform.processor() or None


def is_low_memory() -> bool:
    """True when local, full-precision generation is likely to be tight."""
    mem = total_memory_gb()
    return mem is not None and mem <= LOW_MEMORY_GB


def machine_summary() -> str:
    """A one-line human description of the machine, for the `backends` table."""
    mem = total_memory_gb()
    if is_apple_silicon():
        chip = chip_name() or "Apple Silicon"
        mem_txt = f"{mem:.0f} GB unified memory" if mem else "unified memory"
        return f"macOS · {chip} · {mem_txt}"
    system = platform.system()
    if mem is not None:
        return f"{system} · {mem:.0f} GB memory"
    return system
