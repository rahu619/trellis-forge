from __future__ import annotations

from .base import Backend, BackendResult, ForgeError
from .hf_space import HfSpaceBackend
from .mlx import MlxBackend
from .official import OfficialBackend

# Preference order for --backend auto: real GPU first, Mac port second,
# MPS third (slow), hosted demo last.
AUTO_PRIORITY = ("official-cuda", "mlx", "official-mps", "hf-space")


def all_backends() -> dict[str, Backend]:
    backends = [
        OfficialBackend("cuda"),
        MlxBackend(),
        OfficialBackend("mps"),
        HfSpaceBackend(),
    ]
    return {b.name: b for b in backends}


def resolve_backend(name: str) -> Backend:
    registry = all_backends()
    if name == "auto":
        for candidate in AUTO_PRIORITY:
            backend = registry[candidate]
            ok, _ = backend.is_available()
            if ok:
                return backend
        raise ForgeError(
            "no backend available on this machine — run `trellis-forge backends` "
            "to see what each one needs"
        )
    if name not in registry:
        raise ForgeError(f"unknown backend {name!r}; options: auto, {', '.join(registry)}")
    backend = registry[name]
    ok, reason = backend.is_available()
    if not ok:
        raise ForgeError(f"backend {name!r} is not available: {reason}")
    return backend


__all__ = [
    "AUTO_PRIORITY",
    "Backend",
    "BackendResult",
    "ForgeError",
    "HfSpaceBackend",
    "MlxBackend",
    "OfficialBackend",
    "all_backends",
    "resolve_backend",
]
