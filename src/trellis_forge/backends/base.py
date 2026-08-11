from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..config import GenerationParams


class ForgeError(RuntimeError):
    """User-facing failure: missing setup, unavailable backend, or engine error."""


@dataclass
class BackendResult:
    glb_path: Path
    meta: dict = field(default_factory=dict)


class Backend(ABC):
    """A TRELLIS.2 inference engine.

    Heavy imports (torch, trellis2, mlx, gradio_client) must stay inside methods
    so the CLI, preprocessing and tests work on machines without any of them.
    """

    name: str = "base"
    description: str = ""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Return (available, human-readable reason or status note)."""

    @abstractmethod
    def generate(
        self, image: Image.Image, out_dir: Path, params: GenerationParams
    ) -> BackendResult:
        """Generate one asset from `image`, writing at least a GLB into out_dir."""
