from __future__ import annotations

from dataclasses import dataclass

VALID_FORMATS = ("glb", "obj", "stl")
VALID_RESOLUTIONS = (512, 1024, 1536)


@dataclass(frozen=True)
class GenerationParams:
    """Knobs for a generation run. The seed is combined with each image's hash
    so a batch is reproducible without every asset sharing one literal seed."""

    resolution: int = 1024
    seed: int = 42
    formats: tuple[str, ...] = ("glb",)
    decimate_to: int | None = None
    max_side: int | None = None

    def validate(self) -> None:
        if self.resolution not in VALID_RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {VALID_RESOLUTIONS}, got {self.resolution}"
            )
        unknown = [f for f in self.formats if f not in VALID_FORMATS]
        if unknown:
            raise ValueError(f"unknown output formats {unknown}; valid: {VALID_FORMATS}")
        if self.decimate_to is not None and self.decimate_to < 100:
            raise ValueError("decimate_to below 100 faces rarely survives contact with a viewer")
