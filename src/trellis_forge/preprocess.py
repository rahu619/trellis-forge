from __future__ import annotations

from pathlib import Path

from PIL import Image

from .backends.base import ForgeError


def load_image(path: Path) -> Image.Image:
    """Load an input photo as RGB/RGBA, keeping any alpha channel intact.

    Background removal is the backend's job, and it does it better than we can
    locally: the official pipeline runs BiRefNet/RMBG-2.0 plus a square
    foreground crop, and the demo Space exposes the same step as an endpoint.
    Both honour an alpha channel that is already there, so a pre-cut PNG has to
    stay RGBA rather than being flattened onto a background colour.
    """
    try:
        with Image.open(path) as img:
            has_alpha = "A" in img.getbands() or "transparency" in img.info
            return img.convert("RGBA" if has_alpha else "RGB")
    except Exception as exc:
        raise ForgeError(f"could not open image {path}: {exc}") from exc
