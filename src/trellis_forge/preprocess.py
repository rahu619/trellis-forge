from __future__ import annotations

from pathlib import Path

from PIL import Image

from .backends.base import ForgeError


def load_image(path: Path, remove_background: bool = False) -> Image.Image:
    """Load an input photo as RGB, optionally isolating the subject.

    TRELLIS.2 wants a single object, not a scene; --rembg cuts the subject out
    and composites it onto white, which is what real-world photos usually need.
    """
    try:
        img = Image.open(path)
    except Exception as exc:
        raise ForgeError(f"could not open image {path}: {exc}") from exc

    if remove_background:
        try:
            from rembg import remove
        except ImportError as exc:
            raise ForgeError(
                '--rembg needs the extra dependency: pip install "trellis-forge[rembg]"'
            ) from exc
        cutout = remove(img)  # RGBA with alpha matte
        background = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, cutout).convert("RGB")
    else:
        img = img.convert("RGB")
    return img
