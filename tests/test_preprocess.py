from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from trellis_forge.backends.base import ForgeError
from trellis_forge.preprocess import load_image


def test_rgb_input_stays_rgb(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (32, 16), (200, 180, 160)).save(path)
    img = load_image(path)
    assert img.mode == "RGB"
    assert img.size == (32, 16)


def test_alpha_survives_for_the_backend_to_use(tmp_path: Path) -> None:
    # Both backends run their own BiRefNet/RMBG-2.0 cutout and honour an alpha
    # channel that is already there, so a pre-cut PNG must not be flattened.
    path = tmp_path / "cutout.png"
    Image.new("RGBA", (32, 16), (255, 0, 0, 128)).save(path)
    assert load_image(path).mode == "RGBA"


def test_indexed_transparency_becomes_rgba(tmp_path: Path) -> None:
    path = tmp_path / "indexed.png"
    img = Image.new("P", (8, 8))
    img.info["transparency"] = 0
    img.save(path)
    assert load_image(path).mode == "RGBA"


def test_unreadable_file_is_a_forge_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not a png")
    with pytest.raises(ForgeError):
        load_image(path)
