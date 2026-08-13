from __future__ import annotations

from pathlib import Path

from PIL import Image

from trellis_forge.preprocess import load_image


def test_load_rgb(tmp_path: Path) -> None:
    path = tmp_path / "img.png"
    Image.new("RGBA", (32, 16), (255, 0, 0, 128)).save(path)
    img = load_image(path)
    assert img.mode == "RGB"
    assert img.size == (32, 16)
