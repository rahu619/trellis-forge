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


def test_max_side_downscale(tmp_path: Path) -> None:
    path = tmp_path / "img.png"
    Image.new("RGB", (400, 200)).save(path)
    img = load_image(path, max_side=100)
    assert img.size == (100, 50)
