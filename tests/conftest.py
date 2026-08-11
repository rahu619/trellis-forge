from __future__ import annotations

from pathlib import Path

import pytest
import trimesh
from PIL import Image

from trellis_forge.backends.base import Backend, BackendResult, ForgeError


class FakeBackend(Backend):
    """Writes a small watertight box GLB — enough to exercise everything
    downstream of inference (export, QC, manifest, resume) on any machine."""

    name = "fake"
    description = "test double"

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[Path] = []

    def is_available(self) -> tuple[bool, str]:
        return True, "always"

    def generate(self, image, out_dir: Path, params) -> BackendResult:
        self.calls.append(out_dir)
        if self.fail_on and self.fail_on in str(out_dir):
            raise ForgeError("boom")
        out_dir.mkdir(parents=True, exist_ok=True)
        glb_path = out_dir / "mesh.glb"
        trimesh.creation.box().export(glb_path)
        return BackendResult(glb_path=glb_path, meta={"fake": True})


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "photos"
    folder.mkdir()
    # Distinct pixel content per file: the manifest dedupes by content hash,
    # so identical images would (correctly) count as one asset.
    colors = {"molar-01.png": (200, 180, 160), "molar-02.png": (120, 90, 80)}
    for name, color in colors.items():
        Image.new("RGB", (8, 8), color).save(folder / name)
    Image.new("RGB", (8, 8), (60, 70, 80)).save(folder / "premolar.jpg")
    (folder / "notes.txt").write_text("not an image")
    return folder
