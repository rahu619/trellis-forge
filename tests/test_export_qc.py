from __future__ import annotations

from pathlib import Path

import trimesh

from trellis_forge.export import export_formats
from trellis_forge.qc import inspect


def _box_glb(tmp_path: Path) -> Path:
    glb = tmp_path / "mesh.glb"
    trimesh.creation.box().export(glb)
    return glb


def test_qc_reports_watertight_box(tmp_path: Path) -> None:
    report = inspect(_box_glb(tmp_path))
    assert report["watertight"] is True
    assert report["faces"] == 12
    assert report["vertices"] == 8
    assert "error" not in report


def test_qc_never_raises(tmp_path: Path) -> None:
    report = inspect(tmp_path / "missing.glb")
    assert "error" in report


def test_glb_passthrough_only(tmp_path: Path) -> None:
    glb = _box_glb(tmp_path)
    outputs = export_formats(glb, tmp_path, ("glb",))
    assert outputs == {"glb": glb}


def test_obj_and_stl_conversion(tmp_path: Path) -> None:
    out_dir = tmp_path / "asset"
    out_dir.mkdir()
    glb = _box_glb(out_dir)
    outputs = export_formats(glb, out_dir, ("glb", "obj", "stl"))

    assert set(outputs) == {"glb", "obj", "stl"}
    assert outputs["obj"].exists() and outputs["obj"].stat().st_size > 0
    assert outputs["stl"].exists() and outputs["stl"].stat().st_size > 0
    reloaded = trimesh.load(outputs["stl"])
    assert len(reloaded.faces) == 12
