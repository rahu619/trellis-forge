from __future__ import annotations

from pathlib import Path


def export_formats(glb_path: Path, out_dir: Path, formats: tuple[str, ...]) -> dict[str, Path]:
    """Materialise requested formats next to the backend's GLB.

    The GLB stays the primary, texture-carrying artifact. OBJ/STL are
    geometry-only convenience exports (e.g. STL for 3D printing) — conversion
    drops the PBR textures by construction.
    """
    outputs: dict[str, Path] = {}
    if "glb" in formats:
        outputs["glb"] = glb_path

    extras = [f for f in formats if f != "glb"]
    if not extras:
        return outputs

    import trimesh

    mesh = trimesh.load(glb_path, force="mesh")
    for fmt in extras:
        dest = out_dir / f"mesh.{fmt}"
        mesh.export(dest, file_type=fmt)
        outputs[fmt] = dest
    return outputs
