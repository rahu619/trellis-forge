from __future__ import annotations

from pathlib import Path


def inspect(glb_path: Path) -> dict:
    """Cheap geometry QC report. A failed inspection never sinks the run —
    the asset is still produced, the report just says why it couldn't judge."""
    try:
        import trimesh

        mesh = trimesh.load(glb_path, force="mesh")
        return {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "bounds": mesh.bounds.tolist(),
        }
    except Exception as exc:
        return {"error": str(exc)}
