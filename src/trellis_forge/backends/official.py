from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from ..config import GenerationParams
from .base import Backend, BackendResult, ForgeError

# Read once when CUDA initialises, so it has to be set at import time — the same
# thing upstream's app.py and example.py do. Meaningfully cuts peak VRAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "microsoft/TRELLIS.2-4B"

# Voxel resolutions -> upstream pipeline_type identifiers
# (trellis2/pipelines/trellis2_image_to_3d.py). 1024/1536 run cascade sampling.
PIPELINE_TYPES = {512: "512", 1024: "1024_cascade", 1536: "1536_cascade"}

# The pipeline normalizes the subject into a unit cube centered on the origin.
_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


class OfficialBackend(Backend):
    """Wraps microsoft/TRELLIS.2's own pipeline on CUDA.

    The supported local path: Linux, NVIDIA >= 24 GB VRAM, upstream repo
    installed (or use the Dockerfile). `--resolution` maps to upstream's
    pipeline_type, and export is wired exactly like upstream's app.py.
    """

    name = "official-cuda"
    description = "official microsoft/TRELLIS.2 pipeline (CUDA)"

    def __init__(self) -> None:
        self._pipe = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import torch
        except ImportError:
            return False, "PyTorch not installed in this environment"
        if not torch.cuda.is_available():
            return False, "no CUDA device visible to PyTorch"
        try:
            import trellis2  # noqa: F401
        except ImportError:
            return False, (
                "TRELLIS.2 repo not installed in this environment — "
                "see README.md or use the Dockerfile"
            )
        return True, "official pipeline on CUDA"

    def _pipeline(self):
        if self._pipe is None:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            pipe = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
            try:
                # Returns None upstream, so this must stay a bare statement —
                # rebinding would throw the loaded pipeline away.
                pipe.cuda()
            except Exception as exc:
                raise ForgeError(f"could not move pipeline to CUDA: {exc}") from exc
            self._pipe = pipe
        return self._pipe

    def generate(
        self, image: Image.Image, out_dir: Path, params: GenerationParams
    ) -> BackendResult:
        import torch

        pipe = self._pipeline()
        try:
            # run() defaults to preprocess_image=True, so the pipeline applies its
            # own BiRefNet/RMBG-2.0 cutout, downscale and square crop. Upstream's
            # app.py passes False only because its UI preprocesses on upload.
            meshes = pipe.run(
                image,
                seed=params.seed,
                pipeline_type=PIPELINE_TYPES[params.resolution],
            )
            if not meshes:
                raise ForgeError("pipeline returned no mesh — try a lower resolution")
            glb = _to_glb(meshes[0], params)

            out_dir.mkdir(parents=True, exist_ok=True)
            glb_path = out_dir / "mesh.glb"
            glb.export(str(glb_path), extension_webp=True)
            return BackendResult(glb_path=glb_path, meta={"model": MODEL_ID})
        finally:
            # A batch holds the GPU for minutes; release cached blocks between
            # images so fragmentation can't OOM a long run.
            torch.cuda.empty_cache()


def _to_glb(mesh, params: GenerationParams):
    """Same wiring as extract_glb() in upstream's app.py, except that layout and
    voxel size are read off the mesh: a cascade run can decode at a resolution
    other than the one that was requested."""
    try:
        from o_voxel.postprocess import to_glb
    except ImportError as exc:
        raise ForgeError(
            "o_voxel postprocessor not importable — is the TRELLIS.2 checkout built "
            "(see its setup.sh, or use the Dockerfile)?"
        ) from exc

    export_kwargs = {}
    if params.decimate_to:
        export_kwargs["decimation_target"] = params.decimate_to
    try:
        return to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=_AABB,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            **export_kwargs,
        )
    except Exception as exc:
        raise ForgeError(f"mesh export failed: {exc}") from exc
