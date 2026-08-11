from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from ..config import GenerationParams
from .base import Backend, BackendResult, ForgeError

MODEL_ID = "microsoft/TRELLIS.2-4B"


class OfficialBackend(Backend):
    """Wraps microsoft/TRELLIS.2's own pipeline on CUDA or MPS.

    cuda: the supported path — Linux, NVIDIA >= 24 GB VRAM, official repo installed.
    mps:  experimental — needs the MPS fix from microsoft/TRELLIS.2 PR #167
          (unmerged upstream as of 2026-08; apply it to your TRELLIS.2 checkout).
          Expect roughly 20-30 min per asset at 1024^3 on M-series chips.
    """

    description = "official microsoft/TRELLIS.2 pipeline"

    def __init__(self, device: str) -> None:
        if device not in ("cuda", "mps"):
            raise ValueError(f"unsupported device {device!r}")
        self.device = device
        self.name = f"official-{device}"
        self._pipe = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import torch
        except ImportError:
            return False, "PyTorch not installed in this environment"
        if self.device == "cuda" and not torch.cuda.is_available():
            return False, "no CUDA device visible to PyTorch"
        if self.device == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                return False, "MPS not available on this machine/PyTorch build"
        try:
            import trellis2  # noqa: F401
        except ImportError:
            return False, (
                "TRELLIS.2 repo not installed in this environment — "
                "see README.md 'CUDA backend' or use the Dockerfile"
            )
        return True, f"official pipeline on {self.device}"

    def _pipeline(self):
        if self._pipe is None:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            pipe = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
            try:
                pipe = pipe.to(self.device)
            except Exception as exc:  # upstream API drift or missing MPS patch
                raise ForgeError(
                    f"could not move pipeline to {self.device}: {exc}. "
                    "For MPS, apply microsoft/TRELLIS.2 PR #167 to your checkout."
                ) from exc
            self._pipe = pipe
        return self._pipe

    def generate(
        self, image: Image.Image, out_dir: Path, params: GenerationParams
    ) -> BackendResult:
        pipe = self._pipeline()
        outputs = pipe.run(image, seed=params.seed)
        meshes = outputs.get("mesh")
        if not meshes:
            raise ForgeError("pipeline returned no mesh — try a lower resolution")
        mesh = meshes[0]

        out_dir.mkdir(parents=True, exist_ok=True)
        glb_path = out_dir / "mesh.glb"

        export_kwargs = {}
        if params.decimate_to:
            export_kwargs["decimation_target"] = params.decimate_to
        try:
            # TRELLIS.2 exports through the O-Voxel postprocessor (see upstream README).
            from o_voxel.postprocess import to_glb

            result = to_glb(mesh, **export_kwargs)
        except ImportError as exc:
            raise ForgeError(
                "o_voxel postprocessor not importable — is the TRELLIS.2 checkout built?"
            ) from exc
        except Exception as exc:
            raise ForgeError(f"mesh export failed: {exc}") from exc

        if isinstance(result, (bytes, bytearray)):
            glb_path.write_bytes(bytes(result))
        elif isinstance(result, (str, Path)):
            shutil.copyfile(result, glb_path)
        else:  # signature drift upstream: let the user see it rather than guess
            raise ForgeError(f"unexpected to_glb() return type: {type(result)!r}")
        return BackendResult(glb_path=glb_path, meta={"model": MODEL_ID, "device": self.device})
