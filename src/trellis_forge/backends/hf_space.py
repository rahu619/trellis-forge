from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from PIL import Image

from ..config import GenerationParams
from .base import Backend, BackendResult, ForgeError

SPACE_ID = "microsoft/TRELLIS.2"


class HfSpaceBackend(Backend):
    """Free hosted TRELLIS.2 demo via gradio_client.

    The no-GPU fallback: works on any machine with network access (including a
    16 GB MacBook), at the cost of queue waits and ZeroGPU quota — anonymous
    visitors get only a few minutes of GPU per day, an HF token buys more. The
    Space is a research demo and its API can change without notice; generation
    is currently a stateful three-step flow: /start_session, /image_to_3d,
    /extract_glb.
    """

    name = "hf-space"
    description = f"free hosted demo Space ({SPACE_ID})"

    def __init__(self) -> None:
        self._client = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import gradio_client  # noqa: F401
        except ImportError:
            return False, 'install with: pip install "trellis-forge[hf]"'
        return True, "remote demo Space — needs network, may queue (ZeroGPU quota)"

    def generate(
        self, image: Image.Image, out_dir: Path, params: GenerationParams
    ) -> BackendResult:
        from gradio_client import Client, handle_file

        if self._client is None:
            try:
                self._client = Client(SPACE_ID)
            except Exception as exc:
                raise ForgeError(f"could not connect to {SPACE_ID}: {exc}") from exc

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            image.save(tmp_path)
            try:
                self._client.predict(api_name="/start_session")
                self._client.predict(
                    handle_file(str(tmp_path)),
                    float(params.seed),
                    str(params.resolution),
                    api_name="/image_to_3d",
                )
                # Omitting args keeps the Space's own defaults (300k faces, 2K
                # textures); only override decimation when the user asked.
                extract_args = () if params.decimate_to is None else (float(params.decimate_to),)
                result = self._client.predict(*extract_args, api_name="/extract_glb")
            except Exception as exc:
                raise ForgeError(
                    f"Space call failed: {exc}. The demo API may have changed — inspect it "
                    f"with: python -c \"from gradio_client import Client; "
                    f"Client('{SPACE_ID}').view_api()\""
                ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        glb_src = _find_glb(result)
        if glb_src is None:
            raise ForgeError(f"no GLB found in Space response: {result!r}")
        out_dir.mkdir(parents=True, exist_ok=True)
        glb_path = out_dir / "mesh.glb"
        shutil.copyfile(glb_src, glb_path)
        return BackendResult(glb_path=glb_path, meta={"space": SPACE_ID})


def _find_glb(obj):
    """Recursively search a gradio result (str / dict / list) for a .glb path."""
    if isinstance(obj, str):
        return obj if obj.lower().endswith(".glb") else None
    if isinstance(obj, dict):
        for value in obj.values():
            if found := _find_glb(value):
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for value in obj:
            if found := _find_glb(value):
                return found
        return None
    return None
