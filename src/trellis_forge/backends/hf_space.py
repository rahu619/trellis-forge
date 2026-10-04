from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from ..config import GenerationParams
from .base import Backend, BackendResult, ForgeError

SPACE_ID = "microsoft/TRELLIS.2"

# Driven in this order, and probed up front so that API drift fails backend
# selection instead of dying partway through a batch.
ENDPOINTS = ("/preprocess_image", "/image_to_3d", "/extract_glb")

_DRIFT_HINT = (
    "the demo API may have changed — inspect it with: "
    f'python -c "from gradio_client import Client; Client(\'{SPACE_ID}\').view_api()"'
)


class HfSpaceBackend(Backend):
    """Free hosted TRELLIS.2 demo via gradio_client.

    The no-GPU fallback: works on any machine with network access (including a
    16 GB MacBook), at the cost of queue waits and ZeroGPU quota — anonymous
    visitors get only a few minutes of GPU per day, and exporting HF_TOKEN buys
    more. Generation is stateful: the Space keeps the sampled latents against
    the session between /image_to_3d and /extract_glb, so the client never
    handles them.
    """

    name = "hf-space"
    description = f"free hosted demo Space ({SPACE_ID})"

    def __init__(self) -> None:
        self._client = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import gradio_client  # noqa: F401
        except ImportError:
            return False, "gradio-client is missing — reinstall trellis-forge"
        try:
            api = self._connect().view_api(return_format="dict", print_info=False) or {}
        except Exception as exc:  # noqa: BLE001 — unreachable is just unavailable
            return False, f"could not reach {SPACE_ID}: {exc}"
        named = api.get("named_endpoints", {})
        if missing := [endpoint for endpoint in ENDPOINTS if endpoint not in named]:
            return False, f"{SPACE_ID} no longer exposes {', '.join(missing)} — {_DRIFT_HINT}"
        return True, "remote demo Space — needs network, may queue (ZeroGPU quota)"

    def _connect(self):
        """Open a client for the Space once, then reuse it for the whole batch."""
        if self._client is None:
            from gradio_client import Client

            # gradio_client does not read HF_TOKEN itself, so without this the
            # Space only ever serves the small anonymous quota.
            self._client = Client(SPACE_ID, token=os.environ.get("HF_TOKEN"), verbose=False)
        return self._client

    def generate(
        self, image: Image.Image, out_dir: Path, params: GenerationParams
    ) -> BackendResult:
        from gradio_client import handle_file

        client = self._connect()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            image.save(tmp_path)
            try:
                client.predict(api_name="/start_session")
                # The Space runs /image_to_3d with preprocess_image=False, so the
                # cutout and square crop have to be asked for separately — which
                # is exactly what its own UI does when an image is uploaded.
                # gradio_client downloads file outputs to a local path.
                cutout = client.predict(handle_file(str(tmp_path)), api_name="/preprocess_image")
                client.predict(
                    handle_file(str(cutout)),
                    float(params.seed),
                    str(params.resolution),
                    api_name="/image_to_3d",
                )
                # Omitting args keeps the Space's own defaults (300k faces, 2K
                # textures); only override decimation when the user asked.
                extract_args = () if params.decimate_to is None else (float(params.decimate_to),)
                result = client.predict(*extract_args, api_name="/extract_glb")
            except Exception as exc:
                raise ForgeError(_explain(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        glb_src = _glb_from(result)
        out_dir.mkdir(parents=True, exist_ok=True)
        glb_path = out_dir / "mesh.glb"
        shutil.copyfile(glb_src, glb_path)
        return BackendResult(glb_path=glb_path, meta={"space": SPACE_ID})


def _explain(exc: Exception) -> str:
    text = str(exc)
    if "quota" in text.lower():
        # Running out of free GPU-minutes is the common case, not API drift —
        # pointing at view_api() here would just send people down a dead end.
        return f"ZeroGPU quota exhausted; set HF_TOKEN for a larger allowance. {text}"
    return f"Space call failed: {text}. {_DRIFT_HINT}"


def _glb_from(result) -> str:
    """/extract_glb returns two filepaths: the GLB and its download button."""
    first = result[0] if isinstance(result, (list, tuple)) and result else result
    if isinstance(first, str) and first.lower().endswith(".glb"):
        return first
    raise ForgeError(f"no GLB in the Space response ({result!r}). {_DRIFT_HINT}")
