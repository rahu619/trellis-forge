from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from ..config import GenerationParams
from .base import Backend, BackendResult, ForgeError

DEFAULT_REPO_ENV = "TRELLIS2_MLX_REPO"
DEFAULT_URL_ENV = "TRELLIS2_MLX_URL"
DEFAULT_URL = "http://127.0.0.1:8000"
UPSTREAM_REPO = "https://github.com/gtrg55/trellis2-mlx"

# /health is a liveness probe; /generate can take minutes (longer on 16 GB), so
# give it a generous budget rather than killing a slow-but-working generation.
HEALTH_TIMEOUT_S = 5
GENERATE_TIMEOUT_S = 3600

# TRELLIS.2 resolutions -> the port's pipeline_type identifiers (see its
# api_models.py). 1024 maps to the cascade variant, which is what 16 GB Macs
# should pair with --resolution 512 to stay in memory.
_PIPELINE_TYPES = {512: "512", 1024: "1024_cascade", 1536: "1536_cascade"}


def pipeline_type_for(resolution: int) -> str:
    try:
        return _PIPELINE_TYPES[resolution]
    except KeyError:
        raise ForgeError(
            f"MLX port supports resolutions {sorted(_PIPELINE_TYPES)}, got {resolution}"
        ) from None


def _build_payload(image: Image.Image, params: GenerationParams) -> dict:
    """Encode an image into the trellis2-mlx GenerateRequest JSON body."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload: dict = {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "seed": params.seed,
        "pipeline_type": pipeline_type_for(params.resolution),
    }
    if params.decimate_to:
        payload["decimation_target"] = params.decimate_to
    return payload


def _parse_glb(response: dict) -> bytes:
    glb_b64 = response.get("glb")
    if not glb_b64:
        raise ForgeError(f"MLX server returned no GLB: {response}")
    return base64.b64decode(glb_b64)


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class MlxBackend(Backend):
    """Apple Silicon backend wrapping the community MLX port (gtrg55/trellis2-mlx).

    The port ships a FastAPI server (``api_server.py``) with a ``POST /generate``
    endpoint, so this adapter talks to it over HTTP instead of importing it — the
    server can live in its own environment with MLX and the 15 GB of weights.

    Status: experimental. Validated upstream only on 128 GB unified memory; on a
    16 GB Mac use ``--resolution 512`` and treat success as best-effort until
    quantized MLX weights exist. Start the server once, then point this backend
    at it (``TRELLIS2_MLX_URL`` if it isn't on 127.0.0.1:8000).
    """

    name = "mlx"
    description = "Apple Silicon MLX port (experimental, HTTP)"

    def __init__(self, repo: str | Path | None = None, url: str | None = None) -> None:
        self.repo = Path(repo) if repo else None
        self._url_override = url

    def _repo(self) -> Path | None:
        if self.repo is not None:
            return self.repo
        env = os.environ.get(DEFAULT_REPO_ENV)
        return Path(env) if env else None

    def _url(self) -> str:
        if self._url_override:
            return self._url_override.rstrip("/")
        return os.environ.get(DEFAULT_URL_ENV, DEFAULT_URL).rstrip("/")

    def _health(self) -> dict | None:
        try:
            return _get_json(f"{self._url()}/health", timeout=HEALTH_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — unreachable server is a valid state
            return None

    def is_available(self) -> tuple[bool, str]:
        url = self._url()
        health = self._health()
        if health is not None:
            if not health.get("weights_loaded", False):
                return False, f"server at {url} is up but its weights are not loaded"
            return True, f"connected to API server at {url}"

        repo = self._repo()
        if repo is None:
            return False, (
                f"no server at {url}; clone {UPSTREAM_REPO}, set {DEFAULT_REPO_ENV} "
                "to its path, then start it with `python api_server.py`"
            )
        if not repo.is_dir():
            return False, f"{DEFAULT_REPO_ENV}={repo} does not exist"
        if not (repo / "api_server.py").exists():
            return False, (
                f"{repo} has no api_server.py — `git pull` your clone; the HTTP "
                "API is required"
            )
        return False, (
            f"repo ready at {repo}; start the server with "
            f"`cd {repo} && python api_server.py`, then re-run"
        )

    def generate(
        self, image: Image.Image, out_dir: Path, params: GenerationParams
    ) -> BackendResult:
        url = self._url()
        health = self._health()
        if health is None:
            raise ForgeError(
                f"no MLX server reachable at {url}. Start it first: clone "
                f"{UPSTREAM_REPO}, then run `python api_server.py` inside it "
                f"(set {DEFAULT_URL_ENV} if it listens elsewhere)."
            )
        if not health.get("weights_loaded", False):
            raise ForgeError(f"MLX server at {url} is up but its weights are not loaded")

        payload = _build_payload(image, params)
        try:
            response = _post_json(f"{url}/generate", payload, timeout=GENERATE_TIMEOUT_S)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ForgeError(f"MLX server call failed: {exc}") from exc

        glb_bytes = _parse_glb(response)
        out_dir.mkdir(parents=True, exist_ok=True)
        glb_path = out_dir / "mesh.glb"
        glb_path.write_bytes(glb_bytes)

        meta: dict = {"pipeline_type": payload["pipeline_type"]}
        for key in ("vertices", "faces", "generation_time"):
            if key in response:
                meta[key] = response[key]
        return BackendResult(glb_path=glb_path, meta=meta)
