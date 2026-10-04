from __future__ import annotations

from pathlib import Path

import pytest
import trimesh
from PIL import Image

from trellis_forge.backends.base import ForgeError
from trellis_forge.backends.hf_space import ENDPOINTS, SPACE_ID, HfSpaceBackend, _explain, _glb_from
from trellis_forge.config import GenerationParams

pytest.importorskip("gradio_client")


class FakeClient:
    """Stands in for gradio_client.Client — no network, same call shape. File
    outputs come back as local paths, which is what the real client returns."""

    def __init__(self, cutout: Path, glb: Path, endpoints=ENDPOINTS) -> None:
        self.cutout = cutout
        self.glb = glb
        self.endpoints = list(endpoints)
        self.calls: list[tuple[str, tuple]] = []

    def view_api(self, return_format: str = "list", print_info: bool = True) -> dict:
        return {
            "named_endpoints": {name: {} for name in self.endpoints},
            "unnamed_endpoints": {},
        }

    def predict(self, *args, api_name: str):
        self.calls.append((api_name, args))
        if api_name == "/preprocess_image":
            return str(self.cutout)
        if api_name == "/extract_glb":
            return str(self.glb), str(self.glb)
        return "<preview html>"


@pytest.fixture
def space(tmp_path: Path, monkeypatch) -> tuple[HfSpaceBackend, FakeClient]:
    cutout = tmp_path / "cutout.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(cutout)
    glb = tmp_path / "remote.glb"
    trimesh.creation.box().export(glb)

    client = FakeClient(cutout, glb)
    backend = HfSpaceBackend()
    monkeypatch.setattr(backend, "_connect", lambda: client)
    return backend, client


def test_generate_drives_the_endpoint_sequence(space, tmp_path: Path) -> None:
    backend, client = space
    out = tmp_path / "asset"
    result = backend.generate(Image.new("RGB", (8, 8)), out, GenerationParams(seed=7))

    assert [name for name, _ in client.calls] == [
        "/start_session",
        "/preprocess_image",
        "/image_to_3d",
        "/extract_glb",
    ]
    # The Space runs /image_to_3d with preprocess_image=False, so the cutout
    # returned by /preprocess_image is what has to be fed forward.
    _, image_to_3d = client.calls[2]
    assert Path(image_to_3d[0]["path"]) == client.cutout
    assert image_to_3d[1:] == (7.0, "1024")

    assert result.glb_path == out / "mesh.glb"
    assert result.glb_path.exists()
    assert result.meta == {"space": SPACE_ID}


def test_decimation_is_only_sent_when_asked(space, tmp_path: Path) -> None:
    backend, client = space
    photo = Image.new("RGB", (8, 8))

    # Omitting the arg keeps the Space's own defaults (300k faces, 2K textures).
    backend.generate(photo, tmp_path / "a", GenerationParams())
    assert client.calls[-1][1] == ()

    backend.generate(photo, tmp_path / "b", GenerationParams(decimate_to=150000))
    assert client.calls[-1][1] == (150000.0,)


def test_is_available_probes_the_space_endpoints(monkeypatch) -> None:
    backend = HfSpaceBackend()
    monkeypatch.setattr(backend, "_connect", lambda: FakeClient(Path("c"), Path("g")))
    ok, reason = backend.is_available()
    assert ok is True
    assert "ZeroGPU" in reason

    drifted = HfSpaceBackend()
    monkeypatch.setattr(
        drifted,
        "_connect",
        lambda: FakeClient(Path("c"), Path("g"), endpoints=["/image_to_3d", "/extract_glb"]),
    )
    ok, reason = drifted.is_available()
    assert ok is False
    assert "/preprocess_image" in reason


def test_is_available_reports_an_unreachable_space(monkeypatch) -> None:
    def unreachable() -> None:
        raise ConnectionError("no route to host")

    backend = HfSpaceBackend()
    monkeypatch.setattr(backend, "_connect", unreachable)
    ok, reason = backend.is_available()
    assert ok is False
    assert "could not reach" in reason


def test_hf_token_is_forwarded_to_the_client(monkeypatch) -> None:
    import gradio_client

    seen: dict = {}

    def spy(src, token=None, verbose=True):
        seen.update(src=src, token=token, verbose=verbose)
        return object()

    monkeypatch.setattr(gradio_client, "Client", spy)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    HfSpaceBackend()._connect()
    assert seen == {"src": SPACE_ID, "token": "hf_secret", "verbose": False}


def test_glb_from_rejects_an_unexpected_payload() -> None:
    assert _glb_from(("out/mesh.glb", "out/mesh.glb")) == "out/mesh.glb"
    with pytest.raises(ForgeError):
        _glb_from({"unexpected": "shape"})


def test_quota_error_is_not_reported_as_api_drift() -> None:
    quota = _explain(RuntimeError("You have exceeded your ZeroGPU quota (120s requested)"))
    assert "HF_TOKEN" in quota
    assert "view_api" not in quota

    drift = _explain(RuntimeError("There is no endpoint named /image_to_3d"))
    assert "view_api" in drift
