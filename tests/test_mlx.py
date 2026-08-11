from __future__ import annotations

import base64
from pathlib import Path

import pytest
from PIL import Image

from trellis_forge import backends
from trellis_forge.backends import mlx as mlx_mod
from trellis_forge.backends.base import ForgeError
from trellis_forge.backends.mlx import (
    MlxBackend,
    _build_payload,
    _parse_glb,
    pipeline_type_for,
)
from trellis_forge.config import GenerationParams


def _image() -> Image.Image:
    return Image.new("RGB", (8, 8), (10, 20, 30))


def test_pipeline_type_mapping() -> None:
    assert pipeline_type_for(512) == "512"
    assert pipeline_type_for(1024) == "1024_cascade"
    assert pipeline_type_for(1536) == "1536_cascade"
    with pytest.raises(ForgeError):
        pipeline_type_for(256)


def test_build_payload_encodes_image_and_seed() -> None:
    payload = _build_payload(_image(), GenerationParams(resolution=512, seed=7))
    assert payload["seed"] == 7
    assert payload["pipeline_type"] == "512"
    assert "decimation_target" not in payload
    # Round-trips to a real PNG.
    assert base64.b64decode(payload["image"]).startswith(b"\x89PNG")


def test_build_payload_includes_decimation_when_set() -> None:
    payload = _build_payload(
        _image(), GenerationParams(resolution=1024, decimate_to=150000)
    )
    assert payload["decimation_target"] == 150000


def test_parse_glb_roundtrip() -> None:
    blob = b"glTF-binary-bytes"
    response = {"glb": base64.b64encode(blob).decode(), "vertices": 8}
    assert _parse_glb(response) == blob


def test_parse_glb_missing_raises() -> None:
    with pytest.raises(ForgeError):
        _parse_glb({"vertices": 8})


def test_unavailable_without_repo_or_server(monkeypatch) -> None:
    monkeypatch.delenv(mlx_mod.DEFAULT_REPO_ENV, raising=False)
    monkeypatch.setattr(mlx_mod, "_get_json", _raise)
    ok, reason = MlxBackend().is_available()
    assert ok is False
    assert "clone" in reason


def test_available_when_server_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        mlx_mod, "_get_json", lambda url, timeout: {"status": "ok", "weights_loaded": True}
    )
    ok, reason = MlxBackend(url="http://mlx.test").is_available()
    assert ok is True
    assert "connected" in reason


def test_unavailable_when_weights_not_loaded(monkeypatch) -> None:
    monkeypatch.setattr(
        mlx_mod, "_get_json", lambda url, timeout: {"status": "ok", "weights_loaded": False}
    )
    ok, reason = MlxBackend(url="http://mlx.test").is_available()
    assert ok is False
    assert "weights" in reason


def test_repo_ready_but_server_down_guides_start(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "api_server.py").write_text("")
    monkeypatch.setattr(mlx_mod, "_get_json", _raise)
    ok, reason = MlxBackend(repo=tmp_path).is_available()
    assert ok is False
    assert "python api_server.py" in reason


def test_generate_writes_glb_and_meta(monkeypatch, tmp_path: Path) -> None:
    blob = b"fake-glb"
    monkeypatch.setattr(
        mlx_mod, "_get_json", lambda url, timeout: {"status": "ok", "weights_loaded": True}
    )
    monkeypatch.setattr(
        mlx_mod,
        "_post_json",
        lambda url, payload, timeout: {
            "glb": base64.b64encode(blob).decode(),
            "vertices": 8,
            "faces": 12,
            "generation_time": 1.5,
        },
    )
    result = MlxBackend(url="http://mlx.test").generate(
        _image(), tmp_path / "asset", GenerationParams(resolution=512)
    )
    assert result.glb_path.read_bytes() == blob
    assert result.meta["pipeline_type"] == "512"
    assert result.meta["faces"] == 12


def test_generate_without_server_raises_guidance(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mlx_mod, "_get_json", _raise)
    with pytest.raises(ForgeError) as exc:
        MlxBackend(url="http://mlx.test").generate(
            _image(), tmp_path / "asset", GenerationParams(resolution=512)
        )
    assert "api_server.py" in str(exc.value)


def test_registry_still_exposes_mlx() -> None:
    # The adapter's constructor signature changed; keep the registry wiring intact.
    registry = backends.all_backends()
    assert "mlx" in registry


def _raise(url, timeout):
    raise ConnectionError("server down")
