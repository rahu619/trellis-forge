from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from trellis_forge.backends.official import OfficialBackend, _to_glb
from trellis_forge.config import GenerationParams


def _install(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_cuda_call_does_not_clobber_the_pipeline(monkeypatch) -> None:
    """Upstream's cuda()/to() return None, so the result must never be rebound —
    doing so left the backend holding None and reloaded ~15 GB per image."""
    fake = SimpleNamespace(cuda_calls=0)

    def cuda() -> None:
        fake.cuda_calls += 1

    fake.cuda = cuda
    _install(
        monkeypatch,
        "trellis2.pipelines",
        Trellis2ImageTo3DPipeline=SimpleNamespace(from_pretrained=lambda _: fake),
    )

    backend = OfficialBackend()
    assert backend._pipeline() is fake
    assert fake.cuda_calls == 1
    # Cached, so a batch loads the weights once rather than per image.
    assert backend._pipeline() is fake
    assert fake.cuda_calls == 1


def test_export_reads_layout_and_voxel_size_off_the_mesh(monkeypatch) -> None:
    """grid_size=params.resolution is wrong for cascade runs, which can decode at
    a resolution other than the one that was requested."""
    captured: dict = {}

    def fake_to_glb(**kwargs):
        captured.update(kwargs)
        return "glb"

    postprocess = _install(monkeypatch, "o_voxel.postprocess", to_glb=fake_to_glb)
    _install(monkeypatch, "o_voxel", postprocess=postprocess)

    layout = {"base_color": slice(0, 3)}
    mesh = SimpleNamespace(
        vertices="v", faces="f", attrs="a", coords="c", layout=layout, voxel_size=1 / 1536
    )
    params = GenerationParams(resolution=1024, decimate_to=150000)

    assert _to_glb(mesh, params) == "glb"
    assert captured["attr_layout"] == layout
    assert captured["voxel_size"] == 1 / 1536
    assert captured["decimation_target"] == 150000
    assert "grid_size" not in captured


def test_decimation_is_omitted_unless_asked(monkeypatch) -> None:
    captured: dict = {}
    postprocess = _install(
        monkeypatch, "o_voxel.postprocess", to_glb=lambda **kw: captured.update(kw)
    )
    _install(monkeypatch, "o_voxel", postprocess=postprocess)
    mesh = SimpleNamespace(
        vertices="v", faces="f", attrs="a", coords="c", layout={}, voxel_size=1 / 1024
    )

    _to_glb(mesh, GenerationParams())
    assert "decimation_target" not in captured, "must fall back to upstream's own default"
