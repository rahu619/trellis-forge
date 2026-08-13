from __future__ import annotations

import pytest

from trellis_forge.backends.hf_space import HfSpaceBackend, _find_glb


def test_is_available_reports_quota() -> None:
    pytest.importorskip("gradio_client")
    ok, reason = HfSpaceBackend().is_available()
    assert ok is True
    assert "ZeroGPU" in reason


def test_find_glb_searches_nested_structures() -> None:
    assert _find_glb("out/mesh.glb") == "out/mesh.glb"
    assert _find_glb("out/mesh.obj") is None
    assert _find_glb({"a": ["x.txt", {"b": "deep/y.glb"}]}) == "deep/y.glb"
    assert _find_glb({"a": ["x.txt", {"b": "nope"}]}) is None
