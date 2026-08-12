"""End-to-end check of `trellis-forge quantize-mlx` against a stubbed port.

Real CLI, real filesystem layout, real imports of a stub trellis2-mlx
checkout; only MLX itself is faked (no weights, no GPU, no network). This is
the smoke gate for the quantization path — deep unit coverage deliberately
lives downstream of an actual 16 GB Mac validation run.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trellis_forge.backends.mlx import DEFAULT_REPO_ENV
from trellis_forge.cli import app
from trellis_forge.quant.markers import QUANT_MARKER, WEIGHTS_ENV

runner = CliRunner()

_FLOW_ARGS = {
    "resolution": 16,
    "in_channels": 8,
    "model_channels": 1536,
    "cond_channels": 1024,
    "out_channels": 8,
    "num_blocks": 30,
}

_MLX_BACKEND_INIT = """
def load_safetensors(path, dtype=None):
    return {"block.weight": b"fp-bytes"}

def remap_flow_model_weights(weights):
    return weights

def remap_vae_decoder_weights(weights):
    return weights
"""

_FLOW_MODELS = """
class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.loaded = None

    def load_weights(self, items):
        self.loaded = dict(items)

    def parameters(self):
        return {"block.weight": b"fp-bytes"}

MlxSparseStructureFlowModel = FakeModel
MlxSLatFlowModel = FakeModel
"""


@pytest.fixture
def stub_port(tmp_path: Path, monkeypatch) -> Path:
    repo = tmp_path / "trellis2-mlx"
    backend = repo / "mlx_backend"
    backend.mkdir(parents=True)
    (repo / "api_server.py").write_text("")
    (backend / "__init__.py").write_text(_MLX_BACKEND_INIT)
    (backend / "flow_models.py").write_text(_FLOW_MODELS)
    (backend / "vae_decoders.py").write_text("")
    (backend / "structure_decoder.py").write_text("")

    weights = repo / "weights" / "TRELLIS.2-4B"
    (weights / "ckpts").mkdir(parents=True)
    (weights / "pipeline.json").write_text(
        json.dumps(
            {
                "models": {
                    "sparse_structure_flow_model": "ckpts/sparse_structure_flow_model",
                    "flow_model": "ckpts/flow_model",
                    "image_cond_model": "remote/dinov3",
                }
            }
        )
    )
    for name, kind in [
        ("sparse_structure_flow_model", "SparseStructureFlowModel"),
        ("flow_model", "SLatFlowModel"),
    ]:
        base = weights / "ckpts" / name
        base.with_suffix(".json").write_text(json.dumps({"name": kind, "args": _FLOW_ARGS}))
        base.with_suffix(".safetensors").write_bytes(b"\x00" * 64)

    # Fake MLX: quantize is a no-op marker, save writes a small real file so
    # the size accounting and verify-reload path run for real.
    mx_core = types.ModuleType("mlx.core")
    mx_core.quantize = lambda model, group_size=64, bits=4: setattr(model, "bits", bits)
    mx_core.save_safetensors = lambda path, tensors, metadata=None: Path(path).write_bytes(
        b"q4" * 8
    )
    mx_core.metal = types.SimpleNamespace(clear_cache=lambda: None)
    mx_utils = types.ModuleType("mlx.utils")
    mx_utils.tree_flatten = lambda tree: list(tree.items())
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core, mlx_pkg.utils = mx_core, mx_utils
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx_core)
    monkeypatch.setitem(sys.modules, "mlx.utils", mx_utils)

    monkeypatch.delenv(DEFAULT_REPO_ENV, raising=False)
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    yield repo
    for name in [m for m in sys.modules if m.split(".")[0] == "mlx_backend"]:
        del sys.modules[name]


def test_quantize_mlx_end_to_end(stub_port: Path) -> None:
    result = runner.invoke(app, ["quantize-mlx", "--repo", str(stub_port)])
    assert result.exit_code == 0, result.output
    assert "done" in result.output
    assert "skipped image_cond_model" in result.output

    weights = stub_port / "weights" / "TRELLIS.2-4B"
    for name in ("sparse_structure_flow_model", "flow_model"):
        assert (weights / "ckpts" / f"{name}.q4.safetensors").is_file()
    marker = json.loads((weights / QUANT_MARKER).read_text())
    assert marker["bits"] == 4
    assert set(marker["components"]) == {"sparse_structure_flow_model", "flow_model"}

    rerun = runner.invoke(app, ["quantize-mlx", "--repo", str(stub_port)])
    assert rerun.exit_code == 0
    assert "already quantized" in rerun.output


def test_quantize_mlx_without_checkout_guides(monkeypatch) -> None:
    monkeypatch.delenv(DEFAULT_REPO_ENV, raising=False)
    result = runner.invoke(app, ["quantize-mlx"])
    assert result.exit_code == 2
    assert "clone" in result.output
