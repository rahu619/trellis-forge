"""4-bit weight quantization for the trellis2-mlx port.

TRELLIS.2 ships ~15 GB of full-precision weights, which leaves a 16 GB
unified-memory Mac nothing for activations and the OS. Quantizing the Linear
layers of the flow transformers and VAE decoders to 4-bit shrinks the
footprint to ~4 GB — the difference between "best-effort 512^3" and local
generation that actually fits.

Orchestration against a local checkout of gtrg55/trellis2-mlx: each model is
constructed exactly the way the port's pipeline builds it, full-precision
weights are loaded, ``mx.quantize`` is applied, and ``{name}.q4.safetensors``
files are written alongside the originals (recorded in ``quantized.json``).

The port's ``api_server.py`` does not load quantized weights yet — the small
upstream hook is documented in ``docs/mlx-quantization.md``. Until it lands,
this module produces and verifies the weights; serving them is a follow-up.

All MLX / port imports happen inside functions: importing this module must
work on machines with none of them installed.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.markup import escape

from .. import __version__
from ..backends.base import ForgeError
from ..backends.mlx import DEFAULT_REPO_ENV
from .markers import DEFAULT_WEIGHTS_REL, WEIGHTS_ENV, read_marker, write_marker

Q4_SUFFIX = ".q4.safetensors"
UPSTREAM_REPO = "https://github.com/gtrg55/trellis2-mlx"

# Constructor kwargs shared by both flow-model classes in the port
# (mlx_backend/flow_models.py). Mirror the port's loader defaults so the
# quantized module tree matches the one that serves the fp weights.
_FLOW_KWARG_DEFAULTS = {
    "num_heads": 12,
    "mlp_ratio": 5.3334,
    "pe_mode": "rope",
    "share_mod": True,
    "qk_rms_norm": True,
    "qk_rms_norm_cross": True,
}
_FLOW_REQUIRED = (
    "resolution",
    "in_channels",
    "model_channels",
    "cond_channels",
    "out_channels",
    "num_blocks",
)


@dataclasses.dataclass(frozen=True)
class Component:
    """One quantizable model: a config/weights pair under ``ckpts/``."""

    name: str
    config_path: Path
    weights_path: Path

    @property
    def q4_path(self) -> Path:
        return self.weights_path.with_name(self.weights_path.stem + Q4_SUFFIX)


@dataclasses.dataclass(frozen=True)
class ComponentResult:
    name: str
    q4_path: Path
    before_bytes: int
    after_bytes: int
    verified: bool


@dataclasses.dataclass
class QuantSummary:
    bits: int
    group_size: int
    results: list[ComponentResult] = dataclasses.field(default_factory=list)
    existing: list[str] = dataclasses.field(default_factory=list)
    skipped: list[tuple[str, str]] = dataclasses.field(default_factory=list)

    @property
    def saved_bytes(self) -> int:
        return sum(r.before_bytes - r.after_bytes for r in self.results)


@dataclasses.dataclass
class PortHandles:
    """Everything quantization needs from MLX and the port, lazily imported."""

    mx: Any
    quantize: Callable[..., None]
    tree_flatten: Callable[..., list]
    load_safetensors: Callable[..., dict]
    remap_flow_model_weights: Callable[[dict], dict]
    remap_vae_decoder_weights: Callable[[dict], dict]
    flow_models: Any
    vae_decoders: Any
    structure_decoder: Any


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def resolve_repo(repo: Path | None) -> Path:
    if repo is None:
        env = os.environ.get(DEFAULT_REPO_ENV)
        repo = Path(env) if env else None
    if repo is None:
        raise ForgeError(
            f"no trellis2-mlx checkout found — clone {UPSTREAM_REPO}, then pass "
            f"--repo or set {DEFAULT_REPO_ENV}"
        )
    if not repo.is_dir():
        raise ForgeError(f"{repo} is not a directory")
    if not (repo / "mlx_backend").is_dir():
        raise ForgeError(
            f"{repo} does not look like a trellis2-mlx checkout (no mlx_backend/)"
        )
    return repo


def resolve_weights_dir(repo: Path, weights: Path | None) -> Path:
    if weights is None:
        env = os.environ.get(WEIGHTS_ENV)
        weights = Path(env) if env else repo / DEFAULT_WEIGHTS_REL
    if not (weights / "pipeline.json").is_file():
        raise ForgeError(
            f"no pipeline.json in {weights} — download the weights first: "
            f"`cd {repo} && python scripts/download_weights.py`"
        )
    if not (weights / "ckpts").is_dir():
        raise ForgeError(f"no ckpts/ directory in {weights} — the download looks incomplete")
    return weights


def discover_components(weights_dir: Path) -> tuple[list[Component], list[tuple[str, str]]]:
    """Map pipeline.json model entries to local checkpoint pairs.

    Entries without a local ``{base}.json``/``{base}.safetensors`` pair (e.g.
    remotely-resolved paths) are reported as skipped, not errors.
    """
    pipeline = json.loads((weights_dir / "pipeline.json").read_text())
    models = pipeline.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ForgeError(f"{weights_dir / 'pipeline.json'} lists no models to quantize")
    components: list[Component] = []
    skipped: list[tuple[str, str]] = []
    for name, rel_path in models.items():
        base = weights_dir / rel_path
        config_path = base.with_suffix(".json")
        weights_path = base.with_suffix(".safetensors")
        if not config_path.is_file() or not weights_path.is_file():
            skipped.append((name, f"no local checkpoint pair at {rel_path}"))
            continue
        components.append(Component(name=name, config_path=config_path, weights_path=weights_path))
    return components, skipped


def find_quant_marker(repo: Path | None = None, weights: Path | None = None) -> dict | None:
    """Best-effort marker lookup for status displays; never raises."""
    try:
        weights_dir = resolve_weights_dir(resolve_repo(repo), weights)
    except ForgeError:
        return None
    return read_marker(weights_dir)


def _import_port(repo: Path) -> PortHandles:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_backend import (
            flow_models,
            load_safetensors,
            remap_flow_model_weights,
            remap_vae_decoder_weights,
            structure_decoder,
            vae_decoders,
        )
    except ImportError as exc:
        raise ForgeError(
            f"could not import MLX and the port's mlx_backend from {repo}: {exc}. "
            f"Run this command in the port's own environment: `cd {repo} && "
            "pip install -r requirements_macos.txt`."
        ) from exc
    quantize_fn = getattr(mx, "quantize", None)
    if quantize_fn is None:
        try:
            from mlx.utils import quantize as quantize_fn  # older MLX layouts
        except ImportError as exc:
            raise ForgeError("this MLX version has no quantize() — upgrade mlx") from exc
    return PortHandles(
        mx=mx,
        quantize=quantize_fn,
        tree_flatten=tree_flatten,
        load_safetensors=load_safetensors,
        remap_flow_model_weights=remap_flow_model_weights,
        remap_vae_decoder_weights=remap_vae_decoder_weights,
        flow_models=flow_models,
        vae_decoders=vae_decoders,
        structure_decoder=structure_decoder,
    )


def _flow_kwargs(args: dict) -> dict:
    kwargs = {key: args[key] for key in _FLOW_REQUIRED}
    kwargs.update({key: args.get(key, default) for key, default in _FLOW_KWARG_DEFAULTS.items()})
    return kwargs


def _build_model(component: Component, port: PortHandles) -> Any:
    """Construct the bare MLX module the same way the port's pipeline does."""
    config = json.loads(component.config_path.read_text())
    kind = config["name"]
    args = config.get("args", {})
    if "flow_model" in component.name:
        if kind == "SparseStructureFlowModel":
            return port.flow_models.MlxSparseStructureFlowModel(**_flow_kwargs(args))
        if kind in ("SLatFlowModel", "ElasticSLatFlowModel"):
            return port.flow_models.MlxSLatFlowModel(**_flow_kwargs(args))
        raise ForgeError(f"unknown flow model kind {kind!r} for {component.name}")
    if kind == "FlexiDualGridVaeDecoder":
        return port.vae_decoders.MlxFlexiDualGridVaeDecoder(
            resolution=args["resolution"],
            model_channels=args["model_channels"],
            latent_channels=args["latent_channels"],
            num_blocks=args["num_blocks"],
            block_type=args["block_type"],
            up_block_type=args["up_block_type"],
            block_args=args.get("block_args"),
            use_fp16=args.get("use_fp16", False),
        )
    if kind == "SparseUnetVaeDecoder":
        return port.vae_decoders.MlxSparseUnetVaeDecoder(
            out_channels=args["out_channels"],
            model_channels=args["model_channels"],
            latent_channels=args["latent_channels"],
            num_blocks=args["num_blocks"],
            block_type=args["block_type"],
            up_block_type=args["up_block_type"],
            block_args=args.get("block_args"),
            use_fp16=args.get("use_fp16", False),
            pred_subdiv=args.get("pred_subdiv", True),
        )
    raise ForgeError(
        f"no quantization recipe for model kind {kind!r} ({component.name}) — the port "
        "may have changed; please report it"
    )


def _load_fp_weights(model: Any, component: Component, port: PortHandles) -> None:
    """Load full-precision weights with the port's remapping rules."""
    config = json.loads(component.config_path.read_text())
    weights = port.load_safetensors(str(component.weights_path))
    if "flow_model" in component.name:
        weights = port.remap_flow_model_weights(weights)
    elif config["name"] == "FlexiDualGridVaeDecoder":
        weights = port.remap_vae_decoder_weights(weights)
        weights = {f"decoder.{key}": value for key, value in weights.items()}
    else:
        weights = port.remap_vae_decoder_weights(weights)
    model.load_weights(list(weights.items()))


def _quantize_component(
    component: Component, port: PortHandles, bits: int, group_size: int
) -> ComponentResult:
    if component.name == "sparse_structure_decoder":
        # The port's helper constructs and loads this one in a single call.
        base = component.config_path.with_suffix("")
        model = port.structure_decoder.load_structure_decoder(str(base))
    else:
        model = _build_model(component, port)
        _load_fp_weights(model, component, port)

    before = component.weights_path.stat().st_size
    port.quantize(model, group_size=group_size, bits=bits)
    weights = dict(port.tree_flatten(model.parameters()))
    metadata = {
        "quantized": "true",
        "bits": str(bits),
        "group_size": str(group_size),
        "source": component.weights_path.name,
        "tool": f"trellis-forge {__version__}",
    }
    port.mx.save_safetensors(str(component.q4_path), weights, metadata=metadata)
    del model, weights

    verified = False
    if component.name != "sparse_structure_decoder":
        # Rebuild from scratch and load the q4 file back: proves the saved
        # keys match the module tree a serving pipeline would quantize.
        fresh = _build_model(component, port)
        port.quantize(fresh, group_size=group_size, bits=bits)
        fresh.load_weights(list(port.load_safetensors(str(component.q4_path)).items()))
        del fresh
        verified = True

    _clear_cache(port)
    return ComponentResult(
        name=component.name,
        q4_path=component.q4_path,
        before_bytes=before,
        after_bytes=component.q4_path.stat().st_size,
        verified=verified,
    )


def _clear_cache(port: PortHandles) -> None:
    # Sequential components on a 16 GB box: give each one its memory back.
    clear = getattr(getattr(port.mx, "metal", None), "clear_cache", None)
    if clear is not None:
        clear()


def quantize_weights(
    repo: Path,
    weights_dir: Path,
    components: list[Component],
    bits: int = 4,
    group_size: int = 64,
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> QuantSummary:
    if bits not in (2, 3, 4, 6, 8):
        raise ForgeError(f"unsupported bit width {bits} — use 4")
    port = _import_port(repo)
    summary = QuantSummary(bits=bits, group_size=group_size)
    say = log or (lambda _message: None)
    for component in components:
        if component.q4_path.is_file() and not force:
            summary.existing.append(component.name)
            say(f"[dim]{escape(component.name)}:[/dim] {component.q4_path.name} exists, skipping")
            continue
        say(f"[cyan]{escape(component.name)}:[/cyan] quantizing to {bits}-bit…")
        try:
            result = _quantize_component(component, port, bits, group_size)
        except ForgeError:
            raise
        except Exception as exc:  # noqa: BLE001 — port drift must fail loudly, not silently
            raise ForgeError(
                f"quantizing {component.name} failed: {exc}. The port's internals may "
                "have drifted — report this with your trellis2-mlx commit hash."
            ) from exc
        summary.results.append(result)
        saved = (1 - result.after_bytes / result.before_bytes) * 100 if result.before_bytes else 0
        check = "verified" if result.verified else "saved (verify hook unavailable)"
        say(
            f"[green]✓ {escape(component.name)}[/green]: "
            f"{human_bytes(result.before_bytes)} → {human_bytes(result.after_bytes)} "
            f"(-{saved:.0f}%, {check})"
        )
    if summary.results:
        write_marker(
            weights_dir,
            bits=summary.bits,
            group_size=summary.group_size,
            components={
                r.name: str(r.q4_path.relative_to(weights_dir)) for r in summary.results
            },
            tool_version=__version__,
        )
    return summary
