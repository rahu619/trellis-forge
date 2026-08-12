# AGENTS.md

Guidance for AI coding agents (Qwen Code, Claude Code, etc.) working in this repo.
This file is loaded into context every session — keep it short and factual.
`QWEN.md` and `CLAUDE.md` are one-line imports of this file; edit here, not there.

## Project in one paragraph

trellis-forge is a Python CLI that batch-converts images into 3D assets (GLB) using
Microsoft's TRELLIS.2 image-to-3D model. It wraps four inference backends
(`official-cuda`, `mlx`, `official-mps`, `hf-space`) behind one `Backend` interface
and adds batching, deterministic seeding, a provenance manifest with resume, and mesh
QC. It is a headless, CI-style pipeline — deliberately not an interactive tool.

## Commands

Repo venv lives at `.venv` (dev machine: Apple Silicon macOS).

```bash
pip install -e ".[dev]"           # editable install + pytest + ruff
pytest                            # full suite — fast, needs no GPU/torch/network
pytest tests/test_pipeline.py::test_resume_skips_finished_images   # single test
ruff check src tests              # run before finishing any change
trellis-forge backends            # probe which backends are usable on this box
```

Tests never run real inference engines — they use `FakeBackend`
(`tests/conftest.py`), which writes a real tiny watertight GLB via trimesh.

## Architecture

```
src/trellis_forge/
├── cli.py        typer app: generate / backends / quantize-mlx / version
├── config.py     GenerationParams (frozen dataclass) + validation
├── pipeline.py   collect_images → run_batch (per-image loop, resume, isolation)
├── preprocess.py load / downscale / optional rembg
├── backends/     Backend ABC; official.py (cuda & mps), mlx.py, hf_space.py
├── quant/        quantization library: markers.py + mlx_weights.py (4-bit for ≤16 GB Macs)
├── export.py     GLB primary; OBJ/STL geometry-only convenience exports
├── qc.py         trimesh watertight / face-count / bounds report
├── manifest.py   sha256-keyed provenance ledger + resume
└── sysinfo.py    stdlib-only machine probing (memory-aware on Apple Silicon)
```

Flow: `generate` → `resolve_backend()` → `run_batch()` → per image:
preprocess → `backend.generate()` → `export_formats()` → `qc.inspect()` →
manifest entry, saved after every success.

## Invariants — do not break these

1. **Lazy heavy imports.** torch / trellis2 / mlx / gradio_client / rembg are
   imported inside methods only, never at module top level. The CLI and tests must
   run on machines with none of them installed. Heavy deps belong in extras
   (`[hf]`, `[rembg]`), never in core `[project.dependencies]`.
2. **Manifest is the resume ledger.** Keyed by source-image sha256; asset dirs are
   named `{stem}-{hash[:8]}`; saved after every success. Don't change the schema
   casually — shipped output directories depend on it. Corrupt manifests are
   silently restarted, never crash.
3. **Per-image failure isolation.** `run_batch` catches all exceptions per image on
   purpose (`noqa: BLE001`); one bad image must never stop a batch. Report failures
   in `BatchSummary.failed`, don't raise.
4. **Reproducibility.** Base seed is combined with each image's hash; same inputs +
   flags ⇒ same asset dir names on rerun.
5. **GLB is the primary artifact.** OBJ/STL are geometry-only exports.
6. **sysinfo probes never raise** — they degrade to `None` / `False`.
7. **Backend contract.** Implement `is_available() -> (bool, reason)` and
   `generate(image, out_dir, params) -> BackendResult`; raise `ForgeError` for
   user-facing setup problems. `AUTO_PRIORITY` order in `backends/__init__.py`
   matters (real GPU → Mac ports → hosted).

## Testing conventions

- Extend `FakeBackend` / fixtures in `tests/conftest.py` rather than mocking
  trimesh or the export path — the suite exercises real export + QC on a box mesh.
- Fixture images have deliberately distinct pixels: the manifest dedupes by content
  hash, so identical images would correctly collapse into one asset.
- New backend? Test `is_available()` logic and its HTTP/process glue with fakes;
  never require real weights, network, or GPU in CI.

## Style

- ruff: line-length 100, rules `E,F,I,UP,B`, `B008` ignored (typer
  `Option(...)`/`Argument(...)` in signature defaults is the idiom). Match it.
- `from __future__ import annotations` everywhere; dataclasses for param objects;
  type hints throughout; no comment unless the *why* is non-obvious.
- `ForgeError` for user-facing failures; rich `Console` for output — escape
  user-provided strings with `rich.markup.escape`.

## Docs & licensing

- Keep `README.md` and `docs/backends.md` honest about backend maturity
  (supported vs experimental) and the 16 GB-Mac limitations.
- TRELLIS.2 weights/code are restricted to research/academic use by Microsoft's
  project page; never write copy or code that promises commercial use of generated
  assets.
