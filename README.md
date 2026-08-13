# trellis-forge

> **A folder of images in, production-usable 3D assets out** — batched, reproducible, and traced.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/rahulrajan/trellis-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/rahulrajan/trellis-forge/actions/workflows/ci.yml)

```bash
trellis-forge generate ./photos -o ./assets
```

TRELLIS.2 is Microsoft Research's 4-billion-parameter model that turns a single
image into a fully textured PBR mesh (GLB). Upstream ships a research repo and
a demo Space, but no batch tooling. **trellis-forge is the batch front door:**
point it at a folder and every image becomes an asset — on your own NVIDIA GPU,
or on the free hosted demo when you don't have one.

![trellis-forge architecture — how a photo becomes a 3D asset](docs/images/architecture.svg)

---

## What is trellis-forge?

trellis-forge is a command-line wrapper around [Microsoft TRELLIS.2](https://microsoft.github.io/TRELLIS.2/)
that adds the things a research repo never will:

| Capability | What you get |
| --- | --- |
| **Batch** | Point it at a folder; every image becomes an asset, with per-image failure isolation and a live progress bar. |
| **Reproducibility** | The base seed is combined with each image's content hash, so every image gets its own seed and a rerun draws exactly the same samples. |
| **Provenance** | `manifest.json` records source-image hash, backend, seed, resolution, timings, and a geometry QC report. Interrupted batches resume where they stopped. |
| **Portability** | Two backends behind one flag: `official-cuda` (your GPU) and `hf-space` (the free hosted demo), auto-detected with `--backend auto`. |

It is deliberately *not* an interactive authoring tool.

> [!TIP]
> **ComfyUI wrappers are excellent for interactive asset authoring — use them.**
> trellis-forge is for everything ComfyUI is bad at: headless batches, CI-style
> reproducibility, and machine-readable provenance.

---

## How it works

Every run flows through the same five stages, no matter which backend does the
heavy lifting:

1. **Photos in** — `.jpg` / `.png` / `.webp` are collected (optionally recursed).
2. **Preprocess** — `--rembg` isolates the subject onto a clean white background.
3. **Backend** — `--backend auto` picks the first available engine (see below).
4. **Export + QC** — the backend's GLB is kept as the primary artifact; OBJ/STL are
   geometry-only convenience exports, and a watertight/face-count QC report is run.
5. **Assets out** — each asset lands in its own folder and the manifest is updated.

```mermaid
flowchart LR
    P["Photos"] --> Pre["Preprocess<br/>rembg"]
    Pre --> B{"Backend<br/>auto-select"}
    B --> C["official-cuda"]
    B --> H["hf-space"]
    C & H --> E["Export<br/>GLB · OBJ · STL"]
    E --> Q["QC report"]
    Q --> Man["manifest.json"]
```

---

## Which backend should I use?

The short version: **an NVIDIA GPU if you have one, the hosted demo if you
don't.** `--backend auto` does exactly that.

```mermaid
flowchart TD
    A["What machine are you on?"] -->|"Linux + NVIDIA ≥ 24 GB"| B["official-cuda<br/>the fast path"]
    A -->|"Anything else"| E["hf-space<br/>free hosted demo — works everywhere"]
```

| Backend | Runs on | Speed | Status |
| --- | --- | --- | --- |
| `official-cuda` | Linux + NVIDIA ≥ 24 GB (A100/H100/4090-class) | ~1–2 min/asset | supported path (via Dockerfile) |
| `hf-space` | **any machine with network** | queue-dependent | works today |

> [!NOTE]
> **`trellis-forge backends` tells you exactly which engines are usable on your
> machine.** It never guesses — it probes.

---

## Install

```bash
pip install trellis-forge             # everything, including the hosted backend
pip install "trellis-forge[rembg]"    # + --rembg background removal
```

(`pipx install trellis-forge` works too, if you prefer isolated CLI installs.)

The hosted backend needs nothing else. The CUDA backend additionally needs the
upstream TRELLIS.2 repo installed — the [Dockerfile](#official-cuda--the-fast-path)
packages that for you.

### Quick start

1. Install: `pip install trellis-forge`
2. Point it at a folder of photos:
   ```bash
   trellis-forge generate ./photos -o ./assets --rembg
   ```
3. Inspect the output: each image gets a folder with `mesh.glb`, and the run is
   recorded in `assets/manifest.json`.

---

## Backends, in depth

### official-cuda — the fast path

Requires Linux, CUDA 12.4, an NVIDIA GPU with ≥ 24 GB VRAM, and the upstream repo
installed into the same environment. The Dockerfile packages all of that:

```bash
docker build -t trellis-forge .
docker run --gpus all -v "$PWD/photos:/in" -v "$PWD/assets:/out" trellis-forge \
    generate /in -o /out --backend official-cuda --resolution 1024
```

No GPU of your own? Any cloud GPU works (RunPod/Vast/Lambda); an A100 hour
produces dozens of assets.

Details worth knowing:

- `--resolution` maps to upstream's `pipeline_type`: 512 runs direct sampling,
  1024 and 1536 run the cascade samplers.
- `--decimate` decimates backend-side during texture baking, so PBR textures
  survive; the OBJ/STL conversions are geometry-only.
- The export is wired exactly like upstream's own demo (`app.py`). It has not
  yet been validated on a live GPU run, so treat your first run as a shakedown
  and report anything odd.

### hf-space — works on any machine

Uses `gradio_client` against the official `microsoft/TRELLIS.2` demo Space. No
local GPU, no 15 GB weights download — the realistic path for any laptop, at
the cost of queue waits and ZeroGPU quota (anonymous visitors get only a few
GPU-minutes per day; an HF token buys more). The Space is a shared research
demo — rate-limit yourself, and if its API drifts the adapter fails loudly with
a `view_api()` hint.

---

## Usage

```bash
# What works on this machine?
trellis-forge backends

# A folder of product photos -> GLBs, subject auto-isolated
trellis-forge generate ./photos -o ./assets --rembg

# Game/web budget + printable STL alongside
trellis-forge generate ./photos -o ./assets --decimate 150000 --format glb --format stl

# Reproducible rerun of a subset (manifest skips finished images)
trellis-forge generate ./photos -o ./assets --seed 7 --limit 5

# No GPU at all — free hosted demo Space
trellis-forge generate ./photos -o ./assets --backend hf-space
```

Output layout:

```
assets/
├── manifest.json
├── molar-01-3fa2b9c1/
│   ├── mesh.glb          # textured PBR asset (primary artifact)
│   └── mesh.stl          # only if requested (geometry-only)
└── ...
```

Manifest paths are stored relative to `assets/`, so the whole tree can be moved,
synced, or shipped as-is.

---

## Notes on quality

- **Garbage in, garbage out.** Single-image 3D is only as good as the photo:
  isolated subject, neutral background, sharp focus, object filling the frame.
- **Mesh QC is in the manifest** (`watertight`, face count, bounds). Generated
  meshes can have small holes; filter on `qc.watertight` if your downstream
  pipeline (3D printing, boolean CSG) needs manifolds.
- **Expect ~1–2 min/asset on a 4090-class GPU** at 1024³ including export.
  1536³ is a quality/VRAM step up; 512³ is the low-memory fallback.

---

## Licensing — read before shipping assets

> [!IMPORTANT]
> trellis-forge itself is MIT. **TRELLIS.2 is not simply "yours to ship."** The
> code/weights carry MIT labels on GitHub/Hugging Face, but Microsoft's official
> project page restricts the materials to *academic and research use* and
> explicitly prohibits commercial exploitation. Treat the stricter statement as
> governing.

- **Fine:** research, education, non-commercial projects, internal prototyping.
- **Get written confirmation first:** shipping generated assets in anything that
  takes money (paid app, ads, subscriptions).

You are also responsible for the rights to your **input images** — don't feed it
photos of people, patients, or copyrighted work you don't hold.

---

## Roadmap

- [ ] GPU validation run for the rewritten `official-cuda` export path
- [ ] Preview renders (turntable/contact sheet) per asset
- [ ] Watch upstream Apple Silicon / MPS support; when it stabilizes it becomes
      a third backend

---

## Development

```bash
pip install -e ".[dev]"     # or: uv pip install -e ".[dev]"
pytest                      # fast — needs no GPU, torch, or network
ruff check src tests
```

CI runs the same two commands on Python 3.11–3.14. The heavy engines (torch,
trellis2, gradio_client) are imported lazily inside backend methods, so the CLI,
preprocessing, and tests run on machines that have none of them.
