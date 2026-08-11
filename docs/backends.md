# Backend notes

## official-cuda

- Requirements: Linux, CUDA 12.4, NVIDIA ≥24 GB VRAM, `microsoft/TRELLIS.2`
  installed in the active environment (or use the Dockerfile).
- Model weights (`microsoft/TRELLIS.2-4B`, ~15 GB) download on first run via
  Hugging Face Hub.
- Decimation is backend-side (`to_glb(decimation_target=...)`) so textures
  survive; trimesh-based OBJ/STL conversion is geometry-only.

## official-mps

- Same pipeline, Apple Silicon device. Requires the MPS fix from
  microsoft/TRELLIS.2 PR #167 applied to your checkout (unmerged upstream at
  time of writing; blocked on CLA only).
- Verified upstream on M3 Pro: ~24 min generation + ~25 s baking per asset at
  1024³, producing 2K PBR GLB. 1536³ unverified on MPS — use 512³/1024³.
- Memory: unified memory is shared with the OS; close heavy apps during runs.

## mlx

- The port now ships a FastAPI server (`api_server.py`) with `POST /generate`
  (base64 image in, base64 GLB out) and `GET /health`, so the adapter talks to it
  over HTTP instead of importing it — the server can live in its own environment
  with MLX and the ~15 GB of weights. Start it once (`python api_server.py`),
  then run with `--backend mlx`.
- Discovery: `TRELLIS2_MLX_REPO` points at the clone (used for guidance when no
  server is reachable); `TRELLIS2_MLX_URL` overrides the server address
  (default `http://127.0.0.1:8000`). `is_available()` is true only when a server
  responds to `/health` with `weights_loaded: true`.
- Resolution maps to the port's `pipeline_type`: 512→`512`, 1024→`1024_cascade`,
  1536→`1536_cascade`. `--decimate` maps to `decimation_target`.
- 16 GB Macs: port docs validate only 128 GB; use `--resolution 512` and treat
  success as best-effort. Quantized weights (4-bit via `mlx.utils.quantize`) are
  the planned fix and don't exist yet — contributing them is the project's
  headline roadmap item.

## hf-space

- Uses `gradio_client` against the official `microsoft/TRELLIS.2` demo Space.
- No local GPU, no weights download — the realistic path for any laptop.
- Costs: queue waits, Space-chosen resolution, possible API drift (the adapter
  fails loudly with a `view_api()` hint if the signature changes).
- Rate-limit yourself; it's a shared research demo.

## Adding a backend

Implement `trellis_forge.backends.base.Backend` (two methods: `is_available`,
`generate` writing a GLB), register it in `backends/__init__.py`, and keep all
heavy imports inside methods so machines without that stack can still run the
rest of the tool.
