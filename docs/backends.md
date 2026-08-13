# Backend notes

## official-cuda

- Requirements: Linux, CUDA 12.4, NVIDIA ≥24 GB VRAM, `microsoft/TRELLIS.2`
  installed in the active environment (or use the Dockerfile).
- Model weights (`microsoft/TRELLIS.2-4B`, ~15 GB) download on first run via
  Hugging Face Hub.
- `--resolution` maps to upstream's `pipeline_type`
  (`trellis2/pipelines/trellis2_image_to_3d.py`): 512→`512` (direct),
  1024→`1024_cascade`, 1536→`1536_cascade`.
- Export is wired exactly like upstream's demo (`app.py` → `extract_glb`):
  `o_voxel.postprocess.to_glb(vertices, faces, attr_volume=mesh.attrs,
  coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
  grid_size=resolution, aabb=[±0.5]³, remesh=True)`. Decimation is
  backend-side (`decimation_target`) so textures survive; trimesh-based
  OBJ/STL conversion is geometry-only.
- The upstream pipeline preprocesses the input image internally (its own
  BiRefNet-based rembg). `--rembg` still matters: it composites the subject
  onto white before the backend ever sees it, which is what `hf-space`
  benefits from most.
- v0.2 status: the adapter tracks upstream source but has not yet been
  validated on a live GPU run — treat the first run as a shakedown.

## hf-space

- Uses `gradio_client` against the official `microsoft/TRELLIS.2` demo Space.
  Generation is a stateful three-step flow: `/start_session` →
  `/image_to_3d(image, seed, resolution)` → `/extract_glb`.
- No local GPU, no weights download — the realistic path for any laptop.
- Costs: queue waits, ZeroGPU quota (anonymous: a few GPU-minutes/day; an HF
  token buys more), possible API drift (the adapter fails loudly with a
  `view_api()` hint if the signature changes).
- Rate-limit yourself; it's a shared research demo.

## Adding a backend

Implement `trellis_forge.backends.base.Backend` (two methods: `is_available`,
`generate` writing a GLB), register it in `backends/__init__.py`, and keep all
heavy imports inside methods so machines without that stack can still run the
rest of the tool.
