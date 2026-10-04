# Backend notes

Verified against `microsoft/TRELLIS.2` `main` and the live demo Space on
2026-10-04. Upstream shipped its Init Release on 2025-12-09 and its last feature
commit on 2026-01-10 ("Release Training Code"); everything after that is CodeQL
workflow config from 2026-06-05. Treat the API as stable — drift risk is low.
There is no PyPI package (`trellis2` 404s), so a git clone is still the only
install path, and `microsoft/TRELLIS.2-4B` is the only checkpoint — no smaller
variant exists.

## official-cuda

- Requirements: Linux, CUDA 12.4, NVIDIA ≥24 GB VRAM (upstream: "verified on
  A100 and H100"), and `microsoft/TRELLIS.2` installed in the active environment
  — or use the Dockerfile. Weights (~15 GB) download on first run.
- Install is `git clone --recursive` plus a **sourced** setup script:
  `. ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm`.
  Two traps: with no flags the script forces `HELP=true` (it checks `$# -eq 1`)
  and installs nothing, and its help path uses `return`, which fails outright
  under `bash setup.sh` instead of `. ./setup.sh`. setup.sh pins
  torch 2.6.0 / torchvision 0.21.0 on cu124 and, under `--new-env`, creates a
  py3.10 conda env. The Dockerfile skips `--new-env` and installs into a py3.11
  venv, since trellis-forge requires ≥3.11 and upstream's README allows 3.8+.
- `--resolution` maps to `pipeline_type` in
  `trellis2/pipelines/trellis2_image_to_3d.py`: 512→`512` (direct),
  1024→`1024_cascade`, 1536→`1536_cascade`. A fourth value, `1024` (direct),
  exists upstream and is deliberately not exposed.
- **`Pipeline.to()` and `.cuda()` both return `None`.** Call `cuda()` as a bare
  statement — rebinding its result silently discards the loaded pipeline.
- `low_vram` defaults to `True`, which streams each sub-model (rembg, the DINOv3
  conditioner, both flow models, both decoders) onto the GPU only while it runs;
  in that mode `to()` merely records the device.
- Export follows `app.py`'s `extract_glb`: `o_voxel.postprocess.to_glb(vertices,
  faces, attr_volume=mesh.attrs, coords=mesh.coords, aabb=[±0.5]³, remesh=True,
  remesh_band=1, remesh_project=0)` then `glb.export(path, extension_webp=True)`.
  We pass `attr_layout=mesh.layout` and `voxel_size=mesh.voxel_size` where app.py
  passes `attr_layout=pipeline.pbr_attr_layout` and `grid_size=res`. Those are the
  same values — `MeshWithVoxel` is built with `layout=self.pbr_attr_layout` and
  `voxel_size=1/resolution` — but reading them off the mesh stays correct when a
  cascade decodes at a resolution other than the one requested. `to_glb` takes
  `voxel_size` **or** `grid_size`, never both.
- `to_glb` defaults to `decimation_target=1_000_000`, `texture_size=2048`; the
  Space defaults decimation to 300_000. Texture size matches, so there's no flag
  for it, but the same `--decimate` value is needed on both paths for comparable
  file sizes.
- Do **not** call `mesh.simplify(16777216)`. `example.py` needs it for the
  nvdiffrast render path; `app.py` re-decodes a fresh mesh for export and never
  simplifies it.
- Preprocessing is upstream's, not ours: `run(preprocess_image=True)` is the
  default and runs BiRefNet with `briaai/RMBG-2.0` (per the checkpoint's
  `pipeline.json`), downscales the long edge to ≤1024, crops to a square around
  the alpha>0.8 bbox, and premultiplies — using an existing alpha channel directly
  when the input has one. That's why `load_image` preserves RGBA rather than
  flattening it.
- Status: wiring matches upstream source line for line, but has never executed on
  a real GPU.

## hf-space

- `gradio_client` ≥2.1 (1.x named the auth kwarg `hf_token`, 2.x calls it `token`)
  against the official `microsoft/TRELLIS.2` demo Space, which runs gradio 6.1.0
  on ZeroGPU `zero-a10g`.
- Endpoints, verified against the live `/gradio_api/info`:
  - `/start_session` — no arguments; creates the per-session temp dir that later
    calls write into.
  - `/preprocess_image(image)` — returns the cutout, which gradio_client downloads
    to a local path. Carries no `@spaces.GPU` decorator, so it costs no quota.
  - `/image_to_3d(image, seed, resolution, …12 sampler params…)` — only `image` is
    required; the rest default to the UI values (guidance 7.5/7.5/1.0, 12 steps
    each). Runs `pipeline.run(..., preprocess_image=False, return_latent=True)` and
    parks the latents in a `gr.State`, which gradio hides from the API — that's why
    the client never handles them.
  - `/extract_glb(decimation_target=300000, texture_size=2048)` — returns two
    filepaths (the GLB and its download button), so take `[0]`.
- Because `/image_to_3d` hardcodes `preprocess_image=False`, calling
  `/preprocess_image` first is what makes our output match the demo UI. Skipping
  it silently feeds raw, uncropped photos to the model.
- `is_available()` constructs the client and checks that all three endpoint names
  appear in `view_api(return_format="dict")`, so drift surfaces during backend
  selection instead of mid-batch. That is a real network call and `Client(...)`
  takes no short timeout, so an unreachable host makes `trellis-forge backends`
  slow rather than failing fast.
- `HF_TOKEN` is read from the environment and passed as `Client(..., token=...)`;
  gradio_client does not pick it up on its own.
- Live-verified on 2026-10-04: the endpoint probe and the `/preprocess_image`
  round-trip, which returns a local path to a square-cropped, premultiplied RGB
  image. The two `@spaces.GPU` steps have **not** been executed from this
  adapter — anonymous ZeroGPU quota ran out first ("120s requested vs. 179s
  left", ~24 h reset), so their argument shapes come from the live API schema
  rather than from a successful run.

## Adding a backend

Implement `trellis_forge.backends.base.Backend` (two methods: `is_available`,
`generate` writing a GLB), register it in `backends/__init__.py`, and keep all
heavy imports inside methods so machines without that stack can still run the
rest of the tool.
