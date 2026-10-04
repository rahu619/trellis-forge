from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .backends.base import Backend
from .config import GenerationParams
from .export import export_formats
from .manifest import Manifest
from .preprocess import load_image
from .qc import inspect

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Called after each image with outcome "generated" | "skipped" | "failed".
OnImage = Callable[[Path, str], None]


@dataclass
class BatchSummary:
    generated: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(paths: list[Path], recursive: bool = False) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            found.extend(p for p in iterator if p.is_file() and _is_image(p))
        elif path.is_file() and _is_image(path):
            found.append(path)
    return sorted(set(found))


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def image_seed(base_seed: int, source_hash: str) -> int:
    """Combine the base seed with the image's hash so every image in a batch
    gets its own seed, and a rerun draws exactly the same samples per image."""
    return (base_seed + int(source_hash[:8], 16)) % 2**31


def run_batch(
    images: list[Path],
    out_dir: Path,
    backend: Backend,
    params: GenerationParams,
    force: bool = False,
    limit: int | None = None,
    on_image: OnImage | None = None,
) -> BatchSummary:
    """Generate assets for a batch. Per-image failures never stop the batch,
    and the manifest is saved after every success so runs resume cleanly."""
    params.validate()
    manifest = Manifest.load(out_dir / "manifest.json")
    summary = BatchSummary()
    candidates = images if limit is None else images[:limit]

    for image_path in candidates:
        source_hash = sha256_file(image_path)
        if manifest.has(source_hash) and not force:
            summary.skipped += 1
            if on_image:
                on_image(image_path, "skipped")
            continue

        asset_dir = out_dir / f"{image_path.stem}-{source_hash[:8]}"
        run_params = dataclasses.replace(params, seed=image_seed(params.seed, source_hash))
        started = time.monotonic()
        try:
            image = load_image(image_path)
            result = backend.generate(image, asset_dir, run_params)
            outputs = export_formats(result.glb_path, asset_dir, run_params.formats)
            qc_report = inspect(result.glb_path)
            manifest.add(
                source_hash,
                {
                    "source": str(image_path),
                    "backend": backend.name,
                    "seed": run_params.seed,
                    "resolution": run_params.resolution,
                    "elapsed_s": round(time.monotonic() - started, 1),
                    "outputs": {
                        fmt: str(path.relative_to(out_dir)) for fmt, path in outputs.items()
                    },
                    "qc": qc_report,
                    **result.meta,
                },
            )
            manifest.save()
            summary.generated += 1
            if on_image:
                on_image(image_path, "generated")
        except Exception as exc:  # noqa: BLE001 — isolation is the point
            summary.failed.append((image_path.name, str(exc)))
            if on_image:
                on_image(image_path, "failed")
    return summary
