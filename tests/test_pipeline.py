from __future__ import annotations

import json
from pathlib import Path

from conftest import FakeBackend

from trellis_forge.config import GenerationParams
from trellis_forge.pipeline import collect_images, run_batch


def test_collect_images_filters_and_sorts(image_folder: Path) -> None:
    images = collect_images([image_folder])
    assert [p.name for p in images] == ["molar-01.png", "molar-02.png", "premolar.jpg"]


def test_collect_images_recursive(image_folder: Path) -> None:
    nested = image_folder / "nested"
    nested.mkdir()
    from PIL import Image

    Image.new("RGB", (8, 8)).save(nested / "deep.png")
    assert len(collect_images([image_folder])) == 3
    assert len(collect_images([image_folder], recursive=True)) == 4


def test_batch_generates_assets_and_manifest(image_folder: Path, tmp_path: Path) -> None:
    out = tmp_path / "assets"
    summary = run_batch(collect_images([image_folder]), out, FakeBackend(), GenerationParams())

    assert summary.generated == 3
    assert summary.skipped == 0
    assert summary.ok

    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["entries"]) == 3
    entry = next(iter(manifest["entries"].values()))
    assert entry["backend"] == "fake"
    assert entry["qc"]["watertight"] is True
    assert entry["qc"]["faces"] == 12
    glb = Path(entry["outputs"]["glb"])
    assert glb.exists() and glb.stat().st_size > 0


def test_resume_skips_finished_images(image_folder: Path, tmp_path: Path) -> None:
    out = tmp_path / "assets"
    images = collect_images([image_folder])
    backend = FakeBackend()

    run_batch(images, out, backend, GenerationParams())
    summary = run_batch(images, out, backend, GenerationParams())

    assert summary.generated == 0
    assert summary.skipped == 3
    assert len(backend.calls) == 3  # second run hit the backend zero times


def test_force_regenerates(image_folder: Path, tmp_path: Path) -> None:
    out = tmp_path / "assets"
    images = collect_images([image_folder])
    run_batch(images, out, FakeBackend(), GenerationParams())
    summary = run_batch(images, out, FakeBackend(), GenerationParams(), force=True)
    assert summary.generated == 3


def test_failure_isolation(image_folder: Path, tmp_path: Path) -> None:
    out = tmp_path / "assets"
    backend = FakeBackend(fail_on="molar-02")
    summary = run_batch(collect_images([image_folder]), out, backend, GenerationParams())

    assert summary.generated == 2
    assert not summary.ok
    [(name, reason)] = summary.failed
    assert name == "molar-02.png"
    assert reason == "boom"


def test_limit(image_folder: Path, tmp_path: Path) -> None:
    summary = run_batch(
        collect_images([image_folder]), tmp_path / "assets", FakeBackend(),
        GenerationParams(), limit=1,
    )
    assert summary.generated == 1


def test_invalid_params_rejected(image_folder: Path, tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError):
        run_batch(
            collect_images([image_folder]), tmp_path / "assets", FakeBackend(),
            GenerationParams(resolution=999),
        )
