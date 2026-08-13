from __future__ import annotations

import json
from pathlib import Path

from trellis_forge.manifest import SCHEMA, Manifest


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = Manifest.load(path)
    assert not manifest.has("abc")
    assert manifest.data["schema"] == SCHEMA

    manifest.add("abc", {"source": "photo.png", "backend": "fake"})
    manifest.save()

    reloaded = Manifest.load(path)
    assert reloaded.has("abc")
    assert reloaded.data["schema"] == SCHEMA
    assert reloaded.data["entries"]["abc"]["backend"] == "fake"
    assert "generated_at" in reloaded.data["entries"]["abc"]


def test_corrupt_manifest_recovers(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not json")
    manifest = Manifest.load(path)
    assert manifest.data["entries"] == {}


def test_legacy_manifest_still_resumes(tmp_path: Path) -> None:
    # Schema 1 ledgers (no "schema" key, absolute output paths) must keep
    # working for resume after the v2 clean break.
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "tool": "trellis-forge",
                "version": "0.1.0",
                "created": "2026-01-01T00:00:00+00:00",
                "entries": {"abc": {"source": "/home/u/photo.png", "backend": "fake"}},
            }
        )
    )
    manifest = Manifest.load(path)
    assert manifest.has("abc")
    assert manifest.data["schema"] == SCHEMA
