from __future__ import annotations

from pathlib import Path

from trellis_forge.manifest import Manifest


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = Manifest.load(path)
    assert not manifest.has("abc")

    manifest.add("abc", {"source": "photo.png", "backend": "fake"})
    manifest.save()

    reloaded = Manifest.load(path)
    assert reloaded.has("abc")
    assert reloaded.data["entries"]["abc"]["backend"] == "fake"
    assert "generated_at" in reloaded.data["entries"]["abc"]


def test_corrupt_manifest_recovers(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not json")
    manifest = Manifest.load(path)
    assert manifest.data["entries"] == {}
