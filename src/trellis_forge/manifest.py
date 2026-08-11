from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


class Manifest:
    """Provenance + resume ledger for an output directory.

    Keyed by source-image sha256 so re-running a batch skips what already
    succeeded, and every shipped asset can be traced to the exact input,
    backend, seed and resolution that produced it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict = {
            "tool": "trellis-forge",
            "version": __version__,
            "created": _now(),
            "entries": {},
        }

    @classmethod
    def load(cls, path: Path) -> Manifest:
        manifest = cls(path)
        if path.exists():
            try:
                manifest.data = json.loads(path.read_text())
                manifest.data.setdefault("entries", {})
            except (json.JSONDecodeError, OSError):
                # Corrupt manifest: keep generating, start a fresh ledger.
                manifest.data["entries"] = {}
        return manifest

    def has(self, source_hash: str) -> bool:
        return source_hash in self.data["entries"]

    def add(self, source_hash: str, entry: dict) -> None:
        entry["generated_at"] = _now()
        self.data["entries"][source_hash] = entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
