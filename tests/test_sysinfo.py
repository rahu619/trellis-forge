from __future__ import annotations

from trellis_forge import sysinfo


def test_machine_summary_is_a_nonempty_string() -> None:
    summary = sysinfo.machine_summary()
    assert isinstance(summary, str)
    assert summary.strip()


def test_total_memory_gb_is_sane() -> None:
    mem = sysinfo.total_memory_gb()
    assert mem is None or mem > 0


def test_is_low_memory_threshold(monkeypatch) -> None:
    monkeypatch.setattr(sysinfo, "total_memory_gb", lambda: 16.0)
    assert sysinfo.is_low_memory() is True

    monkeypatch.setattr(sysinfo, "total_memory_gb", lambda: 128.0)
    assert sysinfo.is_low_memory() is False


def test_is_low_memory_unknown_memory(monkeypatch) -> None:
    monkeypatch.setattr(sysinfo, "total_memory_gb", lambda: None)
    assert sysinfo.is_low_memory() is False
