"""Durable run-state primitives: atomic writes, the per-run lock, UTC moments."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_migrate.core.moments import utc_moment
from llm_migrate.core.runstate import (
    LOCK_FILENAME,
    RunStateLockTimeout,
    atomic_write_text,
    run_state_lock,
)


def test_atomic_write_replaces_content_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "state.yaml"
    atomic_write_text(target, "first: 1\n")
    atomic_write_text(target, "second: 2\n")
    assert target.read_text(encoding="utf-8") == "second: 2\n"
    leftovers = [path for path in target.parent.iterdir() if path != target]
    assert leftovers == []


def test_run_state_lock_serializes_read_modify_write(tmp_path: Path) -> None:
    counter = tmp_path / "counter.txt"
    atomic_write_text(counter, "0")
    iterations, workers = 25, 4

    def bump() -> None:
        for _ in range(iterations):
            with run_state_lock(tmp_path):
                value = int(counter.read_text(encoding="utf-8"))
                atomic_write_text(counter, str(value + 1))

    threads = [threading.Thread(target=bump) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert counter.read_text(encoding="utf-8") == str(iterations * workers)


def test_run_state_lock_times_out_when_held_elsewhere(tmp_path: Path) -> None:
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with run_state_lock(tmp_path):
            held.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(timeout=5)
        with (
            pytest.raises(RunStateLockTimeout, match="another llm-migrate operation"),
            run_state_lock(tmp_path, timeout=0.2),
        ):
            pass  # pragma: no cover - the lock must not be acquired
    finally:
        release.set()
        holder.join()
    assert (tmp_path / LOCK_FILENAME).exists()


def test_utc_moment_normalizes_naive_and_aware_timestamps() -> None:
    assert utc_moment(None) is None
    naive = utc_moment("2026-09-22T10:00:00")
    assert naive == datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
    offset = utc_moment("2026-09-22T12:00:00+02:00")
    assert offset == datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
    aware = datetime(2026, 9, 22, 5, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert utc_moment(aware) == datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
    already_naive = utc_moment(datetime(2026, 9, 22, 10, 0))
    assert already_naive == datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
