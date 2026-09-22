"""Durable run-state I/O: atomic writes and a portable per-run lock.

Run workspaces are shared mutable state: host agents submit adaptations in
parallel, and every submission is a read-modify-write over ``changes.yaml``
and the decision logs. Plain ``write_text`` loses updates under concurrency
and can leave torn files on crash. This module provides the two primitives
every run-state write goes through:

- :func:`atomic_write_text` — temp file in the destination directory plus
  ``os.replace``, atomic on both POSIX and Windows.
- :func:`run_state_lock` — an OS-level advisory lock on one file inside the
  run directory (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows),
  released automatically by the OS when the process exits, so a crashed
  holder never leaves a stale lock behind.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

LOCK_FILENAME = ".llm-migrate.lock"
_LOCK_POLL_SECONDS = 0.05

if sys.platform == "win32":
    import msvcrt

    def _try_lock(handle: IO[bytes]) -> bool:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(handle: IO[bytes]) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RunStateLockTimeout(TimeoutError):
    """The per-run lock could not be acquired within the timeout."""


def atomic_write_text(path: Path | str, content: str, *, encoding: str = "utf-8") -> None:
    """Replace ``path`` with ``content`` atomically (temp file + ``os.replace``).

    Readers never observe a torn or partially written file: they see either
    the previous complete content or the new complete content.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise


@contextmanager
def run_state_lock(run_dir: Path | str, *, timeout: float = 10.0) -> Iterator[None]:
    """Serialize run-state read-modify-write cycles across processes.

    Locks ``<run_dir>/.llm-migrate.lock`` with a real OS advisory lock, so a
    crashed holder is released by the OS instead of wedging the run. Lock
    scopes must cover the read AND the write of a read-modify-write cycle;
    they are not reentrant, so callers never nest them.
    """
    lock_path = Path(run_dir) / LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle: IO[bytes] = open(lock_path, "a+b")  # noqa: SIM115 - closed after unlock below
    try:
        deadline = time.monotonic() + timeout
        while not _try_lock(handle):
            if time.monotonic() >= deadline:
                raise RunStateLockTimeout(
                    f"could not lock the run workspace {Path(run_dir)} within "
                    f"{timeout:g}s; another llm-migrate operation holds "
                    f"{lock_path.name}. Retry once it finishes."
                )
            time.sleep(_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            _unlock(handle)
    finally:
        handle.close()
