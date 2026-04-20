from __future__ import annotations

import fcntl
import os
from pathlib import Path


class LockUnavailableError(TimeoutError):
    pass


class FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self, *, blocking: bool = True) -> FileLock:
        if self._fd is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(self._fd, flags)
        except BlockingIOError as exc:
            os.close(self._fd)
            self._fd = None
            raise LockUnavailableError(f"{self.path} is already locked") from exc
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> FileLock:
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


__all__ = ["FileLock", "LockUnavailableError"]
