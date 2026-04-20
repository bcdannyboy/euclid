from __future__ import annotations

from pathlib import Path

import pytest

from euclid.control_plane import FileLock, LockUnavailableError


def test_file_lock_rejects_parallel_nonblocking_acquire(tmp_path: Path) -> None:
    lock_path = tmp_path / "locks" / "run.lock"
    left = FileLock(lock_path)
    right = FileLock(lock_path)

    with left:
        with pytest.raises(LockUnavailableError, match="already locked"):
            right.acquire(blocking=False)

    with right:
        assert lock_path.exists()

