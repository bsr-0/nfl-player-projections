"""One atomic write path for every cache and artifact this repo persists.

Four independent copies of "write to a temp file, then rename" had grown up
in `src/models/simulation_io.py`, `src/utils/data_manager.py`,
`src/data/pbp_stats_aggregator.py`, and `scripts/generate_app_data.py`. Only
the first fsynced. The other three carried a comment promising to "prevent
corruption on crash", which `os.replace` alone does not deliver: the rename
is atomic with respect to other readers, but until the file's data blocks
reach disk a power loss can leave a correctly-named, zero-length artifact --
exactly the corruption the comment claims to rule out. Consolidating on the
fsyncing version makes that promise true everywhere.

Only the file is fsynced, not the containing directory. A directory fsync
would additionally make the *rename* durable, but it is not portable, and
its absence weakens durability rather than producing a wrong file -- the
same trade-off the strongest existing implementation already made.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


def atomic_write(path: str | Path, writer: Callable[[Path], None]) -> Path:
    """Build the file at a temp path via ``writer``, fsync it, then rename.

    ``writer`` receives the temporary path and must write the complete file
    to it. A partially written or failed temp file is always removed, so a
    raising ``writer`` can never leave a stray ``.tmp`` behind or clobber
    the existing ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Dotted prefix keeps the temp file out of glob patterns that scan these
    # directories for real artifacts while a write is in flight.
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        with open(tmp, "r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def atomic_write_json(payload: Any, path: str | Path, *, indent: int | None = 2) -> Path:
    """Atomically write ``payload`` as JSON.

    ``allow_nan=False`` deliberately: Python's default emits bare ``NaN`` /
    ``Infinity`` tokens, which are not valid JSON and which every non-Python
    reader of these artifacts (the site bundle, the browser) rejects. Failing
    at write time surfaces the bad value instead of persisting a file that
    only looks fine until something else reads it.
    """
    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=indent, allow_nan=False))
            handle.write("\n")

    return atomic_write(path, _write)


def atomic_write_parquet(df, path: str | Path, *, index: bool = False) -> Path:
    """Atomically write a DataFrame to parquet."""
    return atomic_write(path, lambda tmp: df.to_parquet(tmp, index=index))
