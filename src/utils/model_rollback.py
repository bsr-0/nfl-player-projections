"""Real rollback for the served weekly model artifacts.

`model_metadata.json` advertised `rollback_available: true` whenever a
previous *metadata* entry existed. Nothing was ever archived except that
metadata: `model_version_history.json` holds a list of prior training
summaries (dates, OOF metrics), not weights, and there was no artifact
archive anywhere. So the flag promised a capability that did not exist --
once a training run overwrote `model_{pos}_1w.joblib` /
`multiweek_{pos}.joblib`, the previous models were gone.

That stopped being hypothetical on 2026-09-24, when a walk-forward run
overwrote all four positions' served artifacts (GAPS.md, that date). The
only reason the QB models were recoverable was an ad-hoc manual copy taken
for an unrelated reason an hour earlier.

Snapshots are taken *before* training writes anything, which is the only
moment the outgoing weights still exist. Retention is deliberately small:
one snapshot is ~500 MB, so the default keeps 2 (the version being replaced
plus one before it) rather than mirroring the 5 versions of metadata
history, which costs nothing to keep.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

ROLLBACK_DIRNAME = "rollback"
DEFAULT_KEEP = 2

# The artifacts EnsemblePredictor.load_models() actually serves, plus the
# bookkeeping that describes them -- restoring weights without their
# metadata would recreate the 2026-09-24 mismatch in the other direction.
SERVED_GLOBS = ("model_*_1w.joblib", "multiweek_*.joblib")
BOOKKEEPING_FILES = ("model_metadata.json", "feature_version.txt")


def _served_artifacts(models_dir: Path) -> List[Path]:
    found: List[Path] = []
    for pattern in SERVED_GLOBS:
        found.extend(sorted(models_dir.glob(pattern)))
    for name in BOOKKEEPING_FILES:
        candidate = models_dir / name
        if candidate.is_file():
            found.append(candidate)
    return found


def available_rollbacks(models_dir: Path) -> List[Path]:
    """Snapshot directories, oldest first. Only complete ones are listed."""
    root = Path(models_dir) / ROLLBACK_DIRNAME
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.endswith(".partial"))


def snapshot_models(models_dir: Path, *, keep: int = DEFAULT_KEEP) -> Optional[Path]:
    """Copy the currently-served artifacts aside before they are replaced.

    Returns the snapshot directory, or None when there is nothing to
    archive (a first-ever training run). Builds into a ``.partial``
    directory and renames on completion, so an interrupted snapshot is
    never mistaken for a restorable one.
    """
    models_dir = Path(models_dir)
    artifacts = _served_artifacts(models_dir)
    if not artifacts:
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = models_dir / ROLLBACK_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f"{stamp}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    try:
        for artifact in artifacts:
            shutil.copy2(artifact, staging / artifact.name)
        final = root / stamp
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _prune(root, keep=keep)
    return final


def _prune(root: Path, *, keep: int) -> None:
    snapshots = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.endswith(".partial"))
    for stale in snapshots[:-keep] if keep > 0 else snapshots:
        shutil.rmtree(stale, ignore_errors=True)
        logger.info("pruned old model snapshot %s", stale.name)


def restore_models(version_dir: Path, models_dir: Path) -> List[Path]:
    """Copy a snapshot's artifacts back over the served ones.

    Deliberately does not snapshot the current state first: the caller is
    restoring precisely because the current state is unwanted, and doing it
    silently would evict a good snapshot under the default retention.
    """
    version_dir, models_dir = Path(version_dir), Path(models_dir)
    if not version_dir.is_dir():
        raise FileNotFoundError(f"no such snapshot: {version_dir}")
    restored = []
    for source in sorted(version_dir.iterdir()):
        if not source.is_file():
            continue
        destination = models_dir / source.name
        shutil.copy2(source, destination)
        restored.append(destination)
    if not restored:
        raise ValueError(f"snapshot {version_dir.name} contains no artifacts")
    return restored
