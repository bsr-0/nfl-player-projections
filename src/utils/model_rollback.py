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

Snapshots used to hold only the weights and two bookkeeping files. The
models are trained on features passed through the bounded scaler and the
utilization weights/bounds, so restoring weights alone served them against
whatever preprocessing the later run left behind (GAPS.md 2026-09-28).
Snapshots now carry everything a production run rewrites that serving or
monitoring reads, plus a manifest; a snapshot without one predates that and
is refused by default.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

ROLLBACK_DIRNAME = "rollback"
DEFAULT_KEEP = 2
MANIFEST_NAME = "rollback_manifest.json"

# The artifacts EnsemblePredictor.load_models() actually serves.
SERVED_GLOBS = ("model_*_1w.joblib", "multiweek_*.joblib", "util_to_fp_*.joblib")
# Preprocessing and choices those models were trained with and are served
# through. Literal names; tests pin them to the constants their writers use.
PREPROCESSING_FILES = (
    "feature_scaler_bounded.joblib",
    "utilization_weights.json",
    "utilization_percentile_bounds.json",
    "snap_imputation.json",
    "qb_target_choice.json",
)
# Drift-monitoring baselines describing the served models' training labels.
MONITORING_GLOBS = ("label_baseline*.json",)
# Restoring weights without their metadata would recreate the 2026-09-24
# mismatch in the other direction.
BOOKKEEPING_FILES = ("model_metadata.json", "feature_version.txt")

COVERAGE: Tuple[str, ...] = (*SERVED_GLOBS, *PREPROCESSING_FILES, *MONITORING_GLOBS,
                             *BOOKKEEPING_FILES)


class LegacySnapshotError(ValueError):
    """A snapshot from before full coverage: weights without their preprocessing."""


def _covered(models_dir: Path, coverage=COVERAGE) -> List[Path]:
    found = {path for pattern in coverage for path in models_dir.glob(pattern)
             if path.is_file()}
    return sorted(found)


def _served_artifacts(models_dir: Path) -> List[Path]:
    return _covered(models_dir)


def covered_artifacts(models_dir: Path) -> List[Path]:
    """Every file in `models_dir` that serving or monitoring reads from a
    trained model set (the snapshot/restore coverage)."""
    return _covered(Path(models_dir))


def is_legacy_snapshot(snapshot: Path) -> bool:
    return not (Path(snapshot) / MANIFEST_NAME).is_file()


def available_rollbacks(models_dir: Path, *, include_legacy: bool = False) -> List[Path]:
    """Snapshot directories, oldest first. Only complete ones are listed, and
    legacy (manifest-less) ones only when asked for: they cannot be restored
    consistently, so they must not make rollback look available."""
    root = Path(models_dir) / ROLLBACK_DIRNAME
    if not root.is_dir():
        return []
    snapshots = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.endswith(".partial"))
    if include_legacy:
        return snapshots
    return [d for d in snapshots if not is_legacy_snapshot(d)]


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
        # Written last: a snapshot with a manifest is a complete one.
        (staging / MANIFEST_NAME).write_text(json.dumps({
            "coverage": list(COVERAGE),
            "files": [artifact.name for artifact in artifacts],
        }, indent=2))
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


def restore_models(version_dir: Path, models_dir: Path, *,
                   allow_legacy: bool = False) -> List[Path]:
    """Make the covered artifacts exactly the snapshot's.

    Copies every snapshotted file back and deletes covered files the snapshot
    did not have (e.g. a converter or snap_imputation.json a later run
    created), so nothing from the unwanted run is left serving alongside the
    restored models.

    A legacy snapshot (no manifest) holds only weights and bookkeeping;
    restoring it pairs those weights with the current preprocessing, so it
    raises LegacySnapshotError unless `allow_legacy=True` -- appropriate only
    when the preprocessing files are known unchanged since the snapshot.
    Legacy restores copy what is there and delete nothing.

    Deliberately does not snapshot the current state first: the caller is
    restoring precisely because the current state is unwanted, and doing it
    silently would evict a good snapshot under the default retention.
    """
    version_dir, models_dir = Path(version_dir), Path(models_dir)
    if not version_dir.is_dir():
        raise FileNotFoundError(f"no such snapshot: {version_dir}")
    if is_legacy_snapshot(version_dir):
        if not allow_legacy:
            raise LegacySnapshotError(
                f"snapshot {version_dir.name} predates full coverage: it has model "
                "weights but not the scaler/utilization files they were trained "
                "with, so restoring it would serve them against today's preprocessing")
        names = sorted(f.name for f in version_dir.iterdir() if f.is_file())
        coverage: Tuple[str, ...] = ()
    else:
        manifest = json.loads((version_dir / MANIFEST_NAME).read_text())
        names, coverage = list(manifest["files"]), tuple(manifest["coverage"])
    if not names:
        raise ValueError(f"snapshot {version_dir.name} contains no artifacts")
    absent = [name for name in names if not (version_dir / name).is_file()]
    if absent:
        raise ValueError(f"snapshot {version_dir.name} is missing {absent}; nothing restored")

    restored = []
    for name in names:
        destination = models_dir / name
        shutil.copy2(version_dir / name, destination)
        restored.append(destination)
    for stale in _covered(models_dir, coverage):
        if stale.name not in names:
            stale.unlink()
            logger.warning("removed %s: not part of snapshot %s", stale.name, version_dir.name)
    return restored
