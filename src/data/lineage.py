"""Helpers for Bronze/Silver/Gold data artifact persistence and lineage."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

_ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "data" / "artifacts"
_MANIFEST_PATH = _ARTIFACT_ROOT / "manifest.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def schema_hash(df: pd.DataFrame) -> str:
    payload = [f"{c}:{str(t)}" for c, t in zip(df.columns.tolist(), df.dtypes.tolist())]
    return hashlib.sha256("|".join(payload).encode("utf-8")).hexdigest()


def dataframe_hash(df: pd.DataFrame) -> str:
    cols_sorted = sorted(df.columns.tolist())
    hasher = hashlib.sha256()
    hasher.update(f"shape={df.shape}".encode())
    hasher.update(",".join(cols_sorted).encode())
    hasher.update(str(df.dtypes.tolist()).encode())
    sample_size = min(1000, len(df))
    if sample_size:
        hasher.update(df.head(sample_size).to_csv(index=False).encode())
        hasher.update(df.tail(sample_size).to_csv(index=False).encode())
    return hasher.hexdigest()


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def build_artifact_id(layer: str, table: str, run_id: str, fingerprint: str) -> str:
    return f"{_safe_token(layer)}_{_safe_token(table)}_{_safe_token(run_id)}_{fingerprint[:12]}"


def persist_dataframe_artifact(
    df: pd.DataFrame,
    *,
    layer: str,
    table: str,
    run_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    parent_artifact_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Persist parquet + metadata JSON for a dataframe artifact."""
    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    layer_dir = _ARTIFACT_ROOT / layer
    layer_dir.mkdir(parents=True, exist_ok=True)

    content_hash = dataframe_hash(df)
    artifact_id = build_artifact_id(layer, table, run_id, content_hash)
    base_name = f"{artifact_id}"
    parquet_path = layer_dir / f"{base_name}.parquet"
    metadata_path = layer_dir / f"{base_name}.metadata.json"

    df.to_parquet(parquet_path, index=False)

    payload: Dict[str, Any] = {
        "artifact_id": artifact_id,
        "layer": layer,
        "table": table,
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "schema_hash": schema_hash(df),
        "dataset_hash": content_hash,
        "parent_artifact_ids": sorted(set(parent_artifact_ids or [])),
        "parquet_path": str(parquet_path.relative_to(_ARTIFACT_ROOT.parent.parent)),
    }
    if metadata:
        payload.update(metadata)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    with open(_MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")

    return payload


def set_artifact_id(df: pd.DataFrame, artifact_id: str) -> pd.DataFrame:
    df.attrs["artifact_id"] = artifact_id
    return df


def get_artifact_id(df: pd.DataFrame) -> str:
    return str(df.attrs.get("artifact_id", ""))


def find_artifact_ids(
    *,
    layer: Optional[str] = None,
    source: Optional[str] = None,
    seasons: Optional[Iterable[int]] = None,
) -> List[str]:
    if not _MANIFEST_PATH.exists():
        return []
    wanted = set(int(s) for s in seasons) if seasons is not None else None
    ids: List[str] = []
    with open(_MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if layer and rec.get("layer") != layer:
                continue
            if source and rec.get("source") != source:
                continue
            if wanted is not None:
                rec_seasons = set(int(s) for s in rec.get("seasons", []))
                if not (wanted & rec_seasons):
                    continue
            ids.append(str(rec.get("artifact_id", "")))
    return sorted(set(i for i in ids if i))
