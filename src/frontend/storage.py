"""Asset/source storage layout and meta.json persistence for the webapp.

data/assets/<asset_id>/
    meta.json                       -- AssetMeta (name, created_at, sources)
    sources/<source_id>/
        video<ext>                  -- uploaded raw file, original extension preserved
        stage1/                     -- stage1_extract.extract() output, unchanged
        stage2/                     -- stage2_dedup.dedup() output, unchanged
    index/
        candidates.json             -- combined IndexedFrame list across ALL sources
        embeddings.npy              -- combined embeddings, row-aligned

meta.json reads/writes are NOT file-locked: only one background worker
thread ever processes uploads at a time (see worker.py), and GET endpoints
only ever read -- acceptable for a single-user local tool, not safe if this
were ever exposed to concurrent multi-process writers.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

ASSETS_ROOT = Path("data/assets")

SourceStatus = Literal["queued", "extracting", "deduping", "embedding", "ready", "error"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass(kw_only=True)
class SourceMeta:
    source_id: str
    filename: str
    status: SourceStatus
    added_at: str
    error_message: str | None = None
    candidate_count: int | None = None
    # Fully-resolved Stage1Config/Stage2Config used for this source's
    # processing (see cli.py's run_metadata.json precedent) -- recorded at
    # upload time so it's always possible to see after the fact what
    # settings a given upload was actually processed with, not just
    # whatever the current default happens to be.
    settings: dict | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "SourceMeta":
        return cls(**d)


@dataclass(kw_only=True)
class AssetMeta:
    asset_id: str
    name: str
    created_at: str
    sources: list[SourceMeta] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "created_at": self.created_at,
            "sources": [s.__dict__ for s in self.sources],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssetMeta":
        return cls(
            asset_id=d["asset_id"],
            name=d["name"],
            created_at=d["created_at"],
            sources=[SourceMeta.from_dict(s) for s in d.get("sources", [])],
        )


def asset_dir(asset_id: str) -> Path:
    return ASSETS_ROOT / asset_id


def meta_path(asset_id: str) -> Path:
    return asset_dir(asset_id) / "meta.json"


def source_dir(asset_id: str, source_id: str) -> Path:
    return asset_dir(asset_id) / "sources" / source_id


def source_stage1_dir(asset_id: str, source_id: str) -> Path:
    return source_dir(asset_id, source_id) / "stage1"


def source_stage2_dir(asset_id: str, source_id: str) -> Path:
    return source_dir(asset_id, source_id) / "stage2"


def index_dir(asset_id: str) -> Path:
    return asset_dir(asset_id) / "index"


def asset_exists(asset_id: str) -> bool:
    return meta_path(asset_id).exists()


def load_asset_meta(asset_id: str) -> AssetMeta:
    path = meta_path(asset_id)
    if not path.exists():
        raise FileNotFoundError(f"no asset {asset_id!r} at {path}")
    return AssetMeta.from_dict(json.loads(path.read_text()))


def save_asset_meta(meta: AssetMeta) -> None:
    path = meta_path(meta.asset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.to_dict(), indent=2))


def list_assets() -> list[AssetMeta]:
    if not ASSETS_ROOT.exists():
        return []
    metas = []
    for child in sorted(ASSETS_ROOT.iterdir()):
        if (child / "meta.json").exists():
            metas.append(load_asset_meta(child.name))
    return metas


def create_asset(name: str) -> AssetMeta:
    meta = AssetMeta(asset_id=new_id(), name=name, created_at=_now_iso(), sources=[])
    save_asset_meta(meta)
    return meta


def add_source(asset_id: str, filename: str, settings: dict | None = None) -> SourceMeta:
    meta = load_asset_meta(asset_id)
    source = SourceMeta(
        source_id=new_id(), filename=filename, status="queued", added_at=_now_iso(), settings=settings
    )
    meta.sources.append(source)
    save_asset_meta(meta)
    return source


def update_source(
    asset_id: str,
    source_id: str,
    *,
    status: SourceStatus | None = None,
    error_message: str | None = None,
    candidate_count: int | None = None,
) -> None:
    meta = load_asset_meta(asset_id)
    for source in meta.sources:
        if source.source_id == source_id:
            if status is not None:
                source.status = status
            if error_message is not None:
                source.error_message = error_message
            if candidate_count is not None:
                source.candidate_count = candidate_count
            break
    else:
        raise FileNotFoundError(f"no source {source_id!r} in asset {asset_id!r}")
    save_asset_meta(meta)


def get_source(asset_id: str, source_id: str) -> SourceMeta:
    meta = load_asset_meta(asset_id)
    for source in meta.sources:
        if source.source_id == source_id:
            return source
    raise FileNotFoundError(f"no source {source_id!r} in asset {asset_id!r}")


_TERMINAL_STATUSES = ("ready", "error")

ORPHANED_ERROR_MESSAGE = "Interrupted by server restart -- re-upload to retry"


def mark_orphaned_sources_as_error() -> list[tuple[str, str]]:
    """Call once at server startup. A source's status is just a string
    persisted to disk -- nothing re-submits an in-flight job on startup, so
    any source left in a non-terminal status is orphaned (whatever process
    was running it is gone) and would otherwise sit there forever looking
    like it's still progressing. Returns the (asset_id, source_id) pairs
    that got flagged, for a startup log line."""
    recovered = []
    for meta in list_assets():
        changed = False
        for source in meta.sources:
            if source.status not in _TERMINAL_STATUSES:
                source.status = "error"
                source.error_message = ORPHANED_ERROR_MESSAGE
                changed = True
                recovered.append((meta.asset_id, source.source_id))
        if changed:
            save_asset_meta(meta)
    return recovered


def reset_source_for_retry(asset_id: str, source_id: str) -> None:
    """Unlike update_source(), explicitly clears error_message/candidate_count
    back to None rather than leaving them stale -- update_source()'s "only
    overwrite a field if the given value isn't None" pattern can't do that,
    intentionally, for its other callers."""
    meta = load_asset_meta(asset_id)
    for source in meta.sources:
        if source.source_id == source_id:
            source.status = "queued"
            source.error_message = None
            source.candidate_count = None
            break
    else:
        raise FileNotFoundError(f"no source {source_id!r} in asset {asset_id!r}")
    save_asset_meta(meta)
