"""Shared dataclasses passed between pipeline stages, persisted as JSON manifests."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(kw_only=True)
class Frame:
    frame_index: int
    timestamp_ms: float
    image_path: Path

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["image_path"] = str(self.image_path)
        return d


@dataclass(kw_only=True)
class Candidate(Frame):
    reason: Literal["motion", "floor", "fixed_fps"]
    motion_score: float | None = None
    similarity_score: float | None = None  # populated by stage 3; None from stages 1-2

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        d = dict(d)
        d["image_path"] = Path(d["image_path"])
        return cls(**d)


@dataclass(kw_only=True)
class IndexedFrame(Candidate):
    """A Candidate plus source_id, for the webapp's multi-source combined
    index (see multi_source_index.py) -- frame_index/timestamp_ms alone
    aren't globally unique across multiple source videos, so a search
    result needs source_id to be traced back to its original video."""

    source_id: str
    # Hex-encoded perceptual hash (imagehash.hex_to_hash to decode), used by
    # add_to_index()'s opt-in cross_source_dedup to detect the same
    # real-world moment re-appearing in a different uploaded source. Only
    # populated when that source was itself added with cross_source_dedup=
    # True -- None for every frame indexed before this field existed, or
    # added without opting in.
    phash: str | None = None


@dataclass(kw_only=True)
class VerifiedFrame(Candidate):
    verdict: Literal["yes", "no", "error", "skipped"]  # "error" = local failure, not a real judgment;
    # "skipped" = no call was ever attempted (--skip-vlm)
    reasoning: str
    confidence: float | None = None  # None when verdict in ("error", "skipped")


@dataclass(kw_only=True)
class ReviewDecision(VerifiedFrame):
    decision: Literal["keep", "discard"]  # no "undecided" -- skipped/unreached
    # frames simply get no entry at all


def save_candidates(candidates: list[Candidate], path: Path) -> None:
    path.write_text(json.dumps([c.to_dict() for c in candidates], indent=2))


def load_candidates(path: Path) -> list[Candidate]:
    return [Candidate.from_dict(d) for d in json.loads(path.read_text())]


def load_verified_frames(path: Path) -> list[VerifiedFrame]:
    return [VerifiedFrame.from_dict(d) for d in json.loads(path.read_text())]


def load_review_decisions(path: Path) -> list[ReviewDecision]:
    return [ReviewDecision.from_dict(d) for d in json.loads(path.read_text())]


def save_indexed_frames(frames: list[IndexedFrame], path: Path) -> None:
    path.write_text(json.dumps([f.to_dict() for f in frames], indent=2))


def load_indexed_frames(path: Path) -> list[IndexedFrame]:
    return [IndexedFrame.from_dict(d) for d in json.loads(path.read_text())]
