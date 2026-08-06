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
    reason: Literal["motion", "floor"]
    motion_score: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        d = dict(d)
        d["image_path"] = Path(d["image_path"])
        return cls(**d)


def save_candidates(candidates: list[Candidate], path: Path) -> None:
    path.write_text(json.dumps([c.to_dict() for c in candidates], indent=2))


def load_candidates(path: Path) -> list[Candidate]:
    return [Candidate.from_dict(d) for d in json.loads(path.read_text())]
