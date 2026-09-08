"""Tests for export.py: builds a fake stage-5-shaped in_dir directly (no
GPU/Ollama needed -- export's logic only depends on the manifest shape)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend import io_utils, models
from backend.export import export
from backend.models import ReviewDecision


def _make_image(width: int = 160, height: int = 120) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


def test_export_writes_roboflow_layout(tmp_path: Path):
    in_dir = tmp_path / "stage5"
    in_dir.mkdir()
    discarded_source_dir = tmp_path / "stage4"  # simulates the discarded frame's original location
    discarded_source_dir.mkdir()

    kept_path_0 = io_utils.save_frame_image(_make_image(), in_dir, frame_index=0)
    kept_path_5 = io_utils.save_frame_image(_make_image(), in_dir, frame_index=5)
    discarded_path = io_utils.save_frame_image(_make_image(), discarded_source_dir, frame_index=2)

    decisions = [
        ReviewDecision(
            frame_index=0,
            timestamp_ms=0.0,
            image_path=kept_path_0,
            reason="floor",
            verdict="yes",
            reasoning="matches",
            confidence=0.9,
            decision="keep",
        ),
        ReviewDecision(
            frame_index=2,
            timestamp_ms=400.0,
            image_path=discarded_path,  # outside in_dir -- stage 5's real discard behavior
            reason="motion",
            verdict="no",
            reasoning="does not match",
            confidence=0.8,
            decision="discard",
        ),
        ReviewDecision(
            frame_index=5,
            timestamp_ms=1000.0,
            image_path=kept_path_5,
            reason="motion",
            verdict="yes",
            reasoning="matches too",
            confidence=0.95,
            decision="keep",
        ),
    ]
    models.save_candidates(decisions, in_dir / "candidates.json")

    out_dir = tmp_path / "output"
    export(in_dir, out_dir)

    manifest_path = out_dir / "_annotations.coco.json"
    assert manifest_path.exists()
    coco = json.loads(manifest_path.read_text())

    for key in ("info", "licenses", "categories", "images", "annotations"):
        assert key in coco

    assert coco["annotations"] == []
    assert coco["categories"] == []
    assert len(coco["images"]) == 2  # discarded frame excluded

    ids = {entry["id"] for entry in coco["images"]}
    assert ids == {0, 5}  # original frame_index values, not renumbered 0/1

    for entry in coco["images"]:
        image_file = out_dir / "images" / entry["file_name"]
        assert image_file.exists()
        assert entry["width"] == 160
        assert entry["height"] == 120
        assert entry["license"] == 1

    assert len(coco["licenses"]) == 1
    assert coco["licenses"][0]["id"] == 1
