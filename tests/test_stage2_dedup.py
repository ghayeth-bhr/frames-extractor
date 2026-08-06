"""Tests for stage2_dedup.py: synthetic-image unit tests plus a stage1->stage2
integration test proving the manifest handoff works end to end.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imagehash
import numpy as np
import pytest
from PIL import Image

from frames_extractor import io_utils, models, stage1_extract
from frames_extractor.models import Candidate
from frames_extractor.stage2_dedup import Stage2Config, compute_phash, dedup
from video_utils import make_synthetic_video


def _make_base_image(rect_x: int = 60) -> np.ndarray:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (rect_x, 40), (rect_x + 40, 80), (200, 200, 200), -1)
    return image


def _make_near_duplicate(base: np.ndarray, offset: int) -> np.ndarray:
    noisy = base.copy()
    noisy[0:5, 0:5] = (offset * 7) % 256
    return noisy


def _make_checkerboard_image(tile: int = 10) -> np.ndarray:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    for y in range(0, 120, tile * 2):
        image[y : y + tile, :, :] = 255
    for x in range(0, 160, tile * 2):
        image[:, x : x + tile, :] = 255
    return image


def _write_manifest(in_dir: Path, images: list[np.ndarray]) -> list[Candidate]:
    candidates = []
    for i, image in enumerate(images):
        path = io_utils.save_frame_image(image, in_dir, frame_index=i)
        candidates.append(
            Candidate(frame_index=i, timestamp_ms=i * 200.0, image_path=path, reason="floor")
        )
    models.save_candidates(candidates, in_dir / "candidates.json")
    return candidates


def test_near_duplicates_collapse(tmp_path: Path):
    in_dir = tmp_path / "stage1"
    in_dir.mkdir()
    base = _make_base_image()
    images = [_make_near_duplicate(base, offset=i) for i in range(5)]
    _write_manifest(in_dir, images)

    out_dir = tmp_path / "stage2"
    kept = dedup(in_dir, out_dir)

    assert 1 <= len(kept) < len(images)
    assert (out_dir / "candidates.json").exists()


def test_distinct_frames_never_merged(tmp_path: Path):
    in_dir = tmp_path / "stage1"
    in_dir.mkdir()
    # Large-scale structural differences (rectangle position, checkerboard
    # tile size) rather than just color/brightness -- pHash's low-frequency
    # DCT is most discriminative on this kind of change, and is intentionally
    # tolerant of small/uniform brightness shifts.
    images = [
        _make_base_image(rect_x=0),
        _make_checkerboard_image(tile=8),
        _make_base_image(rect_x=100),
        _make_checkerboard_image(tile=16),
        _make_base_image(rect_x=50),
        _make_checkerboard_image(tile=24),
    ]
    _write_manifest(in_dir, images)

    out_dir = tmp_path / "stage2"
    kept = dedup(in_dir, out_dir)

    assert len(kept) == len(images)


def test_phash_uses_correct_channel_order():
    # A single vertical two-tone split only varies horizontal frequency,
    # leaving most of pHash's 8x8 DCT block at zero either way (verified
    # empirically -- it under-discriminates, giving a Hamming distance of
    # only ~4). A 2x2 quadrant layout with asymmetric R/B per quadrant
    # varies both horizontal and vertical frequency and reliably flips the
    # brightness ranking of the four corners between the correct and the
    # (deliberately wrong) un-converted interpretation.
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:32, :32] = (200, 0, 0)  # BGR strong blue
    image[:32, 32:] = (0, 0, 200)  # BGR strong red
    image[32:, :32] = (0, 0, 100)  # BGR dim red
    image[32:, 32:] = (100, 0, 0)  # BGR dim blue

    correct_hash = compute_phash(image)
    # Deliberately skip the BGR->RGB conversion to reproduce the bug.
    wrong_hash = imagehash.phash(Image.fromarray(image))

    assert (correct_hash - wrong_hash) > Stage2Config().hamming_threshold


def test_stage1_to_stage2_integration(tmp_path: Path):
    video_path = tmp_path / "clip.avi"
    fps = 20.0
    duration_sec = 6.0
    motion_window = (60, 80)
    make_synthetic_video(
        video_path,
        fps=fps,
        duration_sec=duration_sec,
        width=160,
        height=120,
        motion_window=motion_window,
    )

    stage1_config = stage1_extract.Stage1Config(floor_interval_sec=0.5)
    stage1_candidates = stage1_extract.extract(video_path, tmp_path / "stage1", stage1_config)

    stage2_candidates = dedup(tmp_path / "stage1", tmp_path / "stage2")

    loaded = models.load_candidates(tmp_path / "stage2" / "candidates.json")
    assert len(loaded) == len(stage2_candidates)

    assert 0 < len(stage2_candidates) <= len(stage1_candidates)

    stage1_by_index = {c.frame_index: c for c in stage1_candidates}
    for candidate in stage2_candidates:
        original = stage1_by_index[candidate.frame_index]
        assert candidate.timestamp_ms == original.timestamp_ms
        assert candidate.reason == original.reason
        assert candidate.motion_score == original.motion_score
        assert candidate.image_path.exists()
        assert candidate.image_path.parent == tmp_path / "stage2"

    stage1_floor_count = sum(1 for c in stage1_candidates if c.reason == "floor")
    stage2_floor_count = sum(1 for c in stage2_candidates if c.reason == "floor")
    assert stage2_floor_count < stage1_floor_count
