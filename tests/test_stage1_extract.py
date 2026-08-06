"""Tests for stage1_extract.py against a synthetic video.

This is a smoke test proving the extraction mechanics (motion recall, floor
sampling, clock-overlay masking) work, NOT real-footage threshold tuning --
that requires a real Absar clip, which isn't available yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frames_extractor.models import load_candidates
from frames_extractor.stage1_extract import Stage1Config, extract
from video_utils import make_synthetic_video

FPS = 20.0
DURATION_SEC = 6.0
WIDTH, HEIGHT = 160, 120
MOTION_WINDOW = (60, 80)  # frames 60-79 (~3.0s-4.0s): sweeping rectangle
CLOCK_REGION = (0, 0, 10, 10)
FLOOR_INTERVAL_SEC = 1.0


def _make_clip(tmp_path: Path) -> Path:
    video_path = tmp_path / "clip.avi"
    make_synthetic_video(
        video_path,
        fps=FPS,
        duration_sec=DURATION_SEC,
        width=WIDTH,
        height=HEIGHT,
        motion_window=MOTION_WINDOW,
        clock_corner=True,
    )
    return video_path


def _static_region_candidates(candidates, motion_window=MOTION_WINDOW):
    start, end = motion_window
    return [c for c in candidates if c.frame_index < start or c.frame_index >= end]


def test_extract_returns_nonempty_candidates(tmp_path: Path):
    video_path = _make_clip(tmp_path)
    config = Stage1Config(floor_interval_sec=FLOOR_INTERVAL_SEC, mask_regions=[CLOCK_REGION])

    candidates = extract(video_path, tmp_path / "out", config)

    assert len(candidates) > 0
    assert (tmp_path / "out" / "candidates.json").exists()


def test_motion_detected_inside_sweep_window(tmp_path: Path):
    video_path = _make_clip(tmp_path)
    config = Stage1Config(floor_interval_sec=FLOOR_INTERVAL_SEC, mask_regions=[CLOCK_REGION])

    candidates = extract(video_path, tmp_path / "out", config)

    start, end = MOTION_WINDOW
    motion_hits_in_window = [
        c for c in candidates if c.reason == "motion" and start <= c.frame_index < end
    ]
    # Must actually exercise MOG2, not just coincidentally overlap a floor sample.
    assert len(motion_hits_in_window) > 0


def test_floor_sampling_covers_static_region(tmp_path: Path):
    video_path = _make_clip(tmp_path)
    config = Stage1Config(floor_interval_sec=FLOOR_INTERVAL_SEC, mask_regions=[CLOCK_REGION])

    candidates = extract(video_path, tmp_path / "out", config)

    static_candidates = _static_region_candidates(candidates)
    floor_hits = [c for c in static_candidates if c.reason == "floor"]

    static_duration_sec = DURATION_SEC - (MOTION_WINDOW[1] - MOTION_WINDOW[0]) / FPS
    expected_floor_count = static_duration_sec / FLOOR_INTERVAL_SEC

    assert len(floor_hits) > 0
    assert len(floor_hits) >= expected_floor_count - 2


def test_clock_overlay_masking_suppresses_false_motion(tmp_path: Path):
    video_path = _make_clip(tmp_path)

    masked_config = Stage1Config(floor_interval_sec=FLOOR_INTERVAL_SEC, mask_regions=[CLOCK_REGION])
    unmasked_config = Stage1Config(floor_interval_sec=FLOOR_INTERVAL_SEC, mask_regions=[])

    masked_candidates = extract(video_path, tmp_path / "masked", masked_config)
    unmasked_candidates = extract(video_path, tmp_path / "unmasked", unmasked_config)

    def static_motion_count(candidates):
        return len([c for c in _static_region_candidates(candidates) if c.reason == "motion"])

    masked_static_motion = static_motion_count(masked_candidates)
    unmasked_static_motion = static_motion_count(unmasked_candidates)

    # Unmasked run should be flooded with false "motion" from the ticking
    # clock corner in the static region; masking it should suppress that.
    assert unmasked_static_motion > masked_static_motion
    assert masked_static_motion <= 3  # allow a little MOG2 warm-up noise


def test_candidate_timestamps_monotonic_and_track_fps(tmp_path: Path):
    video_path = _make_clip(tmp_path)
    config = Stage1Config(floor_interval_sec=FLOOR_INTERVAL_SEC, mask_regions=[CLOCK_REGION])

    candidates = extract(video_path, tmp_path / "out", config)
    candidates_sorted = sorted(candidates, key=lambda c: c.frame_index)

    timestamps = [c.timestamp_ms for c in candidates_sorted]
    assert all(b >= a for a, b in zip(timestamps, timestamps[1:]))

    for c in candidates_sorted:
        expected_ms = (c.frame_index / FPS) * 1000.0
        assert c.timestamp_ms == pytest.approx(expected_ms, abs=100.0)


def test_manifest_round_trips_through_json(tmp_path: Path):
    video_path = _make_clip(tmp_path)
    config = Stage1Config(floor_interval_sec=FLOOR_INTERVAL_SEC, mask_regions=[CLOCK_REGION])
    out_dir = tmp_path / "out"

    candidates = extract(video_path, out_dir, config)
    loaded = load_candidates(out_dir / "candidates.json")

    assert len(loaded) == len(candidates)
    assert {c.frame_index for c in loaded} == {c.frame_index for c in candidates}
