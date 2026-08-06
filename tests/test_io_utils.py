"""Tests for io_utils.py video I/O and timestamp plumbing.

Uses a synthetic, constant-fps clip generated locally -- no real Absar
footage available yet. The timestamp/fps checks here are NOT a proof of
variable-frame-rate handling; that validation is a follow-up once a real
sample clip exists.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from frames_extractor import io_utils
from video_utils import make_synthetic_video

FPS = 20.0
DURATION_SEC = 4.0
WIDTH, HEIGHT = 160, 120


def test_open_video_and_metadata(tmp_path: Path):
    video_path = tmp_path / "clip.avi"
    expected_frames = make_synthetic_video(
        video_path, fps=FPS, duration_sec=DURATION_SEC, width=WIDTH, height=HEIGHT
    )

    cap = io_utils.open_video(video_path)
    try:
        metadata = io_utils.get_video_metadata(cap)
    finally:
        cap.release()

    assert metadata.width == WIDTH
    assert metadata.height == HEIGHT
    assert metadata.fps == pytest.approx(FPS, rel=0.05)
    assert metadata.frame_count == pytest.approx(expected_frames, abs=2)


def test_open_video_missing_file_raises(tmp_path: Path):
    with pytest.raises(IOError):
        io_utils.open_video(tmp_path / "does_not_exist.avi")


def test_iter_frames_timestamps_monotonic_and_track_fps(tmp_path: Path):
    video_path = tmp_path / "clip.avi"
    make_synthetic_video(video_path, fps=FPS, duration_sec=DURATION_SEC, width=WIDTH, height=HEIGHT)

    cap = io_utils.open_video(video_path)
    try:
        decoded_frames = list(io_utils.iter_frames(cap))
    finally:
        cap.release()

    assert len(decoded_frames) > 0

    timestamps = [f.timestamp_ms for f in decoded_frames]
    assert all(b >= a for a, b in zip(timestamps, timestamps[1:]))

    for frame in decoded_frames:
        expected_ms = (frame.frame_index / FPS) * 1000.0
        assert frame.timestamp_ms == pytest.approx(expected_ms, abs=100.0)


def test_check_duration_sanity_warns_on_mismatch(caplog: pytest.LogCaptureFixture):
    metadata = io_utils.VideoMetadata(
        fps=20.0, frame_count=200, width=160, height=120, nominal_duration_sec=10.0
    )

    with caplog.at_level(logging.WARNING, logger="frames_extractor.io_utils"):
        io_utils.check_duration_sanity(metadata, frames_decoded=200, last_timestamp_ms=2000.0)

    assert any("duration mismatch" in r.message.lower() for r in caplog.records)


def test_check_duration_sanity_silent_when_consistent(caplog: pytest.LogCaptureFixture):
    metadata = io_utils.VideoMetadata(
        fps=20.0, frame_count=200, width=160, height=120, nominal_duration_sec=10.0
    )

    with caplog.at_level(logging.WARNING, logger="frames_extractor.io_utils"):
        io_utils.check_duration_sanity(metadata, frames_decoded=200, last_timestamp_ms=9950.0)

    assert not any("duration mismatch" in r.message.lower() for r in caplog.records)


def test_save_and_load_frame_image_roundtrip(tmp_path: Path):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)

    path = io_utils.save_frame_image(image, tmp_path, frame_index=0)
    assert path.exists()

    loaded = io_utils.load_frame_image(path)
    assert loaded.shape == image.shape
