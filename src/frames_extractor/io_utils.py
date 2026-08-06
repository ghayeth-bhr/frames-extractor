"""Video reading, frame save/load, and timestamp helpers.

Per-frame timestamps are read via CAP_PROP_POS_MSEC rather than derived from
frame_index / nominal fps, since CCTV exports are prone to variable frame
rate and dropped frames -- index-based timestamps can silently drift over a
long clip. See check_duration_sanity for the corresponding sanity check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    nominal_duration_sec: float


@dataclass(frozen=True, kw_only=True)
class DecodedFrame:
    frame_index: int
    timestamp_ms: float
    image: np.ndarray


def open_video(video_path: Path) -> cv2.VideoCapture:
    """Open a video file for reading.

    Forces the FFMPEG backend -- on Windows, cv2.VideoCapture can silently
    fall back to MSMF, whose CAP_PROP_POS_MSEC reporting is unreliable.
    """
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    return cap


def get_video_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nominal_duration_sec = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        nominal_duration_sec=nominal_duration_sec,
    )


def iter_frames(cap: cv2.VideoCapture) -> Iterator[DecodedFrame]:
    """Yield decoded frames one at a time -- does not load the whole video into memory."""
    index = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        yield DecodedFrame(frame_index=index, timestamp_ms=timestamp_ms, image=image)
        index += 1


def check_duration_sanity(
    metadata: VideoMetadata,
    frames_decoded: int,
    last_timestamp_ms: float,
    *,
    rel_tolerance: float = 0.05,
    abs_tolerance_sec: float = 5.0,
) -> None:
    """Warn if the video's nominal duration (frame_count / fps) diverges from
    the actual last-decoded-frame timestamp -- a sign of variable frame rate
    or dropped frames that would otherwise silently corrupt timestamp-based
    scoring downstream. Never raises.
    """
    actual_duration_sec = last_timestamp_ms / 1000.0
    nominal_duration_sec = metadata.nominal_duration_sec
    tolerance_sec = max(abs_tolerance_sec, rel_tolerance * nominal_duration_sec)

    if abs(actual_duration_sec - nominal_duration_sec) > tolerance_sec:
        logger.warning(
            "Video duration mismatch: nominal %.1fs (frame_count=%d / fps=%.3f) vs "
            "actual last-frame timestamp %.1fs (decoded %d frames). This can indicate "
            "variable frame rate or dropped frames -- timestamp-based recall/precision "
            "scoring may be affected.",
            nominal_duration_sec,
            metadata.frame_count,
            metadata.fps,
            actual_duration_sec,
            frames_decoded,
        )


def save_frame_image(image: np.ndarray, out_dir: Path, frame_index: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_{frame_index:06d}.jpg"
    cv2.imwrite(str(path), image)
    return path


def load_frame_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise IOError(f"Could not read frame image: {path}")
    return image
