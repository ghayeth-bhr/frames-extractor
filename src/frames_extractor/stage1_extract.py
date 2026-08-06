"""Stage 1 -- coarse extraction via MOG2 motion detection + fixed-interval floor sampling.

Recall-biased by design: thresholds default loose, and floor sampling
guarantees output even for a clip with no detected motion (see SPEC.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import io_utils, models
from .models import Candidate


@dataclass(kw_only=True)
class Stage1Config:
    mog2_history: int = 500
    mog2_var_threshold: float = 16.0
    mog2_learning_rate: float = 0.001
    motion_area_ratio_threshold: float = 0.001
    floor_interval_sec: float = 5.0
    mask_regions: list[tuple[int, int, int, int]] = field(default_factory=list)


def apply_masks(image: np.ndarray, mask_regions: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Zero out configured regions (e.g. a burned-in clock overlay) before motion
    detection. Copies first -- must never mutate the frame that gets saved to disk.
    """
    if not mask_regions:
        return image
    masked = image.copy()
    for x, y, w, h in mask_regions:
        masked[y : y + h, x : x + w] = 0
    return masked


def extract(video_path: Path, out_dir: Path, config: Stage1Config | None = None) -> list[Candidate]:
    config = config or Stage1Config()
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = io_utils.open_video(video_path)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    try:
        metadata = io_utils.get_video_metadata(cap)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.mog2_history,
            varThreshold=config.mog2_var_threshold,
            detectShadows=True,
        )
        floor_interval_ms = config.floor_interval_sec * 1000.0
        candidates: list[Candidate] = []
        last_floor_ts_ms = float("-inf")
        frames_decoded = 0
        last_timestamp_ms = 0.0

        for decoded in io_utils.iter_frames(cap):
            frames_decoded += 1
            last_timestamp_ms = decoded.timestamp_ms
            height, width = decoded.image.shape[:2]

            masked = apply_masks(decoded.image, config.mask_regions)
            fg_mask = bg_subtractor.apply(masked, learningRate=config.mog2_learning_rate)
            # MOG2 shadow pixels are 127 (with detectShadows=True); drop them
            # before counting so shadow flicker can't inflate the motion score.
            _, fg_solid = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            fg_solid = cv2.morphologyEx(fg_solid, cv2.MORPH_OPEN, morph_kernel)
            motion_ratio = cv2.countNonZero(fg_solid) / (width * height)

            is_motion = motion_ratio > config.motion_area_ratio_threshold
            is_floor = (decoded.timestamp_ms - last_floor_ts_ms) >= floor_interval_ms

            if is_motion or is_floor:
                path = io_utils.save_frame_image(decoded.image, out_dir, decoded.frame_index)
                reason = "motion" if is_motion else "floor"
                candidates.append(
                    Candidate(
                        frame_index=decoded.frame_index,
                        timestamp_ms=decoded.timestamp_ms,
                        image_path=path,
                        reason=reason,
                        motion_score=motion_ratio if is_motion else None,
                    )
                )
                if is_floor:
                    last_floor_ts_ms = decoded.timestamp_ms
    finally:
        cap.release()

    if frames_decoded == 0:
        raise ValueError(f"No frames decoded from {video_path} -- file may be corrupt or unreadable")

    io_utils.check_duration_sanity(metadata, frames_decoded, last_timestamp_ms)
    models.save_candidates(candidates, out_dir / "candidates.json")
    return candidates
