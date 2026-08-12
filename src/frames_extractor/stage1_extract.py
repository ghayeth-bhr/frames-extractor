"""Stage 1 -- coarse extraction via MOG2 motion detection + fixed-interval floor sampling.

Recall-biased by design: thresholds default loose, and floor sampling
guarantees output even for a clip with no detected motion (see SPEC.md).
"""

from __future__ import annotations

from dataclasses import dataclass
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
    mask_regions: list[tuple[int, int, int, int]] | None = None  # None = not specified
    run_auto_detect: bool = False  # opt-in: auto-detect only runs if True AND mask_regions is None


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


def detect_overlay_regions(
    video_path: Path,
    sample_interval_sec: float = 1.0,
    diff_threshold: int = 25,
    fraction_threshold: float = 0.8,
    min_area: int = 20,
    max_scan_sec: float = 300.0,
) -> list[tuple[int, int, int, int]]:
    """Auto-detects a burned-in ticking timestamp/clock overlay by sampling
    frames spread across the video via seeking (cheap pre-pass, not a full
    sequential decode), diffing consecutive samples, and finding regions
    that changed on >= fraction_threshold of sampled transitions.

    A real ticking overlay changes on nearly every 1-second sample for as
    long as it's on screen; real scene motion is transient and moves, so it
    doesn't accumulate that consistently in one fixed spot. Validated
    against a real Absar clip and synthetic fixtures: on the real clip this
    only finds the fastest-ticking sub-component (e.g. seconds digits, not
    the whole visible timestamp text) -- confirmed this is correct, not a
    tuning gap (no threshold safely captures the whole text without also
    catching real scene motion), and confirmed masking just that
    sub-region is exactly as effective as masking the whole visible overlay
    (a static remainder can't cause false motion regardless of masking,
    since MOG2 only reacts to change). See SPEC.md / the approved plan for
    the full validation writeup.

    Scans at most the first `max_scan_sec` of video, not the whole clip --
    a real overlay's behavior doesn't change over time, so scanning a
    bounded prefix is just as reliable and keeps the cost bounded for
    SPEC's 60-minute upper bound (measured ~114s for a full 7-minute scan
    at these defaults; uncapped, a 60-minute clip would scale to several
    minutes just for this pre-pass).

    Returns [] if the video is empty or nothing crosses the threshold --
    never forces a false positive.
    """
    cap = io_utils.open_video(video_path)
    try:
        metadata = io_utils.get_video_metadata(cap)
        if metadata.fps <= 0 or metadata.frame_count <= 0:
            return []

        frame_step = max(1, int(round(metadata.fps * sample_interval_sec)))
        max_frame = min(metadata.frame_count, int(metadata.fps * max_scan_sec))
        sample_indices = range(0, max_frame, frame_step)

        prev_gray: np.ndarray | None = None
        change_accumulator: np.ndarray | None = None
        n_pairs = 0

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                changed = (diff > diff_threshold).astype(np.uint8)
                if change_accumulator is None:
                    change_accumulator = np.zeros_like(changed, dtype=np.float64)
                change_accumulator += changed
                n_pairs += 1
            prev_gray = gray
    finally:
        cap.release()

    if n_pairs == 0 or change_accumulator is None:
        return []

    fraction_changed = change_accumulator / n_pairs
    mask = (fraction_changed >= fraction_threshold).astype(np.uint8) * 255

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions: list[tuple[int, int, int, int]] = []
    for label in range(1, num_labels):  # label 0 is the background
        x, y, w, h, area = stats[label]
        if area >= min_area:
            regions.append((int(x), int(y), int(w), int(h)))
    return regions


def extract(video_path: Path, out_dir: Path, config: Stage1Config | None = None) -> list[Candidate]:
    config = config or Stage1Config()

    if config.mask_regions is not None:
        mask_regions = config.mask_regions  # explicit always wins, regardless of run_auto_detect
    elif config.run_auto_detect:
        detected = detect_overlay_regions(video_path)
        if detected:
            print(f"[extract] auto-detected {len(detected)} mask region(s): {detected}")
        else:
            print("[extract] auto-detection found no overlay region")
        mask_regions = detected
    else:
        mask_regions = []  # today's default, unchanged: no masking, no auto-detect

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

            masked = apply_masks(decoded.image, mask_regions)
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
