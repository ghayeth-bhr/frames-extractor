"""Stage 1 -- coarse extraction via MOG2 motion detection + fixed-interval floor sampling.

Recall-biased by design: thresholds default loose, and floor sampling
guarantees output even for a clip with no detected motion (see SPEC.md).
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import io_utils, models
from .models import Candidate


@contextlib.contextmanager
def _suppress_native_stderr():
    """Redirects the OS-level stderr file descriptor to devnull for the
    duration of the block.

    Needed because some HEVC streams make FFmpeg's decoder print benign
    "Could not find ref with POC ..." warnings directly via a C-level
    write to fd 2 whenever cv2.VideoCapture seeks
    (cap.set(CAP_PROP_POS_FRAMES, ...)) into a spot where a B-frame's
    reference isn't available yet -- a normal, harmless artifact of
    frame-based seeking, not file corruption (confirmed on a real clip
    that triggers this: the exact same frames decode with zero warnings
    via plain sequential cap.read(), the decode path used everywhere else
    in this project). Left unsuppressed, hundreds of these lines can
    scroll past during detect_overlay_regions' seek-based pre-pass scan
    and make a fast (seconds), fully successful scan look like a hang or
    crash -- this only wraps that scan, never the main sequential decode
    loop, which doesn't seek and was never observed to warn at all.
    """
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


@dataclass(kw_only=True)
class Stage1Config:
    mog2_history: int = 500
    mog2_var_threshold: float = 16.0
    mog2_learning_rate: float = 0.001
    motion_area_ratio_threshold: float = 0.001
    # Gate on the single largest connected-component area, not just total
    # flagged-area ratio -- filters scattered speckle/compression noise that
    # motion_area_ratio_threshold alone doesn't, without touching MOG2's own
    # sensitivity. Only applied to frames that already pass
    # motion_area_ratio_threshold (cheap pre-filter before the more
    # expensive connected-components call).
    #
    # CALIBRATION CAVEAT: 0.0054 (~20,000px at 2560x1440) was derived from
    # this project's only real ground truth (eval/ground_truth.json), which
    # is exclusively large-body-motion events (people entering/leaving
    # through a door) -- every one of those events' minimum in-frame blob
    # was >=31,900px, so this default keeps a >2x safety margin below the
    # smallest real event actually measured. There is NO ground truth yet
    # for small-gesture queries (e.g. "put a lid on a cup," a spill, a hand
    # reaching for something) -- this default has NOT been validated against
    # that kind of motion and may be too aggressive for it. A query
    # targeting fine-grained motion should pass a smaller value explicitly
    # (via --min-event-area-ratio), the same way a short-duration query
    # should pass a smaller --min-event-duration-sec.
    min_blob_area_ratio: float = 0.0054
    # Downscales the frame fed to MOG2/blob-detection ONLY -- the saved
    # candidate image is always the full-resolution original (see extract()).
    # Every frame is still processed in order (nothing skipped), so MOG2's
    # background model stays continuous -- this is resolution reduction, not
    # frame-skipping, which would corrupt the model (see SPEC.md's Step 6
    # chunking caveat and Step 1's warm-up-bug history for why that matters).
    #
    # VALIDATED at 0.5 (OPTIMIZATION_PLAN.md Step 4): profiling found MOG2/
    # threshold/connectedComponents, not video decode, dominates stage 1's
    # cost (77.6% vs 22.4%) -- all three scale with pixel count, so
    # downscaling before them is the real lever, not GPU-accelerated decode
    # (which was explored and found unavailable on this machine without
    # heavy setup). At 0.5, a full real-clip comparison against the
    # un-downscaled baseline found every one of the 6 known ground-truth
    # events' motion candidates identical by exact frame_index. A separate
    # boundary-focused check (frames with full-res largest_blob_ratio within
    # 0.004-0.007 of this threshold) found a real ~11% decision-flip rate
    # among those borderline frames specifically -- bounded to a narrow band
    # around the threshold, not a general degradation, and not observed to
    # change any of the 6 events' final recall outcome after the full
    # dedup->review pipeline. See run5/run_metadata.json for the full
    # validation writeup. Default kept at 1.0 (off) pending that full
    # end-to-end confirmation -- pass 0.5 explicitly once validated.
    downscale_factor: float = 1.0  # 1.0 = off, today's unchanged default
    floor_interval_sec: float = 5.0
    mask_regions: list[tuple[int, int, int, int]] | None = None  # None = not specified
    run_auto_detect: bool = False  # opt-in: auto-detect only runs if True AND mask_regions is None
    # "motion" (default, unchanged): today's MOG2 + blob-gate detection, with
    # floor_interval_sec as a recall safety net alongside it.
    # "fixed-fps": for continuous/state-based queries (e.g. "is this worker
    # wearing a mask correctly") where the condition persists regardless of
    # motion -- bypasses MOG2/blob-gate detection ENTIRELY, using the same
    # fixed-interval mechanism floor sampling already uses, but as the sole
    # candidate source rather than a safety net alongside motion detection.
    # sample_interval_sec (below) is deliberately a SEPARATE field from
    # floor_interval_sec/min_event_duration_sec -- those stay scoped to
    # motion mode's floor-sampling safety net and answer a different
    # question ("shortest event I might otherwise miss between motion
    # detections") than this one ("how densely to sample a condition that
    # doesn't trigger motion detection at all").
    #
    # VALIDATED: mechanism correctly bypasses motion detection and samples
    # at fixed intervals (confirmed via synthetic tests). NOT YET VALIDATED:
    # whether this actually catches a real sustained condition that motion
    # mode would miss, and whether stage 3's SigLIP2 ranking can meaningfully
    # distinguish a correct-vs-incorrect condition (e.g. mask worn correctly
    # vs. not) among many visually-similar fixed-fps candidates -- both are
    # open until tested against real footage.
    sampling_mode: str = "motion"
    sample_interval_sec: float = 2.0  # only consulted when sampling_mode == "fixed-fps"


def derive_floor_interval_sec(min_event_duration_sec: float) -> float:
    """Derives a floor-sampling interval from the shortest real event a
    caller cares about catching via floor sampling (not motion detection,
    which is duration-independent and already catches an event regardless
    of this interval).

    Interval = min_event_duration_sec / 2, clamped to never exceed
    Stage1Config's own proven-safe default (5.0s) -- an overly generous
    min_event_duration_sec can't silently under-sample below what's
    already validated.

    Why /2 and not exact Nyquist: an event of duration D is guaranteed to
    contain at least one floor sample if the sampling interval is
    strictly less than D -- that's the bare minimum, and it's
    phase-dependent (a sample spaced at exactly D apart can still
    straddle the event in the worst-case alignment, since floor sampling
    isn't reconstructing a periodic waveform, it's guaranteeing a hit on
    a single, one-off window at an unknown phase). Halving gives 2x
    margin below that bare-minimum threshold, so even in the worst-case
    phase alignment at least one sample lands well inside the event, not
    just barely at its edge.
    """
    return min(min_event_duration_sec / 2.0, Stage1Config.floor_interval_sec)


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

        with _suppress_native_stderr():
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


def _extract_fixed_fps(video_path: Path, out_dir: Path, config: Stage1Config) -> list[Candidate]:
    """Sole candidate source for continuous/state-based queries: every frame
    at config.sample_interval_sec becomes a candidate, with no MOG2/blob-gate
    filtering at all -- unlike motion mode's floor sampling (a safety net
    alongside motion detection), this IS the detection here. Masks/downscale
    are irrelevant (nothing is fed to MOG2), so this path skips them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = io_utils.open_video(video_path)
    sample_interval_ms = config.sample_interval_sec * 1000.0

    try:
        metadata = io_utils.get_video_metadata(cap)
        candidates: list[Candidate] = []
        last_sample_ts_ms = float("-inf")
        frames_decoded = 0
        last_timestamp_ms = 0.0

        for decoded in io_utils.iter_frames(cap):
            frames_decoded += 1
            last_timestamp_ms = decoded.timestamp_ms

            if (decoded.timestamp_ms - last_sample_ts_ms) >= sample_interval_ms:
                path = io_utils.save_frame_image(decoded.image, out_dir, decoded.frame_index)
                candidates.append(
                    Candidate(
                        frame_index=decoded.frame_index,
                        timestamp_ms=decoded.timestamp_ms,
                        image_path=path,
                        reason="fixed_fps",
                        motion_score=None,
                    )
                )
                last_sample_ts_ms = decoded.timestamp_ms
    finally:
        cap.release()

    if frames_decoded == 0:
        raise ValueError(f"No frames decoded from {video_path} -- file may be corrupt or unreadable")

    io_utils.check_duration_sanity(metadata, frames_decoded, last_timestamp_ms)
    models.save_candidates(candidates, out_dir / "candidates.json")
    return candidates


def extract(video_path: Path, out_dir: Path, config: Stage1Config | None = None) -> list[Candidate]:
    config = config or Stage1Config()

    if config.sampling_mode not in ("motion", "fixed-fps"):
        raise ValueError(f"sampling_mode must be 'motion' or 'fixed-fps', got {config.sampling_mode!r}")

    if config.sampling_mode == "fixed-fps":
        print(
            "[extract] sampling_mode=fixed-fps: bypassing motion detection entirely -- "
            f"every frame at the {config.sample_interval_sec}s interval becomes a candidate. "
            "Expect a MUCH larger candidate count reaching stage 2/3 than motion mode "
            "produces on the same clip (motion mode filters by detected activity; "
            "fixed-fps mode does not filter at all)."
        )
        return _extract_fixed_fps(video_path, out_dir, config)

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

            masked = apply_masks(decoded.image, mask_regions)
            # Downscale AFTER masking (mask coordinates are in full-resolution
            # pixels) -- detection-only, the saved candidate image below always
            # uses the original full-resolution decoded.image, never `detect_frame`.
            if config.downscale_factor != 1.0:
                orig_h, orig_w = masked.shape[:2]
                detect_frame = cv2.resize(
                    masked,
                    (int(orig_w * config.downscale_factor), int(orig_h * config.downscale_factor)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                detect_frame = masked
            height, width = detect_frame.shape[:2]

            # Passing a fixed learningRate overrides MOG2's own fast-then-decaying
            # auto schedule -- during the history-length warm-up window, let it use
            # that auto schedule (-1) so the background model actually converges,
            # instead of forcing config.mog2_learning_rate's deliberately slow rate
            # from frame 1 (which otherwise misclassifies most of warm-up as motion).
            learning_rate = -1.0 if frames_decoded <= config.mog2_history else config.mog2_learning_rate
            fg_mask = bg_subtractor.apply(detect_frame, learningRate=learning_rate)
            # MOG2 shadow pixels are 127 (with detectShadows=True); drop them
            # before counting so shadow flicker can't inflate the motion score.
            _, fg_solid = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            fg_solid = cv2.morphologyEx(fg_solid, cv2.MORPH_OPEN, morph_kernel)
            motion_ratio = cv2.countNonZero(fg_solid) / (width * height)

            is_area_candidate = motion_ratio > config.motion_area_ratio_threshold
            is_motion = False
            if is_area_candidate:
                # Only pay for connected-components on frames that already
                # cleared the cheap ratio pre-filter. Gating on the single
                # largest blob (not total flagged area) filters scattered
                # speckle/compression noise that the area ratio alone lets
                # through -- see Stage1Config.min_blob_area_ratio's docstring
                # for the calibration caveat.
                num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                    fg_solid, connectivity=8
                )
                largest_blob_area = max((stats[i, 4] for i in range(1, num_labels)), default=0)
                largest_blob_ratio = largest_blob_area / (width * height)
                is_motion = largest_blob_ratio >= config.min_blob_area_ratio
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
