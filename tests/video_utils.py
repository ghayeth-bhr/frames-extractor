"""Synthetic video generation for stage 1 tests -- no real Absar footage required."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_synthetic_video(
    path: Path,
    *,
    fps: float = 20.0,
    duration_sec: float = 6.0,
    width: int = 160,
    height: int = 120,
    motion_window: tuple[int, int] | None = None,
    clock_corner: bool = False,
    realistic_clock_corner: bool = False,
    wide_overlay_bar: tuple[int, int, int, int] | None = None,
    wide_overlay_ticking_subregion: tuple[int, int, int, int] | None = None,
) -> int:
    """Write a synthetic MJPG/AVI clip and return the frame count written.

    - Background is a solid mid-gray frame.
    - If motion_window=(start_frame, end_frame) is given, a filled white
      rectangle sweeps left-to-right across the frame during that window.
    - If clock_corner is True, a small top-left region alternates black/white
      every frame -- an unrealistic-but-simple clock stand-in (changes far
      faster than any real clock, useful for some existing tests, but NOT a
      valid fixture for validating the ~1Hz-sampling overlay auto-detector:
      sampling at a step that's a multiple of this 2-frame toggle period
      aliases and sees no change at all).
    - If realistic_clock_corner is True, the same top-left region instead
      changes to a new value once per REAL SECOND (cycling through 10
      states) -- matches how an actual ticking clock overlay behaves, and
      is what the auto-detector should be validated against.
    - wide_overlay_bar=(x,y,w,h), if given, draws a wide static region (a
      fixed mid-gray shade, never changes) -- simulates the visible bounding
      box of a real burned-in timestamp string.
    - wide_overlay_ticking_subregion=(x,y,w,h), if given, draws a small
      region WITHIN (or overlapping) the wide bar that changes once per real
      second like realistic_clock_corner -- simulates the seconds digits
      within a wider, mostly-static timestamp string. Used together, these
      two reproduce the real 7min.mp4 clip's shape (a wide visible overlay
      where only a small sub-area actually ticks) for validating that
      masking only the detected sub-region is as effective as masking the
      whole visible bar.
    """
    total_frames = int(round(fps * duration_sec))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {path}")

    rect_w, rect_h = 20, 20
    # 15x15=225px gives a ~2x margin above Stage1Config's default
    # min_blob_area_ratio=0.0054 (~104px at this 160x120/19200px frame area) --
    # big enough that masking, not the blob-area gate, is what suppresses this
    # fixture's false motion in tests that assert masking matters.
    clock_size = 15

    try:
        for frame_index in range(total_frames):
            frame = np.full((height, width, 3), 128, dtype=np.uint8)

            if motion_window is not None:
                start, end = motion_window
                if start <= frame_index < end:
                    progress = (frame_index - start) / max(1, end - start - 1)
                    x = int(progress * (width - rect_w))
                    top_left = (x, height // 2 - rect_h // 2)
                    bottom_right = (x + rect_w, height // 2 + rect_h // 2)
                    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)

            if clock_corner:
                value = 255 if frame_index % 2 == 0 else 0
                frame[0:clock_size, 0:clock_size] = value

            if realistic_clock_corner:
                second = int(frame_index / fps) % 10
                value = int(255 * second / 9)
                frame[0:clock_size, 0:clock_size] = value

            if wide_overlay_bar is not None:
                bx, by, bw, bh = wide_overlay_bar
                frame[by : by + bh, bx : bx + bw] = 90

            if wide_overlay_ticking_subregion is not None:
                tx, ty, tw, th = wide_overlay_ticking_subregion
                second = int(frame_index / fps) % 10
                value = int(255 * second / 9)
                frame[ty : ty + th, tx : tx + tw] = value

            writer.write(frame)
    finally:
        writer.release()

    return total_frames
