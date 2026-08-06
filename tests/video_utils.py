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
) -> int:
    """Write a synthetic MJPG/AVI clip and return the frame count written.

    - Background is a solid mid-gray frame.
    - If motion_window=(start_frame, end_frame) is given, a filled white
      rectangle sweeps left-to-right across the frame during that window.
    - If clock_corner is True, a small top-left region alternates black/white
      every frame, simulating a burned-in ticking clock overlay.
    """
    total_frames = int(round(fps * duration_sec))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {path}")

    rect_w, rect_h = 20, 20
    clock_size = 10

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

            writer.write(frame)
    finally:
        writer.release()

    return total_frames
