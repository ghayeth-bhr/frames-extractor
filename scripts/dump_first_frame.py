"""Save frame 0 of a video as a PNG, for reading off pixel coordinates by eye
-- e.g. to find the (x, y, w, h) box for stage1_extract.py's
Stage1Config.mask_regions (a burned-in clock/timestamp overlay).

Manual-use dev tool, not pipeline code -- no tests.

Usage:
    uv run python scripts/dump_first_frame.py --video data/raw/clip1.mp4
    uv run python scripts/dump_first_frame.py --video data/raw/clip1.mp4 --out frame0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from frames_extractor import io_utils


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("frame0.png"))
    args = parser.parse_args()

    cap = io_utils.open_video(args.video)
    try:
        decoded = next(io_utils.iter_frames(cap))
    finally:
        cap.release()

    cv2.imwrite(str(args.out), decoded.image)
    height, width = decoded.image.shape[:2]
    print(f"Wrote {args.out} ({width}x{height})")


if __name__ == "__main__":
    main()
