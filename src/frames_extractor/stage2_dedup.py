"""Stage 2 -- perceptual-hash dedup via a sliding window of recently-kept frames.

Near-duplicates are temporally local, so each candidate is only compared
against a window of the most recently *kept* candidates (not the whole
corpus, and not raw candidates already dropped as duplicates) -- avoids
O(n^2) cost without needing a BK-tree at this scale. Recall-biased: the
default Hamming threshold only collapses very close matches (see SPEC.md).
"""

from __future__ import annotations

import dataclasses
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from . import io_utils, models
from .models import Candidate


@dataclass(kw_only=True)
class Stage2Config:
    hamming_threshold: int = 8
    window_size: int = 5
    hash_size: int = 8  # imagehash.phash hash_size=8 -> 8x8 = 64-bit hash, per SPEC
    max_window_ms: float = 5000.0  # ~one default floor-sample interval; bounds "temporally local"


def compute_phash(image: np.ndarray, hash_size: int = 8) -> imagehash.ImageHash:
    """image is BGR, as returned by io_utils.load_frame_image.

    imagehash.phash converts to grayscale internally via PIL's .convert("L"),
    which assumes RGB channel order for its luma weights -- feeding it a raw
    BGR array would silently use the wrong per-channel weights.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb), hash_size=hash_size)


def dedup(in_dir: Path, out_dir: Path, config: Stage2Config | None = None) -> list[Candidate]:
    config = config or Stage2Config()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        models.load_candidates(in_dir / "candidates.json"), key=lambda c: c.frame_index
    )

    kept: list[Candidate] = []
    window: deque[tuple[float, imagehash.ImageHash]] = deque(maxlen=config.window_size)

    for candidate in candidates:
        image = io_utils.load_frame_image(candidate.image_path)
        phash = compute_phash(image, hash_size=config.hash_size)

        cutoff_ms = candidate.timestamp_ms - config.max_window_ms
        is_duplicate = any(
            phash - h <= config.hamming_threshold for ts, h in window if ts >= cutoff_ms
        )
        if is_duplicate:
            continue  # near-duplicate of a recently-kept, temporally-local frame

        dest_path = out_dir / candidate.image_path.name
        shutil.copy2(candidate.image_path, dest_path)
        kept.append(dataclasses.replace(candidate, image_path=dest_path))
        window.append((candidate.timestamp_ms, phash))

    models.save_candidates(kept, out_dir / "candidates.json")
    return kept
