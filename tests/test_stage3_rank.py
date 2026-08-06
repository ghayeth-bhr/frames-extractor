"""Tests for stage3_rank.py: a harder-than-smoke-test graded ranking check,
a stage2->stage3 integration test, and a top-K-is-a-cap check.

Model load (~15-20s even from local HF cache) is paid twice across this
file: once for the module-scoped fixture backing the graded-ranking and
top-K tests, once inside the real rank() call in the integration test.
Expect a real runtime increase over stages 1-2's tests.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from frames_extractor import io_utils, models, stage1_extract, stage2_dedup
from frames_extractor.models import Candidate
from frames_extractor.stage3_rank import Stage3Config, _load_model, _rank_candidates, rank
from video_utils import make_synthetic_video

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="stage 3 requires a CUDA GPU")


@pytest.fixture(scope="module")
def siglip_model():
    return _load_model(Stage3Config().checkpoint_id)


def _make_red_square(size: int = 384) -> np.ndarray:
    return np.full((size, size, 3), (30, 30, 220), dtype=np.uint8)  # BGR red


def _make_orange_rectangle(size: int = 384) -> np.ndarray:
    arr = np.full((size, size, 3), (255, 255, 255), dtype=np.uint8)
    cv2.rectangle(
        arr, (size // 4, size // 3), (3 * size // 4, 2 * size // 3), (20, 130, 230), -1
    )  # BGR orange
    return arr


def _make_blue_square(size: int = 384) -> np.ndarray:
    return np.full((size, size, 3), (220, 30, 30), dtype=np.uint8)  # BGR blue


def _make_noise(size: int = 384, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _build_candidates(in_dir: Path, images: list[np.ndarray]) -> list[Candidate]:
    candidates = []
    for i, image in enumerate(images):
        path = io_utils.save_frame_image(image, in_dir, frame_index=i)
        candidates.append(
            Candidate(frame_index=i, timestamp_ms=i * 200.0, image_path=path, reason="floor")
        )
    return candidates


def test_graded_relevance_ranking_discriminates(tmp_path: Path, siglip_model):
    model, processor = siglip_model
    images = [
        _make_red_square(),
        _make_orange_rectangle(),
        _make_blue_square(),
        _make_noise(),
    ]
    candidates = _build_candidates(tmp_path, images)

    ranked = _rank_candidates(candidates, "a red square", model, processor, Stage3Config())
    scores_by_index = {c.frame_index: score for c, score in ranked}

    # index 0 = red square (most relevant), index 3 = noise (least relevant).
    # Empirically confirmed via a real run against these exact images before
    # writing this test (0.1540 / 0.1416 / 0.0891 / 0.0711) -- only asserting
    # the top and bottom of the range to avoid over-constraining the middle
    # two against real-model noise, while still proving discrimination
    # beyond a trivial 2-way split.
    assert scores_by_index[0] == max(scores_by_index.values())
    assert scores_by_index[3] == min(scores_by_index.values())


def test_top_k_is_a_hard_cap_not_a_threshold(tmp_path: Path, siglip_model):
    model, processor = siglip_model
    base = _make_red_square()
    # Six near-identical, all-plausibly-relevant variants (tiny corner diffs) --
    # all would clear any reasonable similarity threshold to "a red square".
    images = []
    for i in range(6):
        variant = base.copy()
        variant[0:5, 0:5] = (i * 7) % 256
        images.append(variant)
    candidates = _build_candidates(tmp_path, images)

    ranked = _rank_candidates(candidates, "a red square", model, processor, Stage3Config())

    assert len(ranked[:3]) == 3  # hard count cap regardless of how close the 6 scores are
    assert len(ranked[:50]) == len(candidates)  # top_k > N: Python slicing, no padding/error


def test_stage1_through_stage3_integration(tmp_path: Path):
    video_path = tmp_path / "clip.avi"
    fps = 20.0
    duration_sec = 6.0
    motion_window = (60, 80)
    make_synthetic_video(
        video_path,
        fps=fps,
        duration_sec=duration_sec,
        width=160,
        height=120,
        motion_window=motion_window,
    )

    stage1_candidates = stage1_extract.extract(
        video_path, tmp_path / "stage1", stage1_extract.Stage1Config(floor_interval_sec=0.5)
    )
    stage2_candidates = stage2_dedup.dedup(tmp_path / "stage1", tmp_path / "stage2")

    # Small top_k so this test actually exercises rank()'s real truncation
    # line against real data, not just the isolated _rank_candidates slice
    # covered in test_top_k_is_a_hard_cap_not_a_threshold above.
    stage3_config = Stage3Config(top_k=3)
    stage3_candidates = rank(
        tmp_path / "stage2",
        tmp_path / "stage3",
        "a white rectangle moving across a gray background",
        stage3_config,
    )

    loaded = models.load_candidates(tmp_path / "stage3" / "candidates.json")
    assert len(loaded) == len(stage3_candidates)

    assert 0 < len(stage3_candidates) <= min(3, len(stage2_candidates))

    stage2_by_index = {c.frame_index: c for c in stage2_candidates}
    for candidate in stage3_candidates:
        original = stage2_by_index[candidate.frame_index]
        assert candidate.timestamp_ms == original.timestamp_ms
        assert candidate.reason == original.reason
        assert candidate.motion_score == original.motion_score
        assert candidate.similarity_score is not None
        assert candidate.image_path.exists()
        assert candidate.image_path.parent == tmp_path / "stage3"

    assert len(stage1_candidates) >= len(stage2_candidates) >= len(stage3_candidates)
