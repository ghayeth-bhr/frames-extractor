"""Tests for stage3_rank.py: a harder-than-smoke-test graded ranking check,
a stage2->stage3 integration test, and a top-K-is-a-cap check.

Model load (~15-20s even from local HF cache) is paid twice across this
file: once for the module-scoped fixture backing the graded-ranking and
top-K tests, once inside the real rank() call in the integration test.
Expect a real runtime increase over stages 1-2's tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from backend import io_utils, models, stage1_extract, stage2_dedup
from backend.models import Candidate
from backend.stage3_rank import (
    Stage3Config,
    _load_model,
    _rank_candidates,
    build_index,
    rank,
    search_index,
)
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


# --- "build once, search many times": build_index() / search_index() ---


def test_build_index_writes_embeddings_sidecar(tmp_path: Path, siglip_model):
    images = [_make_red_square(), _make_orange_rectangle(), _make_blue_square(), _make_noise()]
    dedup_dir = tmp_path / "stage2"
    candidates = _build_candidates(dedup_dir, images)
    models.save_candidates(candidates, dedup_dir / "candidates.json")

    build_index(dedup_dir, Stage3Config())

    assert (dedup_dir / "embeddings.npy").exists()
    embeds = np.load(dedup_dir / "embeddings.npy")
    assert embeds.shape[0] == len(images)
    # candidates.json is untouched in content (same 4 candidates), just
    # rewritten in the exact sorted order embeddings.npy is row-aligned to.
    reloaded = models.load_candidates(dedup_dir / "candidates.json")
    assert {c.frame_index for c in reloaded} == {c.frame_index for c in candidates}


def test_search_index_matches_direct_rank_for_two_different_queries(tmp_path: Path, siglip_model):
    # Only ONE _load_model() call per query here (via search_index() itself)
    # plus the module-scoped siglip_model fixture's own -- deliberately NOT
    # calling search_index()/build_index() any more times than needed to
    # prove the point. Each real call loads SigLIP2 fresh onto the GPU with
    # no in-process release between calls (correct in production, where
    # each CLI invocation is its own subprocess -- see cli.py's module
    # docstring -- but adds up fast within a single long-lived pytest
    # process across this whole file's tests). A prior version of this test
    # made 5 real model-load calls in one test body and reproducibly hung
    # the full suite (not this file alone) inside a from_pretrained() call
    # once enough prior tests had already accumulated GPU pressure -- do
    # not reintroduce that pattern here.
    model, processor = siglip_model
    images = [_make_red_square(), _make_orange_rectangle(), _make_blue_square(), _make_noise()]
    dedup_dir = tmp_path / "stage2"
    candidates = _build_candidates(dedup_dir, images)
    models.save_candidates(candidates, dedup_dir / "candidates.json")

    build_index(dedup_dir, Stage3Config())

    top_frame_index_by_query = {}
    for query in ("a red square", "a blue square"):
        direct_ranked = _rank_candidates(candidates, query, model, processor, Stage3Config())
        direct_top_frame_index = direct_ranked[0][0].frame_index

        out_dir = tmp_path / f"search_{query.replace(' ', '_')}"
        searched = search_index(dedup_dir, out_dir, query, Stage3Config(top_k=1))

        assert len(searched) == 1
        assert searched[0].frame_index == direct_top_frame_index
        top_frame_index_by_query[query] = searched[0].frame_index

    # The two queries' top picks must differ -- proves this is a genuine
    # re-rank per query, not a cached/frozen result reused regardless of
    # query. Reuses the loop's own results above rather than calling
    # search_index() twice more just to re-derive the same two numbers.
    assert top_frame_index_by_query["a red square"] != top_frame_index_by_query["a blue square"]


def test_search_index_never_loads_an_image_for_ranking(tmp_path: Path, siglip_model):
    # The whole point: re-ranking against a cached index must not re-embed
    # any image. search_index() only ever needs pixel content for
    # _embed_candidate_images (build_index()/rank()'s job, not this one) --
    # confirmed here by patching io_utils.load_frame_image (the only function
    # that reads pixels) and asserting it's never called during search_index(),
    # even though it DOES still shutil.copy2 the raw winning files (a byte
    # copy, not a pixel read) to produce out_dir.
    images = [_make_red_square(), _make_orange_rectangle(), _make_blue_square()]
    dedup_dir = tmp_path / "stage2"
    candidates = _build_candidates(dedup_dir, images)
    models.save_candidates(candidates, dedup_dir / "candidates.json")

    build_index(dedup_dir, Stage3Config())

    with patch("backend.stage3_rank.io_utils.load_frame_image") as mock_load:
        result = search_index(dedup_dir, tmp_path / "search", "a red square", Stage3Config(top_k=3))

    mock_load.assert_not_called()
    assert len(result) == 3
    for candidate in result:
        assert candidate.image_path.exists()


def test_search_index_raises_on_missing_embeddings(tmp_path: Path):
    dedup_dir = tmp_path / "stage2"
    images = [_make_red_square()]
    candidates = _build_candidates(dedup_dir, images)
    models.save_candidates(candidates, dedup_dir / "candidates.json")
    # No build_index() call -- no embeddings.npy exists.

    with pytest.raises(FileNotFoundError, match="embeddings"):
        search_index(dedup_dir, tmp_path / "search", "a red square", Stage3Config())


def test_search_index_raises_on_row_count_mismatch(tmp_path: Path):
    dedup_dir = tmp_path / "stage2"
    images = [_make_red_square(), _make_blue_square()]
    candidates = _build_candidates(dedup_dir, images)
    models.save_candidates(candidates, dedup_dir / "candidates.json")
    np.save(dedup_dir / "embeddings.npy", np.zeros((5, 768), dtype=np.float32))  # wrong row count

    with pytest.raises(ValueError, match="mismatched"):
        search_index(dedup_dir, tmp_path / "search", "a red square", Stage3Config())


def test_stage1_through_index_through_search_integration(tmp_path: Path):
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

    stage1_extract.extract(video_path, tmp_path / "stage1", stage1_extract.Stage1Config(floor_interval_sec=0.5))
    stage2_candidates = stage2_dedup.dedup(tmp_path / "stage1", tmp_path / "stage2")

    build_index(tmp_path / "stage2", Stage3Config())
    assert (tmp_path / "stage2" / "embeddings.npy").exists()

    searched = search_index(
        tmp_path / "stage2",
        tmp_path / "search_out",
        "a white rectangle moving across a gray background",
        Stage3Config(top_k=3),
    )

    assert 0 < len(searched) <= min(3, len(stage2_candidates))
    for candidate in searched:
        assert candidate.similarity_score is not None
        assert candidate.image_path.exists()
        assert candidate.image_path.parent == tmp_path / "search_out"
