"""Tests for multi_source_index.py -- the webapp's appendable, multi-source
combined index. Real-GPU-gated, same pattern as test_stage3_rank.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from backend import io_utils, models
from backend.models import Candidate
from vectordb.multi_source_index import add_to_index, search_multi_source_index
from backend.stage3_rank import Stage3Config, _load_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="multi-source indexing requires a CUDA GPU")


@pytest.fixture(scope="module")
def siglip_model():
    return _load_model(Stage3Config().checkpoint_id)


def _make_red_square(size: int = 384) -> np.ndarray:
    return np.full((size, size, 3), (30, 30, 220), dtype=np.uint8)  # BGR red


def _make_blue_square(size: int = 384) -> np.ndarray:
    return np.full((size, size, 3), (220, 30, 30), dtype=np.uint8)  # BGR blue


def _build_candidates(in_dir: Path, images: list[np.ndarray]) -> list[Candidate]:
    candidates = []
    for i, image in enumerate(images):
        path = io_utils.save_frame_image(image, in_dir, frame_index=i)
        candidates.append(Candidate(frame_index=i, timestamp_ms=i * 200.0, image_path=path, reason="floor"))
    models.save_candidates(candidates, in_dir / "candidates.json")
    return candidates


def test_add_to_index_creates_fresh_index(tmp_path: Path, siglip_model):
    stage2_dir = tmp_path / "source_a" / "stage2"
    _build_candidates(stage2_dir, [_make_red_square(), _make_red_square()])
    index_dir = tmp_path / "index"

    add_to_index(index_dir, stage2_dir, "source_a", Stage3Config())

    assert (index_dir / "embeddings.npy").exists()
    embeds = np.load(index_dir / "embeddings.npy")
    assert embeds.shape[0] == 2
    frames = models.load_indexed_frames(index_dir / "candidates.json")
    assert len(frames) == 2
    assert all(f.source_id == "source_a" for f in frames)


def test_add_to_index_appends_second_source(tmp_path: Path, siglip_model):
    stage2_dir_a = tmp_path / "source_a" / "stage2"
    _build_candidates(stage2_dir_a, [_make_red_square()])
    stage2_dir_b = tmp_path / "source_b" / "stage2"
    _build_candidates(stage2_dir_b, [_make_blue_square(), _make_blue_square()])
    index_dir = tmp_path / "index"

    add_to_index(index_dir, stage2_dir_a, "source_a", Stage3Config())
    add_to_index(index_dir, stage2_dir_b, "source_b", Stage3Config())

    frames = models.load_indexed_frames(index_dir / "candidates.json")
    embeds = np.load(index_dir / "embeddings.npy")
    assert len(frames) == 3
    assert embeds.shape[0] == 3
    assert {f.source_id for f in frames} == {"source_a", "source_b"}


def test_add_to_index_replaces_not_duplicates_same_source(tmp_path: Path, siglip_model):
    stage2_dir = tmp_path / "source_a" / "stage2"
    _build_candidates(stage2_dir, [_make_red_square()])
    index_dir = tmp_path / "index"

    add_to_index(index_dir, stage2_dir, "source_a", Stage3Config())
    add_to_index(index_dir, stage2_dir, "source_a", Stage3Config())  # simulated retry

    frames = models.load_indexed_frames(index_dir / "candidates.json")
    embeds = np.load(index_dir / "embeddings.npy")
    assert len(frames) == 1  # not 2 -- re-adding the same source_id replaces, doesn't accumulate
    assert embeds.shape[0] == 1


def test_add_to_index_replace_does_not_disturb_other_sources(tmp_path: Path, siglip_model):
    stage2_dir_a = tmp_path / "source_a" / "stage2"
    _build_candidates(stage2_dir_a, [_make_red_square()])
    stage2_dir_b = tmp_path / "source_b" / "stage2"
    _build_candidates(stage2_dir_b, [_make_blue_square()])
    index_dir = tmp_path / "index"

    add_to_index(index_dir, stage2_dir_a, "source_a", Stage3Config())
    add_to_index(index_dir, stage2_dir_b, "source_b", Stage3Config())
    add_to_index(index_dir, stage2_dir_a, "source_a", Stage3Config())  # re-add source_a only

    frames = models.load_indexed_frames(index_dir / "candidates.json")
    embeds = np.load(index_dir / "embeddings.npy")
    assert len(frames) == 2
    assert embeds.shape[0] == 2
    assert {f.source_id for f in frames} == {"source_a", "source_b"}


# --- cross_source_dedup: pure pHash-filtering logic, orthogonal to actual
# embedding values -- mocked out (no real from_pretrained()/CUDA) rather
# than adding more real model loads onto this file's already-real-GPU-heavy
# test list. See CLAUDE.md: many sequential real SigLIP2 loads in one
# pytest session has reproducibly hung this machine; these 4 tests would
# otherwise have doubled the file's real load count.


def _fake_embed_candidates(candidates, model, processor, batch_size):
    return np.zeros((len(candidates), 4), dtype=np.float32)


@pytest.fixture
def mock_embedding(monkeypatch):
    monkeypatch.setattr("vectordb.multi_source_index._load_model", lambda checkpoint_id: (None, None))
    monkeypatch.setattr("vectordb.multi_source_index._embed_candidate_images", _fake_embed_candidates)


def test_add_to_index_cross_source_dedup_off_by_default(tmp_path: Path, mock_embedding):
    stage2_dir_a = tmp_path / "source_a" / "stage2"
    _build_candidates(stage2_dir_a, [_make_red_square()])
    stage2_dir_b = tmp_path / "source_b" / "stage2"
    _build_candidates(stage2_dir_b, [_make_red_square()])  # identical content, different source
    index_dir = tmp_path / "index"

    add_to_index(index_dir, stage2_dir_a, "source_a", Stage3Config())
    add_to_index(index_dir, stage2_dir_b, "source_b", Stage3Config())

    frames = models.load_indexed_frames(index_dir / "candidates.json")
    assert len(frames) == 2  # no cross-source dedup unless opted in -- both kept
    assert {f.source_id for f in frames} == {"source_a", "source_b"}
    assert all(f.phash is None for f in frames)


def test_add_to_index_cross_source_dedup_excludes_visual_duplicate(tmp_path: Path, mock_embedding):
    stage2_dir_a = tmp_path / "source_a" / "stage2"
    _build_candidates(stage2_dir_a, [_make_red_square()])
    stage2_dir_b = tmp_path / "source_b" / "stage2"
    _build_candidates(stage2_dir_b, [_make_red_square()])  # identical content
    index_dir = tmp_path / "index"

    add_to_index(index_dir, stage2_dir_a, "source_a", Stage3Config(), cross_source_dedup=True)
    add_to_index(index_dir, stage2_dir_b, "source_b", Stage3Config(), cross_source_dedup=True)

    frames = models.load_indexed_frames(index_dir / "candidates.json")
    embeds = np.load(index_dir / "embeddings.npy")
    assert len(frames) == 1  # source_b's identical frame excluded from the shared index
    assert frames[0].source_id == "source_a"
    assert embeds.shape[0] == 1
    assert frames[0].phash is not None


def test_add_to_index_cross_source_dedup_keeps_visually_distinct_frames(tmp_path: Path, mock_embedding):
    stage2_dir_a = tmp_path / "source_a" / "stage2"
    _build_candidates(stage2_dir_a, [_make_red_square()])
    stage2_dir_b = tmp_path / "source_b" / "stage2"
    _build_candidates(stage2_dir_b, [_make_blue_square()])
    index_dir = tmp_path / "index"

    add_to_index(index_dir, stage2_dir_a, "source_a", Stage3Config(), cross_source_dedup=True)
    add_to_index(index_dir, stage2_dir_b, "source_b", Stage3Config(), cross_source_dedup=True)

    frames = models.load_indexed_frames(index_dir / "candidates.json")
    assert len(frames) == 2  # visually distinct -- neither excluded
    assert {f.source_id for f in frames} == {"source_a", "source_b"}


def test_add_to_index_cross_source_dedup_needs_both_sources_opted_in(tmp_path: Path, mock_embedding):
    stage2_dir_a = tmp_path / "source_a" / "stage2"
    _build_candidates(stage2_dir_a, [_make_red_square()])
    stage2_dir_b = tmp_path / "source_b" / "stage2"
    _build_candidates(stage2_dir_b, [_make_red_square()])  # identical content
    index_dir = tmp_path / "index"

    add_to_index(index_dir, stage2_dir_a, "source_a", Stage3Config())  # NOT opted in -- no phash stored
    add_to_index(index_dir, stage2_dir_b, "source_b", Stage3Config(), cross_source_dedup=True)

    frames = models.load_indexed_frames(index_dir / "candidates.json")
    # source_a recorded no phash for source_b to compare against, so
    # source_b's identical frame is NOT excluded -- a documented limitation,
    # not a bug: cross-source dedup only ever compares against sources that
    # were themselves indexed with cross_source_dedup=True.
    assert len(frames) == 2
    assert {f.source_id for f in frames} == {"source_a", "source_b"}


def test_search_multi_source_index_returns_results_from_both_sources(tmp_path: Path, siglip_model):
    # Deliberately a single real search covering both the "spans multiple
    # sources" and "never copies images" checks together -- each additional
    # real add_to_index()/search_multi_source_index() call is a fresh
    # from_pretrained() load with no in-process release (correct for
    # production's one-call-per-subprocess model, costly repeated many
    # times within one long pytest session -- see
    # test_stage3_rank.py's test_search_index_matches_direct_rank_for_two_different_queries
    # for the reproduced full-suite hang this is written to avoid repeating).
    stage2_dir_a = tmp_path / "source_a" / "stage2"
    _build_candidates(stage2_dir_a, [_make_red_square(), _make_red_square()])
    stage2_dir_b = tmp_path / "source_b" / "stage2"
    _build_candidates(stage2_dir_b, [_make_blue_square(), _make_blue_square()])
    index_dir = tmp_path / "index"

    add_to_index(index_dir, stage2_dir_a, "source_a", Stage3Config())
    add_to_index(index_dir, stage2_dir_b, "source_b", Stage3Config())

    ranked = search_multi_source_index(index_dir, "a red square", 4, Stage3Config())

    assert len(ranked) == 4
    assert {f.source_id for f in ranked} == {"source_a", "source_b"}
    # The red-square query's top hits should be source_a's frames, since
    # that's the actually-matching content.
    assert ranked[0].source_id == "source_a"
    assert all(f.similarity_score is not None for f in ranked)
    # image_path still points at the ORIGINAL source directory, not a copy.
    for f in ranked:
        expected_dir = stage2_dir_a if f.source_id == "source_a" else stage2_dir_b
        assert f.image_path.parent == expected_dir


def test_search_multi_source_index_raises_on_missing_embeddings(tmp_path: Path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="embeddings"):
        search_multi_source_index(index_dir, "a red square", 5)


def test_search_multi_source_index_raises_on_row_count_mismatch(tmp_path: Path):
    stage2_dir = tmp_path / "source_a" / "stage2"
    candidates = _build_candidates(stage2_dir, [_make_red_square()])
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    models.save_indexed_frames(
        [
            models.IndexedFrame(
                source_id="source_a",
                frame_index=candidates[0].frame_index,
                timestamp_ms=candidates[0].timestamp_ms,
                image_path=candidates[0].image_path,
                reason=candidates[0].reason,
            )
        ],
        index_dir / "candidates.json",
    )
    np.save(index_dir / "embeddings.npy", np.zeros((5, 768), dtype=np.float32))  # wrong row count

    with pytest.raises(ValueError, match="mismatched"):
        search_multi_source_index(index_dir, "a red square", 5)
