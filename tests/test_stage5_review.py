"""Tests for stage5_review.py.

All tests drive _run_review_loop directly via a scripted get_action and a
no-op show_frame -- no real cv2 GUI interaction anywhere in this file,
per stage 5's design (the event loop's actual logic is separated from the
untested cv2.imshow/waitKey wrapper in review()).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from frames_extractor import io_utils, models, stage1_extract, stage2_dedup, stage3_rank, stage4_verify, stage5_review
from frames_extractor.models import ReviewDecision, VerifiedFrame
from frames_extractor.stage3_rank import Stage3Config
from frames_extractor.stage5_review import _action_for_key, _draw_overlay, _run_review_loop
from video_utils import make_synthetic_video

QUERY = "a white rectangle on a gray background"


def _make_image() -> np.ndarray:
    return np.full((120, 160, 3), 128, dtype=np.uint8)


def _make_verified_frames(in_dir: Path, count: int) -> list[VerifiedFrame]:
    frames = []
    for i in range(count):
        path = io_utils.save_frame_image(_make_image(), in_dir, frame_index=i)
        frames.append(
            VerifiedFrame(
                frame_index=i,
                timestamp_ms=i * 200.0,
                image_path=path,
                reason="floor",
                verdict="yes",
                reasoning="test reasoning",
                confidence=0.9,
            )
        )
    return frames


# --- _action_for_key ---


def test_action_for_key_mapping():
    assert _action_for_key(ord("k")) == "keep"
    assert _action_for_key(ord("d")) == "discard"
    assert _action_for_key(ord(" ")) == "skip"
    assert _action_for_key(ord("n")) == "skip"
    assert _action_for_key(ord("q")) == "quit"
    assert _action_for_key(ord("x")) is None


# --- _draw_overlay ---


def test_draw_overlay_normal_confidence():
    image = _make_image()
    candidate = VerifiedFrame(
        frame_index=0,
        timestamp_ms=0.0,
        image_path=Path("dummy.jpg"),
        reason="floor",
        verdict="yes",
        reasoning="looks like a match",
        confidence=0.87,
    )
    annotated = _draw_overlay(image, QUERY, candidate)
    assert annotated.shape == image.shape
    assert annotated.dtype == image.dtype


def test_draw_overlay_none_confidence_does_not_crash():
    image = _make_image()
    candidate = VerifiedFrame(
        frame_index=0,
        timestamp_ms=0.0,
        image_path=Path("dummy.jpg"),
        reason="floor",
        verdict="error",
        reasoning="ConnectionError: could not reach server",
        confidence=None,
    )
    annotated = _draw_overlay(image, QUERY, candidate)
    assert annotated.shape == image.shape


def test_draw_overlay_long_reasoning_truncated_not_crashed():
    image = _make_image()
    candidate = VerifiedFrame(
        frame_index=0,
        timestamp_ms=0.0,
        image_path=Path("dummy.jpg"),
        reason="floor",
        verdict="yes",
        reasoning="x" * 500,
        confidence=0.5,
    )
    annotated = _draw_overlay(image, QUERY, candidate)
    assert annotated.shape == image.shape


# --- _run_review_loop ---


def test_skip_writes_nothing(tmp_path: Path):
    in_dir = tmp_path / "stage4"
    in_dir.mkdir()
    candidates = _make_verified_frames(in_dir, count=1)
    out_dir = tmp_path / "stage5"
    out_dir.mkdir()
    manifest_path = out_dir / "candidates.json"
    decisions: dict[int, ReviewDecision] = {}

    _run_review_loop(candidates, decisions, QUERY, out_dir, manifest_path, get_action=lambda c: "skip")

    assert decisions == {}
    assert not manifest_path.exists()


def test_keep_and_discard_write_incrementally(tmp_path: Path):
    in_dir = tmp_path / "stage4"
    in_dir.mkdir()
    candidates = _make_verified_frames(in_dir, count=2)
    out_dir = tmp_path / "stage5"
    out_dir.mkdir()
    manifest_path = out_dir / "candidates.json"
    decisions: dict[int, ReviewDecision] = {}

    actions = iter(["keep", "discard"])
    _run_review_loop(candidates, decisions, QUERY, out_dir, manifest_path, get_action=lambda c: next(actions))

    # Reload from disk -- proves incremental write, not just in-memory state.
    loaded = models.load_review_decisions(manifest_path)
    assert {d.frame_index: d.decision for d in loaded} == {0: "keep", 1: "discard"}


def test_quit_stops_early(tmp_path: Path):
    in_dir = tmp_path / "stage4"
    in_dir.mkdir()
    candidates = _make_verified_frames(in_dir, count=3)
    out_dir = tmp_path / "stage5"
    out_dir.mkdir()
    manifest_path = out_dir / "candidates.json"
    decisions: dict[int, ReviewDecision] = {}

    actions = iter(["keep", "quit"])
    _run_review_loop(candidates, decisions, QUERY, out_dir, manifest_path, get_action=lambda c: next(actions))

    assert set(decisions.keys()) == {0}  # frame_index 2 never reached


def test_resume_skips_already_decided_frames(tmp_path: Path):
    in_dir = tmp_path / "stage4"
    in_dir.mkdir()
    candidates = _make_verified_frames(in_dir, count=3)
    out_dir = tmp_path / "stage5"
    out_dir.mkdir()
    manifest_path = out_dir / "candidates.json"

    pre_existing = ReviewDecision(
        frame_index=0,
        timestamp_ms=candidates[0].timestamp_ms,
        image_path=candidates[0].image_path,
        reason=candidates[0].reason,
        verdict=candidates[0].verdict,
        reasoning=candidates[0].reasoning,
        confidence=candidates[0].confidence,
        decision="keep",
    )
    decisions: dict[int, ReviewDecision] = {0: pre_existing}

    def get_action(candidate: VerifiedFrame):
        if candidate.frame_index == 0:
            raise AssertionError("get_action must not be called for an already-decided frame")
        return "keep"

    _run_review_loop(candidates, decisions, QUERY, out_dir, manifest_path, get_action=get_action)

    assert set(decisions.keys()) == {0, 1, 2}


def test_discard_never_copies_file_keep_does(tmp_path: Path):
    in_dir = tmp_path / "stage4"
    in_dir.mkdir()
    candidates = _make_verified_frames(in_dir, count=2)
    out_dir = tmp_path / "stage5"
    out_dir.mkdir()
    manifest_path = out_dir / "candidates.json"
    decisions: dict[int, ReviewDecision] = {}

    actions = iter(["keep", "discard"])
    _run_review_loop(candidates, decisions, QUERY, out_dir, manifest_path, get_action=lambda c: next(actions))

    kept_dest = out_dir / candidates[0].image_path.name
    discarded_dest = out_dir / candidates[1].image_path.name
    assert kept_dest.exists()
    assert not discarded_dest.exists()

    loaded_by_index = {d.frame_index: d for d in models.load_review_decisions(manifest_path)}
    assert loaded_by_index[0].image_path == kept_dest
    assert loaded_by_index[1].image_path == candidates[1].image_path  # original stage-4 location, unchanged


# --- stage1->stage2->stage3->stage4->stage5 integration, real code throughout ---


def test_stage1_through_stage5_integration(tmp_path: Path):
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
    stage2_dedup.dedup(tmp_path / "stage1", tmp_path / "stage2")
    stage3_rank.rank(tmp_path / "stage2", tmp_path / "stage3", QUERY, Stage3Config(top_k=3))
    stage4_candidates = stage4_verify.verify(tmp_path / "stage3", tmp_path / "stage4", QUERY)

    out_dir = tmp_path / "stage5"
    out_dir.mkdir()
    manifest_path = out_dir / "candidates.json"
    decisions: dict[int, ReviewDecision] = {}
    candidates = sorted(
        models.load_verified_frames(tmp_path / "stage4" / "candidates.json"), key=lambda c: c.frame_index
    )
    actions = itertools.cycle(["keep", "discard"])
    stage5_review._run_review_loop(
        candidates, decisions, QUERY, out_dir, manifest_path, get_action=lambda c: next(actions)
    )

    loaded = models.load_review_decisions(manifest_path)
    assert len(loaded) == len(candidates) == len(stage4_candidates)

    stage4_by_index = {c.frame_index: c for c in stage4_candidates}
    for decision in loaded:
        original = stage4_by_index[decision.frame_index]
        assert decision.verdict == original.verdict
        assert decision.reasoning == original.reasoning
        assert decision.confidence == original.confidence
        assert decision.motion_score == original.motion_score
        assert decision.similarity_score == original.similarity_score

        if decision.decision == "keep":
            assert decision.image_path.exists()
            assert decision.image_path.parent == out_dir
        else:
            assert decision.image_path == original.image_path
