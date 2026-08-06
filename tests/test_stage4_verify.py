"""Tests for stage4_verify.py.

(a) discrimination cases use the REAL Ollama/qwen3-vl:4b model (no API cost
to worry about, unlike a cloud VLM) -- each is a real network call, so
expect a real runtime increase for this file (~1-2 minutes).
(b)/(b.5)/(c) mock the network boundary deliberately, since they need
precise, deterministic control over which calls succeed/fail and how many
times a function is invoked -- something the real model can't guarantee.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from frames_extractor import io_utils, models, stage1_extract, stage2_dedup, stage3_rank, stage4_verify
from frames_extractor.models import Candidate
from frames_extractor.stage3_rank import Stage3Config
from frames_extractor.stage4_verify import Stage4Config
from video_utils import make_synthetic_video

QUERY = "a white rectangle on a gray background"


def _ollama_available() -> bool:
    try:
        stage4_verify._check_ollama_reachable(Stage4Config().base_url)
        return True
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _ollama_available(), reason="stage 4 requires a reachable Ollama server")


def _make_match_image() -> np.ndarray:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (60, 40), (100, 80), (255, 255, 255), -1)
    return image


def _make_checkerboard_image(tile: int = 10) -> np.ndarray:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    for y in range(0, 120, tile * 2):
        image[y : y + tile, :, :] = 255
    for x in range(0, 160, tile * 2):
        image[:, x : x + tile, :] = 255
    return image


def _make_wrong_color_image() -> np.ndarray:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (60, 40), (100, 80), (220, 30, 30), -1)  # BGR blue
    return image


def _make_low_contrast_image() -> np.ndarray:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (60, 40), (100, 80), (140, 140, 140), -1)  # barely lighter than bg
    return image


def _make_thin_sliver_image() -> np.ndarray:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (78, 20), (82, 100), (255, 255, 255), -1)  # very thin vertical sliver
    return image


# --- (a) discrimination cases, real model ---


def test_verify_frame_clean_match():
    verdict, reasoning, confidence = stage4_verify._verify_frame(_make_match_image(), QUERY, Stage4Config())
    assert verdict == "yes"
    assert confidence is not None


def test_verify_frame_clean_non_match():
    verdict, reasoning, confidence = stage4_verify._verify_frame(
        _make_checkerboard_image(), QUERY, Stage4Config()
    )
    assert verdict == "no"


def test_verify_frame_wrong_color_non_match():
    verdict, reasoning, confidence = stage4_verify._verify_frame(
        _make_wrong_color_image(), QUERY, Stage4Config()
    )
    assert verdict == "no"


def test_verify_frame_ambiguous_low_contrast():
    verdict, reasoning, confidence = stage4_verify._verify_frame(
        _make_low_contrast_image(), QUERY, Stage4Config()
    )
    assert verdict in ("yes", "no")
    assert confidence is not None
    assert 0.0 <= confidence <= 1.0


def test_verify_frame_ambiguous_thin_sliver():
    verdict, reasoning, confidence = stage4_verify._verify_frame(
        _make_thin_sliver_image(), QUERY, Stage4Config()
    )
    assert verdict in ("yes", "no")
    assert confidence is not None
    assert 0.0 <= confidence <= 1.0


# --- stage1->stage2->stage3->stage4 integration, real model, no mocks ---


def test_stage1_through_stage4_integration(tmp_path: Path):
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
    # Small top_k so the real VLM only has to verify a handful of frames -- keeps this fast.
    stage3_candidates = stage3_rank.rank(
        tmp_path / "stage2", tmp_path / "stage3", QUERY, Stage3Config(top_k=3)
    )

    stage4_candidates = stage4_verify.verify(tmp_path / "stage3", tmp_path / "stage4", QUERY)

    loaded = models.load_verified_frames(tmp_path / "stage4" / "candidates.json")
    assert len(loaded) == len(stage4_candidates)
    assert len(stage4_candidates) == len(stage3_candidates) > 0

    stage3_by_index = {c.frame_index: c for c in stage3_candidates}
    for verified in stage4_candidates:
        # Never raises, even if a genuinely ambiguous real frame comes back "error".
        assert verified.verdict is not None
        assert verified.verdict in ("yes", "no", "error")

        original = stage3_by_index[verified.frame_index]
        assert verified.similarity_score == original.similarity_score
        assert verified.motion_score == original.motion_score
        assert verified.image_path.exists()
        assert verified.image_path.parent == tmp_path / "stage4"

    assert len(stage1_candidates) >= len(stage2_candidates) >= len(stage3_candidates)


# --- (b) resumability, mocked ---


def _make_stage3_shaped_input(in_dir: Path, count: int) -> list[Candidate]:
    candidates = []
    for i in range(count):
        path = io_utils.save_frame_image(_make_match_image(), in_dir, frame_index=i)
        candidates.append(
            Candidate(frame_index=i, timestamp_ms=i * 200.0, image_path=path, reason="floor")
        )
    models.save_candidates(candidates, in_dir / "candidates.json")
    return candidates


def test_resumability_only_retries_errored_frames(tmp_path: Path):
    in_dir = tmp_path / "stage3"
    in_dir.mkdir()
    _make_stage3_shaped_input(in_dir, count=5)
    out_dir = tmp_path / "stage4"

    def first_run_side_effect(image, query, config):
        idx = first_run_side_effect.calls
        first_run_side_effect.calls += 1
        if idx < 3:
            return ("yes", f"frame {idx} matches", 0.9)
        return ("error", "simulated mid-batch failure", None)

    first_run_side_effect.calls = 0

    with patch("frames_extractor.stage4_verify._verify_frame", side_effect=first_run_side_effect) as mock1:
        result1 = stage4_verify.verify(in_dir, out_dir, QUERY)

    assert mock1.call_count == 5
    verdicts1 = {vf.frame_index: vf.verdict for vf in result1}
    assert verdicts1 == {0: "yes", 1: "yes", 2: "yes", 3: "error", 4: "error"}

    def second_run_side_effect(image, query, config):
        return ("yes", "retried successfully", 0.9)

    with patch("frames_extractor.stage4_verify._verify_frame", side_effect=second_run_side_effect) as mock2:
        result2 = stage4_verify.verify(in_dir, out_dir, QUERY)

    assert mock2.call_count == 2  # only frames 3 and 4 (previously "error") retried
    verdicts2 = {vf.frame_index: vf.verdict for vf in result2}
    assert len(verdicts2) == 5
    assert all(v == "yes" for v in verdicts2.values())


# --- (b.5) malformed-response handling, mocked at the requests.post boundary ---


def _fake_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"message": {"content": content}}
    return resp


def test_verify_frame_handles_null_confidence_without_raising():
    fake = _fake_response('{"verdict": "yes", "reasoning": "ok", "confidence": null}')
    with patch("frames_extractor.stage4_verify.requests.post", return_value=fake):
        verdict, reasoning, confidence = stage4_verify._verify_frame(
            _make_match_image(), QUERY, Stage4Config()
        )
    assert verdict == "error"
    assert confidence is None


def test_verify_frame_handles_array_content_without_raising():
    fake = _fake_response("[1, 2, 3]")
    with patch("frames_extractor.stage4_verify.requests.post", return_value=fake):
        verdict, reasoning, confidence = stage4_verify._verify_frame(
            _make_match_image(), QUERY, Stage4Config()
        )
    assert verdict == "error"
    assert confidence is None


# --- (c) fail-fast startup check ---


def test_check_ollama_reachable_raises_clear_error_on_bad_port():
    with pytest.raises(RuntimeError, match="localhost:1"):
        stage4_verify._check_ollama_reachable("http://localhost:1")


def test_verify_fails_fast_before_processing_any_frames(tmp_path: Path):
    in_dir = tmp_path / "stage3"
    in_dir.mkdir()
    _make_stage3_shaped_input(in_dir, count=1)
    out_dir = tmp_path / "stage4"

    bad_config = Stage4Config(base_url="http://localhost:1")

    with pytest.raises(RuntimeError, match="localhost:1"):
        stage4_verify.verify(in_dir, out_dir, QUERY, bad_config)

    assert not (out_dir / "candidates.json").exists()
