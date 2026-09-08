"""Tests for stage4_verify.py's --skip-vlm mode (Stage4Config.skip=True).

Deliberately a SEPARATE file from test_stage4_verify.py: that file's
module-level pytestmark skips everything unless a real Ollama server is
reachable, which would defeat the entire point of --skip-vlm (zero Ollama
dependency when set) if its tests lived there too.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend import io_utils, models, stage4_verify
from backend.models import Candidate
from backend.stage4_verify import Stage4Config


def _make_image() -> np.ndarray:
    image = np.full((120, 160, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (60, 40), (100, 80), (255, 255, 255), -1)
    return image


def _make_stage3_shaped_input(in_dir: Path, count: int) -> list[Candidate]:
    candidates = []
    for i in range(count):
        path = io_utils.save_frame_image(_make_image(), in_dir, frame_index=i)
        candidates.append(Candidate(frame_index=i, timestamp_ms=i * 200.0, image_path=path, reason="floor"))
    models.save_candidates(candidates, in_dir / "candidates.json")
    return candidates


QUERY = "a white rectangle on a gray background"


def test_skip_vlm_never_checks_ollama_reachable(tmp_path: Path):
    in_dir = tmp_path / "stage3"
    in_dir.mkdir()
    _make_stage3_shaped_input(in_dir, count=3)
    out_dir = tmp_path / "stage4"

    with patch("backend.stage4_verify._check_ollama_reachable") as mock_reachable, patch(
        "backend.stage4_verify.requests.post"
    ) as mock_post:
        stage4_verify.verify(in_dir, out_dir, QUERY, Stage4Config(skip=True))

    mock_reachable.assert_not_called()
    mock_post.assert_not_called()


def test_skip_vlm_produces_skipped_verdict_for_every_candidate(tmp_path: Path):
    in_dir = tmp_path / "stage3"
    in_dir.mkdir()
    _make_stage3_shaped_input(in_dir, count=4)
    out_dir = tmp_path / "stage4"

    result = stage4_verify.verify(in_dir, out_dir, QUERY, Stage4Config(skip=True))

    assert len(result) == 4
    assert all(vf.verdict == "skipped" for vf in result)
    assert all(vf.confidence is None for vf in result)
    assert all("skip" in vf.reasoning.lower() for vf in result)


def test_skip_vlm_writes_incremental_manifest(tmp_path: Path):
    in_dir = tmp_path / "stage3"
    in_dir.mkdir()
    _make_stage3_shaped_input(in_dir, count=3)
    out_dir = tmp_path / "stage4"

    stage4_verify.verify(in_dir, out_dir, QUERY, Stage4Config(skip=True))

    manifest = models.load_verified_frames(out_dir / "candidates.json")
    assert len(manifest) == 3
    assert all(vf.verdict == "skipped" for vf in manifest)


def test_skipped_verdicts_are_retried_by_a_later_non_skip_run(tmp_path: Path):
    in_dir = tmp_path / "stage3"
    in_dir.mkdir()
    _make_stage3_shaped_input(in_dir, count=3)
    out_dir = tmp_path / "stage4"

    result1 = stage4_verify.verify(in_dir, out_dir, QUERY, Stage4Config(skip=True))
    assert all(vf.verdict == "skipped" for vf in result1)

    def real_verify_side_effect(image, query, config):
        return ("yes", "retried for real", 0.9)

    with patch("backend.stage4_verify._check_ollama_reachable"), patch(
        "backend.stage4_verify._verify_frame", side_effect=real_verify_side_effect
    ) as mock_verify:
        result2 = stage4_verify.verify(in_dir, out_dir, QUERY, Stage4Config(skip=False))

    assert mock_verify.call_count == 3  # every previously-"skipped" frame retried, none treated as done
    assert all(vf.verdict == "yes" for vf in result2)


def test_skip_vlm_resumability_does_not_retry_already_skipped_frames_in_a_second_skip_run(tmp_path: Path):
    in_dir = tmp_path / "stage3"
    in_dir.mkdir()
    _make_stage3_shaped_input(in_dir, count=3)
    out_dir = tmp_path / "stage4"

    stage4_verify.verify(in_dir, out_dir, QUERY, Stage4Config(skip=True))

    # A second --skip-vlm run over the same output should be a no-op re-skip,
    # not an error -- confirmed by making sure it still returns all 3 with no
    # crash and no accidental real call (requests.post still never touched).
    with patch("backend.stage4_verify.requests.post") as mock_post:
        result2 = stage4_verify.verify(in_dir, out_dir, QUERY, Stage4Config(skip=True))

    mock_post.assert_not_called()
    assert len(result2) == 3
    assert all(vf.verdict == "skipped" for vf in result2)
