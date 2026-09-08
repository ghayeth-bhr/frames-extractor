"""Tests for eval/evaluate.py, using this machine's real run1 pipeline
output (data/work/run1, post-frame_index-3254 correction) and the real
eval/ground_truth.json as a fixture -- not synthetic data, since
evaluate.py's whole purpose is to score a real completed run.

Skipped entirely if that real output isn't present (e.g. a fresh clone
without data/, which is gitignored).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluate import DEFAULT_TOLERANCE_MS, evaluate, load_ground_truth
from backend import models

WORK_DIR = Path("data/work/run1")
GROUND_TRUTH_PATH = Path("eval/ground_truth.json")
VIDEO_PATH = Path("data/raw/7min.mp4")

pytestmark = pytest.mark.skipif(
    not (WORK_DIR / "stage5" / "candidates.json").exists(),
    reason="requires this machine's real run1 pipeline output under data/work/run1",
)


def test_ground_truth_loads_expected_events():
    events = load_ground_truth(GROUND_TRUTH_PATH)
    assert len(events) == 6
    assert events[0].start_ts == 46000
    assert events[0].end_ts == 47000
    assert events[0].description == "a man leaves the coffee shop"


def test_evaluate_against_real_corrected_run1():
    report = evaluate(WORK_DIR, GROUND_TRUTH_PATH, VIDEO_PATH)

    assert report.events_total == 6
    assert report.tolerance_ms == DEFAULT_TOLERANCE_MS
    assert 0.0 <= report.recall <= 1.0
    assert 0.0 <= report.precision <= 1.0

    # Post-correction: frame_index 3254 was fixed from keep->discard, so
    # kept_total must be 19, not the original (erroneous) 20, and 3254 must
    # not appear among the actual kept frame_index values.
    assert report.kept_total == 19
    assert report.funnel["stage5_kept"] == 19
    decisions = models.load_review_decisions(WORK_DIR / "stage5" / "candidates.json")
    kept_indices = [d.frame_index for d in decisions if d.decision == "keep"]
    assert 3254 not in kept_indices

    # Funnel must be monotonically non-increasing stage-to-stage, and match
    # the real video's actual frame count for the "raw" entry.
    assert report.funnel["raw"] == 12600
    assert report.funnel["stage1"] >= report.funnel["stage2"] >= report.funnel["stage3"]
    assert report.funnel["stage3"] >= report.funnel["stage4_verified"] >= report.funnel["stage5_kept"]

    # The diagnostic must no longer include frame_index 3254 (it's discarded now).
    assert 3254 not in report.kept_no_verdict_frame_indices


def test_evaluate_tolerance_ms_is_configurable_and_affects_recall():
    report_default = evaluate(WORK_DIR, GROUND_TRUTH_PATH, VIDEO_PATH, tolerance_ms=DEFAULT_TOLERANCE_MS)
    report_zero_tolerance = evaluate(WORK_DIR, GROUND_TRUTH_PATH, VIDEO_PATH, tolerance_ms=0.0)

    # A tighter tolerance can only recall the same or fewer events, never more.
    assert report_zero_tolerance.events_recalled <= report_default.events_recalled


def test_evaluate_without_video_omits_raw_frame_count():
    report = evaluate(WORK_DIR, GROUND_TRUTH_PATH, video_path=None)
    assert report.funnel["raw"] is None
