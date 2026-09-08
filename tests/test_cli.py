"""Tests for cli.py: argument parsing, dispatch, and _run_pipeline's
orchestration. subprocess.run is always mocked here -- no real subprocess,
GPU, or Ollama call is ever spawned by this file.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend import cli, models
from backend.models import Candidate, ReviewDecision, VerifiedFrame

# --- argument parsing ---


def test_parse_extract():
    args = cli.build_parser().parse_args(["extract", "--video", "clip.mp4", "--out", "out1"])
    assert args.command == "extract"
    assert args.video == Path("clip.mp4")
    assert args.out == Path("out1")
    assert args.mask_regions is None  # default: not specified (distinct from explicit [])
    assert args.auto_mask is False


def test_parse_extract_mask_regions():
    args = cli.build_parser().parse_args(
        ["extract", "--video", "clip.mp4", "--out", "out1", "--mask-regions", "0,60,750,75"]
    )
    assert args.mask_regions == [(0, 60, 750, 75)]


def test_parse_extract_multiple_mask_regions():
    args = cli.build_parser().parse_args(
        ["extract", "--video", "clip.mp4", "--out", "out1", "--mask-regions", "0,60,750,75", "800,0,200,50"]
    )
    assert args.mask_regions == [(0, 60, 750, 75), (800, 0, 200, 50)]


def test_parse_extract_invalid_mask_region_raises_system_exit():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["extract", "--video", "clip.mp4", "--out", "out1", "--mask-regions", "not-valid"]
        )


def test_parse_extract_auto_mask():
    args = cli.build_parser().parse_args(["extract", "--video", "clip.mp4", "--out", "out1", "--auto-mask"])
    assert args.auto_mask is True
    assert args.mask_regions is None


def test_parse_dedup_uses_in_dir_dest():
    args = cli.build_parser().parse_args(["dedup", "--in", "stage1", "--out", "stage2"])
    assert args.command == "dedup"
    assert args.in_dir == Path("stage1")
    assert args.out == Path("stage2")


def test_parse_rank():
    args = cli.build_parser().parse_args(["rank", "--in", "stage2", "--query", "a cat", "--out", "stage3"])
    assert args.command == "rank"
    assert args.in_dir == Path("stage2")
    assert args.query == "a cat"
    assert args.out == Path("stage3")


def test_parse_verify():
    args = cli.build_parser().parse_args(["verify", "--in", "stage3", "--query", "a cat", "--out", "stage4"])
    assert args.command == "verify"
    assert args.in_dir == Path("stage3")
    assert args.query == "a cat"
    assert args.out == Path("stage4")


def test_parse_review():
    args = cli.build_parser().parse_args(["review", "--in", "stage4", "--query", "a cat", "--out", "stage5"])
    assert args.command == "review"
    assert args.in_dir == Path("stage4")
    assert args.query == "a cat"
    assert args.out == Path("stage5")


def test_parse_export():
    args = cli.build_parser().parse_args(["export", "--in", "stage5", "--out", "output1"])
    assert args.command == "export"
    assert args.in_dir == Path("stage5")
    assert args.out == Path("output1")


def test_parse_run():
    args = cli.build_parser().parse_args(["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1"])
    assert args.command == "run"
    assert args.video == Path("clip.mp4")
    assert args.query == "a cat"
    assert args.out == Path("output1")
    assert args.mask_regions is None
    assert args.auto_mask is False


def test_parse_run_mask_regions():
    args = cli.build_parser().parse_args(
        ["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1", "--mask-regions", "0,60,750,75"]
    )
    assert args.mask_regions == [(0, 60, 750, 75)]


def test_parse_run_auto_mask():
    args = cli.build_parser().parse_args(
        ["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1", "--auto-mask"]
    )
    assert args.auto_mask is True
    assert args.mask_regions is None


def test_parse_missing_required_flag_raises_system_exit():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["extract", "--video", "clip.mp4"])  # missing --out


# --- dispatch ---


def test_main_dispatches_extract():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1"])
    mock_extract.assert_called_once()
    call_args = mock_extract.call_args[0]
    assert call_args[0] == Path("clip.mp4")
    assert call_args[1] == Path("out1")
    assert call_args[2].mask_regions is None
    assert call_args[2].run_auto_detect is False


def test_main_dispatches_extract_with_mask_regions():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1", "--mask-regions", "0,60,750,75"])
    mock_extract.assert_called_once()
    call_args = mock_extract.call_args[0]
    assert call_args[0] == Path("clip.mp4")
    assert call_args[1] == Path("out1")
    assert call_args[2].mask_regions == [(0, 60, 750, 75)]


def test_main_dispatches_extract_with_auto_mask():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1", "--auto-mask"])
    mock_extract.assert_called_once()
    call_args = mock_extract.call_args[0]
    assert call_args[2].mask_regions is None
    assert call_args[2].run_auto_detect is True


def test_main_dispatches_dedup():
    with patch("backend.cli.stage2_dedup.dedup") as mock_dedup:
        cli.main(["dedup", "--in", "stage1", "--out", "stage2"])
    mock_dedup.assert_called_once_with(Path("stage1"), Path("stage2"), None)


def test_main_dispatches_rank():
    with patch("backend.cli.stage3_rank.rank") as mock_rank:
        cli.main(["rank", "--in", "stage2", "--query", "a cat", "--out", "stage3"])
    mock_rank.assert_called_once_with(Path("stage2"), Path("stage3"), "a cat", None)


def test_main_dispatches_verify():
    with patch("backend.cli.stage4_verify.verify") as mock_verify:
        cli.main(["verify", "--in", "stage3", "--query", "a cat", "--out", "stage4"])
    mock_verify.assert_called_once_with(Path("stage3"), Path("stage4"), "a cat", None)


# --- dedup/rank/verify production-speed knobs ---


def test_parse_dedup_knobs_default_none():
    args = cli.build_parser().parse_args(["dedup", "--in", "stage1", "--out", "stage2"])
    assert args.dedup_hamming_threshold is None
    assert args.dedup_window_size is None


def test_main_dispatches_dedup_with_knobs():
    with patch("backend.cli.stage2_dedup.dedup") as mock_dedup:
        cli.main(
            [
                "dedup", "--in", "stage1", "--out", "stage2",
                "--dedup-hamming-threshold", "12", "--dedup-window-size", "10",
            ]
        )
    mock_dedup.assert_called_once()
    call_args = mock_dedup.call_args[0]
    assert call_args[2].hamming_threshold == 12
    assert call_args[2].window_size == 10


def test_parse_rank_top_k_default_none():
    args = cli.build_parser().parse_args(["rank", "--in", "stage2", "--query", "a cat", "--out", "stage3"])
    assert args.top_k is None


def test_main_dispatches_rank_with_top_k():
    with patch("backend.cli.stage3_rank.rank") as mock_rank:
        cli.main(["rank", "--in", "stage2", "--query", "a cat", "--out", "stage3", "--top-k", "20"])
    mock_rank.assert_called_once()
    call_args = mock_rank.call_args[0]
    assert call_args[3].top_k == 20


def test_parse_verify_skip_vlm_default_false():
    args = cli.build_parser().parse_args(["verify", "--in", "stage3", "--query", "a cat", "--out", "stage4"])
    assert args.skip_vlm is False


def test_main_dispatches_verify_with_skip_vlm():
    with patch("backend.cli.stage4_verify.verify") as mock_verify:
        cli.main(["verify", "--in", "stage3", "--query", "a cat", "--out", "stage4", "--skip-vlm"])
    mock_verify.assert_called_once()
    call_args = mock_verify.call_args[0]
    assert call_args[3].skip is True


def test_main_dispatches_review():
    with patch("backend.cli.stage5_review.review") as mock_review:
        cli.main(["review", "--in", "stage4", "--query", "a cat", "--out", "stage5"])
    mock_review.assert_called_once_with(Path("stage4"), Path("stage5"), "a cat")


def test_main_dispatches_run():
    with patch("backend.cli._run_pipeline") as mock_run:
        cli.main(["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1"])
    mock_run.assert_called_once_with(
        Path("clip.mp4"), "a cat", Path("output1"), None, False, None, None, None, None, None, None,
        None, None, None, False,
    )


def test_main_dispatches_run_with_mask_regions():
    with patch("backend.cli._run_pipeline") as mock_run:
        cli.main(
            ["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1", "--mask-regions", "0,60,750,75"]
        )
    mock_run.assert_called_once_with(
        Path("clip.mp4"), "a cat", Path("output1"), [(0, 60, 750, 75)], False, None, None, None, None, None, None,
        None, None, None, False,
    )


def test_main_dispatches_run_with_auto_mask():
    with patch("backend.cli._run_pipeline") as mock_run:
        cli.main(["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1", "--auto-mask"])
    mock_run.assert_called_once_with(
        Path("clip.mp4"), "a cat", Path("output1"), None, True, None, None, None, None, None, None,
        None, None, None, False,
    )


def test_main_dispatches_run_with_min_event_area_ratio():
    with patch("backend.cli._run_pipeline") as mock_run:
        cli.main(
            ["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1", "--min-event-area-ratio", "0.001"]
        )
    mock_run.assert_called_once_with(
        Path("clip.mp4"), "a cat", Path("output1"), None, False, 0.001, None, None, None, None, None,
        None, None, None, False,
    )


def test_main_dispatches_extract_with_min_event_area_ratio():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1", "--min-event-area-ratio", "0.001"])
    mock_extract.assert_called_once()
    call_args = mock_extract.call_args[0]
    assert call_args[2].min_blob_area_ratio == 0.001


def test_parse_extract_min_event_area_ratio_default_none():
    args = cli.build_parser().parse_args(["extract", "--video", "clip.mp4", "--out", "out1"])
    assert args.min_event_area_ratio is None


def test_parse_run_min_event_area_ratio():
    args = cli.build_parser().parse_args(
        ["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1", "--min-event-area-ratio", "0.002"]
    )
    assert args.min_event_area_ratio == 0.002


# --- --min/max-event-duration-sec ---


def test_parse_extract_event_duration_defaults_none():
    args = cli.build_parser().parse_args(["extract", "--video", "clip.mp4", "--out", "out1"])
    assert args.min_event_duration_sec is None
    assert args.max_event_duration_sec is None


def test_parse_run_event_duration():
    args = cli.build_parser().parse_args(
        [
            "run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1",
            "--min-event-duration-sec", "1", "--max-event-duration-sec", "3",
        ]
    )
    assert args.min_event_duration_sec == 1.0
    assert args.max_event_duration_sec == 3.0


def test_main_dispatches_extract_with_min_event_duration_sec():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1", "--min-event-duration-sec", "1"])
    mock_extract.assert_called_once()
    call_args = mock_extract.call_args[0]
    assert call_args[2].floor_interval_sec == pytest.approx(0.5)


def test_main_dispatches_extract_min_event_duration_sec_clamps_to_ceiling():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1", "--min-event-duration-sec", "15"])
    call_args = mock_extract.call_args[0]
    assert call_args[2].floor_interval_sec == pytest.approx(5.0)  # 15/2=7.5, clamped down


def test_main_dispatches_extract_no_min_event_duration_sec_keeps_default():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1"])
    call_args = mock_extract.call_args[0]
    assert call_args[2].floor_interval_sec == pytest.approx(5.0)  # untouched Stage1Config default


def test_main_dispatches_run_with_min_event_duration_sec():
    with patch("backend.cli._run_pipeline") as mock_run:
        cli.main(
            ["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1", "--min-event-duration-sec", "1"]
        )
    mock_run.assert_called_once_with(
        Path("clip.mp4"), "a cat", Path("output1"), None, False, None, 1.0, None, None, None, None,
        None, None, None, False,
    )


# --- --downscale-factor ---


def test_parse_extract_downscale_factor_default_none():
    args = cli.build_parser().parse_args(["extract", "--video", "clip.mp4", "--out", "out1"])
    assert args.downscale_factor is None


def test_parse_run_downscale_factor():
    args = cli.build_parser().parse_args(
        ["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1", "--downscale-factor", "0.5"]
    )
    assert args.downscale_factor == 0.5


def test_main_dispatches_extract_with_downscale_factor():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1", "--downscale-factor", "0.5"])
    mock_extract.assert_called_once()
    call_args = mock_extract.call_args[0]
    assert call_args[2].downscale_factor == 0.5


def test_main_dispatches_extract_no_downscale_factor_keeps_default():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1"])
    call_args = mock_extract.call_args[0]
    assert call_args[2].downscale_factor == 1.0


def test_main_dispatches_run_with_downscale_factor():
    with patch("backend.cli._run_pipeline") as mock_run:
        cli.main(
            ["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1", "--downscale-factor", "0.5"]
        )
    mock_run.assert_called_once_with(
        Path("clip.mp4"), "a cat", Path("output1"), None, False, None, None, None, 0.5, None, None,
        None, None, None, False,
    )


def test_extract_warns_when_min_exceeds_max_event_duration(capsys: pytest.CaptureFixture[str]):
    with patch("backend.cli.stage1_extract.extract"):
        cli.main(
            [
                "extract", "--video", "clip.mp4", "--out", "out1",
                "--min-event-duration-sec", "10", "--max-event-duration-sec", "3",
            ]
        )
    out = capsys.readouterr().out
    assert "warning" in out.lower()
    assert "10" in out and "3" in out


def test_extract_no_warning_when_bounds_consistent(capsys: pytest.CaptureFixture[str]):
    with patch("backend.cli.stage1_extract.extract"):
        cli.main(
            [
                "extract", "--video", "clip.mp4", "--out", "out1",
                "--min-event-duration-sec", "1", "--max-event-duration-sec", "3",
            ]
        )
    out = capsys.readouterr().out
    assert "warning" not in out.lower()


# --- --sampling-mode / --sample-interval-sec ---


def test_parse_extract_sampling_mode_defaults_none():
    args = cli.build_parser().parse_args(["extract", "--video", "clip.mp4", "--out", "out1"])
    assert args.sampling_mode is None
    assert args.sample_interval_sec is None


def test_parse_extract_sampling_mode_fixed_fps():
    args = cli.build_parser().parse_args(
        ["extract", "--video", "clip.mp4", "--out", "out1", "--sampling-mode", "fixed-fps"]
    )
    assert args.sampling_mode == "fixed-fps"


def test_parse_extract_sampling_mode_rejects_invalid_choice():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["extract", "--video", "clip.mp4", "--out", "out1", "--sampling-mode", "bogus"]
        )


def test_parse_run_sampling_mode_and_sample_interval():
    args = cli.build_parser().parse_args(
        [
            "run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1",
            "--sampling-mode", "fixed-fps", "--sample-interval-sec", "3",
        ]
    )
    assert args.sampling_mode == "fixed-fps"
    assert args.sample_interval_sec == 3.0


def test_main_dispatches_extract_with_sampling_mode():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1", "--sampling-mode", "fixed-fps"])
    call_args = mock_extract.call_args[0]
    assert call_args[2].sampling_mode == "fixed-fps"


def test_main_dispatches_extract_no_sampling_mode_keeps_default():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1"])
    call_args = mock_extract.call_args[0]
    assert call_args[2].sampling_mode == "motion"


def test_main_dispatches_extract_with_sample_interval_sec():
    with patch("backend.cli.stage1_extract.extract") as mock_extract:
        cli.main(
            [
                "extract", "--video", "clip.mp4", "--out", "out1",
                "--sampling-mode", "fixed-fps", "--sample-interval-sec", "3",
            ]
        )
    call_args = mock_extract.call_args[0]
    assert call_args[2].sample_interval_sec == 3.0


def test_main_dispatches_run_with_sampling_mode():
    with patch("backend.cli._run_pipeline") as mock_run:
        cli.main(
            [
                "run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1",
                "--sampling-mode", "fixed-fps", "--sample-interval-sec", "3",
            ]
        )
    mock_run.assert_called_once_with(
        Path("clip.mp4"), "a cat", Path("output1"), None, False, None, None, None, None, "fixed-fps", 3.0,
        None, None, None, False,
    )


def test_extract_warns_when_sample_interval_given_without_fixed_fps_mode(capsys: pytest.CaptureFixture[str]):
    with patch("backend.cli.stage1_extract.extract"):
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1", "--sample-interval-sec", "3"])
    out = capsys.readouterr().out
    assert "warning" in out.lower()
    assert "--sample-interval-sec" in out


def test_extract_warns_when_min_event_duration_given_with_fixed_fps_mode(capsys: pytest.CaptureFixture[str]):
    with patch("backend.cli.stage1_extract.extract"):
        cli.main(
            [
                "extract", "--video", "clip.mp4", "--out", "out1",
                "--sampling-mode", "fixed-fps", "--min-event-duration-sec", "1",
            ]
        )
    out = capsys.readouterr().out
    assert "warning" in out.lower()
    assert "--min-event-duration-sec" in out


def test_extract_no_sampling_mode_warning_when_flags_consistent(capsys: pytest.CaptureFixture[str]):
    with patch("backend.cli.stage1_extract.extract"):
        cli.main(
            [
                "extract", "--video", "clip.mp4", "--out", "out1",
                "--sampling-mode", "fixed-fps", "--sample-interval-sec", "3",
            ]
        )
    out = capsys.readouterr().out
    assert "warning" not in out.lower()


def test_run_pipeline_passes_sampling_mode_to_extract_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir, sampling_mode="fixed-fps", sample_interval_sec=3.0)

    extract_cmd = calls_log[0]
    assert "--sampling-mode" in extract_cmd
    assert extract_cmd[extract_cmd.index("--sampling-mode") + 1] == "fixed-fps"
    assert "--sample-interval-sec" in extract_cmd
    assert extract_cmd[extract_cmd.index("--sample-interval-sec") + 1] == "3.0"


def test_run_pipeline_no_sampling_mode_flags_when_none_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir)

    assert "--sampling-mode" not in calls_log[0]
    assert "--sample-interval-sec" not in calls_log[0]


# --- production-speed knobs: --dedup-hamming-threshold / --dedup-window-size / --top-k / --skip-vlm on run ---


def test_parse_run_production_knobs_default_none():
    args = cli.build_parser().parse_args(["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1"])
    assert args.dedup_hamming_threshold is None
    assert args.dedup_window_size is None
    assert args.top_k is None
    assert args.skip_vlm is False


def test_parse_run_production_knobs():
    args = cli.build_parser().parse_args(
        [
            "run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1",
            "--dedup-hamming-threshold", "12", "--dedup-window-size", "10", "--top-k", "20", "--skip-vlm",
        ]
    )
    assert args.dedup_hamming_threshold == 12
    assert args.dedup_window_size == 10
    assert args.top_k == 20
    assert args.skip_vlm is True


def test_main_dispatches_run_with_production_knobs():
    with patch("backend.cli._run_pipeline") as mock_run:
        cli.main(
            [
                "run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1",
                "--dedup-hamming-threshold", "12", "--dedup-window-size", "10", "--top-k", "20", "--skip-vlm",
            ]
        )
    mock_run.assert_called_once_with(
        Path("clip.mp4"), "a cat", Path("output1"), None, False, None, None, None, None, None, None,
        12, 10, 20, True,
    )


def test_run_pipeline_passes_dedup_knobs_to_dedup_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir, dedup_hamming_threshold=12, dedup_window_size=10)

    dedup_cmd = calls_log[1]
    assert "--dedup-hamming-threshold" in dedup_cmd
    assert dedup_cmd[dedup_cmd.index("--dedup-hamming-threshold") + 1] == "12"
    assert "--dedup-window-size" in dedup_cmd
    assert dedup_cmd[dedup_cmd.index("--dedup-window-size") + 1] == "10"


def test_run_pipeline_no_dedup_knob_flags_when_none_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir)

    assert "--dedup-hamming-threshold" not in calls_log[1]
    assert "--dedup-window-size" not in calls_log[1]


def test_run_pipeline_passes_top_k_to_rank_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir, top_k=20)

    rank_cmd = calls_log[2]
    assert "--top-k" in rank_cmd
    assert rank_cmd[rank_cmd.index("--top-k") + 1] == "20"


def test_run_pipeline_no_top_k_flag_when_none_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir)

    assert "--top-k" not in calls_log[2]


def test_run_pipeline_passes_skip_vlm_to_verify_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir, skip_vlm=True)

    assert "--skip-vlm" in calls_log[3]


def test_run_pipeline_no_skip_vlm_flag_when_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir)

    assert "--skip-vlm" not in calls_log[3]


def test_run_pipeline_prints_skip_vlm_stage4_announcement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    def side_effect(cmd, check=True):
        subcommand = cmd[3]
        flags = _parse_cmd_flags(cmd)
        out_dir_arg = Path(flags["out"])
        if subcommand == "extract":
            models.save_candidates(_fake_candidates(3, out_dir_arg), out_dir_arg / "candidates.json")
        elif subcommand in ("dedup", "rank"):
            models.save_candidates(_fake_candidates(2, out_dir_arg), out_dir_arg / "candidates.json")
        elif subcommand == "verify":
            models.save_candidates(_fake_verified(2, out_dir_arg, ["skipped", "skipped"]), out_dir_arg / "candidates.json")
        elif subcommand == "review":
            models.save_candidates(_fake_decisions(1, out_dir_arg, ["keep"]), out_dir_arg / "candidates.json")
        return MagicMock(returncode=0)

    with patch("backend.cli.subprocess.run", side_effect=side_effect):
        cli._run_pipeline(video_path, "a cat", out_dir, skip_vlm=True)

    out = capsys.readouterr().out
    assert "[run] stage 4/5: verify -- SKIPPED (--skip-vlm)" in out
    assert "stage 4 done: 2 -> 2 verified (0 yes, 0 no, 0 error, 2 skipped)" in out


def test_run_pipeline_production_speed_flags_compose_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Integration check for Part 2, item 3: several production-speed flags
    # chained together in one _run_pipeline call, confirming each reaches
    # its own subprocess without one flag silently overriding another.
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(
            video_path, "a cat", out_dir,
            downscale_factor=0.25,
            auto_mask=True,
            min_event_area_ratio=0.008,
            dedup_hamming_threshold=12,
            top_k=20,
            skip_vlm=True,
        )

    extract_cmd, dedup_cmd, rank_cmd, verify_cmd = calls_log[0], calls_log[1], calls_log[2], calls_log[3]
    assert extract_cmd[extract_cmd.index("--downscale-factor") + 1] == "0.25"
    assert "--auto-mask" in extract_cmd
    assert extract_cmd[extract_cmd.index("--min-event-area-ratio") + 1] == "0.008"
    assert dedup_cmd[dedup_cmd.index("--dedup-hamming-threshold") + 1] == "12"
    assert "--dedup-window-size" not in dedup_cmd  # not passed -- must not leak a value from another flag
    assert rank_cmd[rank_cmd.index("--top-k") + 1] == "20"
    assert "--skip-vlm" in verify_cmd


def test_main_dispatches_export():
    with patch("backend.cli.export.export") as mock_export:
        cli.main(["export", "--in", "stage5", "--out", "output1"])
    mock_export.assert_called_once_with(Path("stage5"), Path("output1"))


# --- "build once, search many times": index / search / search-index ---


def test_parse_index():
    args = cli.build_parser().parse_args(["index", "--video", "clip.mp4", "--out", "idx1"])
    assert args.command == "index"
    assert args.video == Path("clip.mp4")
    assert args.out == Path("idx1")
    # every extract flag reachable on index
    assert args.mask_regions is None
    assert args.auto_mask is False
    assert args.min_event_area_ratio is None
    assert args.min_event_duration_sec is None
    assert args.downscale_factor is None
    assert args.sampling_mode is None
    # every dedup flag reachable on index
    assert args.dedup_hamming_threshold is None
    assert args.dedup_window_size is None


def test_parse_index_with_flags():
    args = cli.build_parser().parse_args(
        [
            "index", "--video", "clip.mp4", "--out", "idx1",
            "--downscale-factor", "0.5", "--auto-mask", "--dedup-hamming-threshold", "12",
        ]
    )
    assert args.downscale_factor == 0.5
    assert args.auto_mask is True
    assert args.dedup_hamming_threshold == 12


def test_main_dispatches_index():
    with patch("backend.cli._index_pipeline") as mock_index:
        cli.main(["index", "--video", "clip.mp4", "--out", "idx1", "--downscale-factor", "0.5"])
    mock_index.assert_called_once_with(
        Path("clip.mp4"), Path("idx1"), None, False, None, None, None, 0.5, None, None, None, None
    )


def test_parse_search():
    args = cli.build_parser().parse_args(["search", "--index", "idx1", "--query", "a cat", "--out", "output1"])
    assert args.command == "search"
    assert args.index_dir == Path("idx1")
    assert args.query == "a cat"
    assert args.out == Path("output1")
    assert args.top_k is None
    assert args.skip_vlm is False


def test_main_dispatches_search():
    with patch("backend.cli._search_pipeline") as mock_search:
        cli.main(
            [
                "search", "--index", "idx1", "--query", "a cat", "--out", "output1",
                "--top-k", "10", "--skip-vlm",
            ]
        )
    mock_search.assert_called_once_with(Path("idx1"), "a cat", Path("output1"), 10, True)


def test_parse_search_index():
    args = cli.build_parser().parse_args(
        ["search-index", "--index", "idx1", "--query", "a cat", "--out", "stage3"]
    )
    assert args.command == "search-index"
    assert args.index_dir == Path("idx1")
    assert args.query == "a cat"
    assert args.out == Path("stage3")
    assert args.top_k is None


def test_main_dispatches_search_index():
    with patch("backend.cli.stage3_rank.search_index") as mock_search_index:
        cli.main(["search-index", "--index", "idx1", "--query", "a cat", "--out", "stage3", "--top-k", "10"])
    mock_search_index.assert_called_once()
    call_args = mock_search_index.call_args[0]
    assert call_args[0] == Path("idx1")
    assert call_args[1] == Path("stage3")
    assert call_args[2] == "a cat"
    assert call_args[3].top_k == 10


def test_main_dispatches_search_index_no_top_k_keeps_default():
    with patch("backend.cli.stage3_rank.search_index") as mock_search_index:
        cli.main(["search-index", "--index", "idx1", "--query", "a cat", "--out", "stage3"])
    call_args = mock_search_index.call_args[0]
    assert call_args[3] is None  # rank_config stays None -> search_index() uses Stage3Config() default


# --- _index_pipeline orchestration, mocked stage functions (in-process, no subprocess) ---


def test_index_pipeline_orchestrates_stages_in_order_and_writes_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "idx1"

    calls_log: list[str] = []

    def fake_extract(video, out, config):
        calls_log.append("extract")
        models.save_candidates(_fake_candidates(3, out), out / "candidates.json")

    def fake_dedup(in_dir, out, config):
        calls_log.append("dedup")
        assert in_dir == out_dir / "stage1"
        models.save_candidates(_fake_candidates(2, out), out / "candidates.json")

    def fake_build_index(dedup_dir, config=None):
        calls_log.append("build_index")
        assert dedup_dir == out_dir

    with patch("backend.cli.stage1_extract.extract", side_effect=fake_extract), patch(
        "backend.cli.stage2_dedup.dedup", side_effect=fake_dedup
    ), patch("backend.cli.stage3_rank.build_index", side_effect=fake_build_index):
        cli._index_pipeline(video_path, out_dir)

    assert calls_log == ["extract", "dedup", "build_index"]

    timing = json.loads((out_dir / "timing.json").read_text())
    assert set(timing.keys()) == {"stage1", "stage2", "stage3"}


def test_index_pipeline_dedup_writes_directly_into_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # build_index()'s sidecar ends up directly in `out` (dedup's own output
    # dir) -- NOT a nested "stage2" subdirectory -- so `search --index`
    # is given the exact same path `index --out` was.
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "idx1"

    def fake_extract(video, out, config):
        models.save_candidates(_fake_candidates(3, out), out / "candidates.json")

    dedup_out_dirs = []

    def fake_dedup(in_dir, out, config):
        dedup_out_dirs.append(out)
        models.save_candidates(_fake_candidates(2, out), out / "candidates.json")

    with patch("backend.cli.stage1_extract.extract", side_effect=fake_extract), patch(
        "backend.cli.stage2_dedup.dedup", side_effect=fake_dedup
    ), patch("backend.cli.stage3_rank.build_index"):
        cli._index_pipeline(video_path, out_dir)

    assert dedup_out_dirs == [out_dir]


def test_index_pipeline_prints_funnel_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "idx1"

    def fake_extract(video, out, config):
        models.save_candidates(_fake_candidates(3, out), out / "candidates.json")

    def fake_dedup(in_dir, out, config):
        models.save_candidates(_fake_candidates(2, out), out / "candidates.json")

    with patch("backend.cli.stage1_extract.extract", side_effect=fake_extract), patch(
        "backend.cli.stage2_dedup.dedup", side_effect=fake_dedup
    ), patch("backend.cli.stage3_rank.build_index"):
        cli._index_pipeline(video_path, out_dir)

    out = capsys.readouterr().out
    assert "stage 1 done: 3 candidates" in out
    assert "stage 2 done: 3 -> 2 candidates" in out
    assert "stage 3 done: 2 candidates embedded" in out
    assert f"search --index {out_dir}" in out


# --- _search_pipeline orchestration, mocked subprocess.run ---


def test_search_pipeline_orchestrates_stages_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / "idx1"
    index_dir.mkdir()
    out_dir = tmp_path / "data" / "output" / "search1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._search_pipeline(index_dir, "a cat", out_dir)

    subcommands = [cmd[3] for cmd in calls_log]
    assert subcommands == ["search-index", "verify", "review", "export"]


def test_search_pipeline_passes_index_dir_and_query_to_search_index_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / "idx1"
    index_dir.mkdir()
    out_dir = tmp_path / "data" / "output" / "search1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._search_pipeline(index_dir, "a cat", out_dir)

    search_cmd = calls_log[0]
    assert "--index" in search_cmd
    assert search_cmd[search_cmd.index("--index") + 1] == str(index_dir)
    assert "--query" in search_cmd
    assert search_cmd[search_cmd.index("--query") + 1] == "a cat"


def test_search_pipeline_passes_top_k_to_search_index_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / "idx1"
    index_dir.mkdir()
    out_dir = tmp_path / "data" / "output" / "search1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._search_pipeline(index_dir, "a cat", out_dir, top_k=10)

    search_cmd = calls_log[0]
    assert "--top-k" in search_cmd
    assert search_cmd[search_cmd.index("--top-k") + 1] == "10"


def test_search_pipeline_passes_skip_vlm_to_verify_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / "idx1"
    index_dir.mkdir()
    out_dir = tmp_path / "data" / "output" / "search1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._search_pipeline(index_dir, "a cat", out_dir, skip_vlm=True)

    assert "--skip-vlm" in calls_log[1]


def test_search_pipeline_writes_timing_json_without_stage1_or_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / "idx1"
    index_dir.mkdir()
    out_dir = tmp_path / "data" / "output" / "search1"

    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect([])):
        cli._search_pipeline(index_dir, "a cat", out_dir)

    timing_path = Path("data") / "work" / "search1" / "timing.json"
    timing = json.loads(timing_path.read_text())
    assert set(timing.keys()) == {"stage3", "stage4", "stage5"}


def test_search_pipeline_never_touches_stage1_or_2_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / "idx1"
    index_dir.mkdir()
    out_dir = tmp_path / "data" / "output" / "search1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._search_pipeline(index_dir, "a cat", out_dir)

    subcommands = {cmd[3] for cmd in calls_log}
    assert "extract" not in subcommands
    assert "dedup" not in subcommands


# --- _run_pipeline orchestration, mocked subprocess.run ---


def _fake_candidates(count: int, out_dir: Path) -> list[Candidate]:
    items = []
    for i in range(count):
        path = out_dir / f"frame_{i:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
        items.append(Candidate(frame_index=i, timestamp_ms=i * 100.0, image_path=path, reason="floor"))
    return items


def _fake_verified(count: int, out_dir: Path, verdicts: list[str]) -> list[VerifiedFrame]:
    items = []
    for i in range(count):
        path = out_dir / f"frame_{i:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
        items.append(
            VerifiedFrame(
                frame_index=i,
                timestamp_ms=i * 100.0,
                image_path=path,
                reason="floor",
                verdict=verdicts[i],
                reasoning="r",
                confidence=None if verdicts[i] == "error" else 0.9,
            )
        )
    return items


def _fake_decisions(count: int, out_dir: Path, decisions: list[str]) -> list[ReviewDecision]:
    items = []
    for i in range(count):
        path = out_dir / f"frame_{i:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
        items.append(
            ReviewDecision(
                frame_index=i,
                timestamp_ms=i * 100.0,
                image_path=path,
                reason="floor",
                verdict="yes",
                reasoning="r",
                confidence=0.9,
                decision=decisions[i],
            )
        )
    return items


def _parse_cmd_flags(cmd: list[str]) -> dict[str, str | bool]:
    flags: dict[str, str | bool] = {}
    tokens = cmd[4:]  # skip [sys.executable, "-m", "backend", subcommand]
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            # Boolean flags (e.g. --auto-mask) take no value -- distinguish
            # from a value-taking flag by checking whether the next token is
            # itself a flag (or there is no next token at all).
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                flags[token[2:]] = tokens[i + 1]
                i += 2
            else:
                flags[token[2:]] = True
                i += 1
        else:
            i += 1
    return flags


def _make_subprocess_side_effect(calls_log: list[list[str]]):
    def side_effect(cmd, check=True):
        calls_log.append(cmd)
        subcommand = cmd[3]
        flags = _parse_cmd_flags(cmd)
        out_dir = Path(flags["out"])

        if subcommand == "extract":
            models.save_candidates(_fake_candidates(3, out_dir), out_dir / "candidates.json")
        elif subcommand == "dedup":
            models.save_candidates(_fake_candidates(2, out_dir), out_dir / "candidates.json")
        elif subcommand in ("rank", "search-index"):
            models.save_candidates(_fake_candidates(2, out_dir), out_dir / "candidates.json")
        elif subcommand == "verify":
            models.save_candidates(_fake_verified(2, out_dir, ["yes", "no"]), out_dir / "candidates.json")
        elif subcommand == "review":
            models.save_candidates(_fake_decisions(1, out_dir, ["keep"]), out_dir / "candidates.json")
        elif subcommand == "export":
            pass
        return MagicMock(returncode=0)

    return side_effect


def test_run_pipeline_orchestrates_all_stages_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir)

    subcommands = [cmd[3] for cmd in calls_log]
    assert subcommands == ["extract", "dedup", "rank", "verify", "review", "export"]

    # run_id derivation: work dirs under data/work/<out.name>/stageN, relative
    # to CWD (not resolved to absolute) -- matches SPEC.md's own relative-path
    # CLI examples. monkeypatch.chdir(tmp_path) above makes CWD == tmp_path,
    # but Path("data/work/...") itself stays relative either way.
    expected_stage1_out = Path("data") / "work" / "run1" / "stage1"
    assert _parse_cmd_flags(calls_log[0])["out"] == str(expected_stage1_out)
    expected_stage5_out = Path("data") / "work" / "run1" / "stage5"
    assert _parse_cmd_flags(calls_log[4])["out"] == str(expected_stage5_out)
    assert _parse_cmd_flags(calls_log[5])["out"] == str(out_dir)  # export writes to the real --out


def test_run_pipeline_passes_mask_regions_to_extract_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir, mask_regions=[(0, 60, 750, 75)])

    extract_cmd = calls_log[0]
    assert "--mask-regions" in extract_cmd
    idx = extract_cmd.index("--mask-regions")
    assert extract_cmd[idx + 1] == "0,60,750,75"


def test_run_pipeline_no_mask_regions_flag_when_none_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir)

    assert "--mask-regions" not in calls_log[0]


def test_run_pipeline_passes_auto_mask_flag_to_extract_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir, auto_mask=True)

    assert "--auto-mask" in calls_log[0]


def test_run_pipeline_no_auto_mask_flag_when_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir)

    assert "--auto-mask" not in calls_log[0]


def test_run_pipeline_passes_min_event_duration_sec_to_extract_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir, min_event_duration_sec=1.0, max_event_duration_sec=3.0)

    extract_cmd = calls_log[0]
    assert "--min-event-duration-sec" in extract_cmd
    assert extract_cmd[extract_cmd.index("--min-event-duration-sec") + 1] == "1.0"
    assert "--max-event-duration-sec" in extract_cmd
    assert extract_cmd[extract_cmd.index("--max-event-duration-sec") + 1] == "3.0"


def test_run_pipeline_no_event_duration_flags_when_none_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir)

    assert "--min-event-duration-sec" not in calls_log[0]
    assert "--max-event-duration-sec" not in calls_log[0]


def test_run_pipeline_passes_downscale_factor_to_extract_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir, downscale_factor=0.5)

    extract_cmd = calls_log[0]
    assert "--downscale-factor" in extract_cmd
    assert extract_cmd[extract_cmd.index("--downscale-factor") + 1] == "0.5"


def test_run_pipeline_no_downscale_factor_flag_when_none_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []
    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
        cli._run_pipeline(video_path, "a cat", out_dir)

    assert "--downscale-factor" not in calls_log[0]


def test_run_pipeline_prints_funnel_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect([])):
        cli._run_pipeline(video_path, "a cat", out_dir)

    out = capsys.readouterr().out
    assert "stage 1 done: 3 candidates" in out
    assert "stage 2 done: 3 -> 2 candidates" in out
    assert "stage 3 done: 2 -> 2 candidates" in out
    assert "stage 4 done: 2 -> 2 verified (1 yes, 1 no, 0 error, 0 skipped)" in out
    assert "stage 5 done: 1 kept, 0 discarded" in out
    assert "[run] export -- writing final output to" in out
    assert "stage 6" not in out  # export is not a numbered stage


def test_run_pipeline_writes_timing_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    with patch("backend.cli.subprocess.run", side_effect=_make_subprocess_side_effect([])):
        cli._run_pipeline(video_path, "a cat", out_dir)

    timing_path = Path("data") / "work" / "run1" / "timing.json"
    assert timing_path.exists()
    timing = json.loads(timing_path.read_text())
    assert set(timing.keys()) == {"stage1", "stage2", "stage3", "stage4", "stage5"}
    assert all(isinstance(v, (int, float)) and v >= 0 for v in timing.values())


def test_run_pipeline_timing_json_survives_partial_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    def failing_side_effect(cmd, check=True):
        subcommand = cmd[3]
        flags = _parse_cmd_flags(cmd)
        if subcommand == "extract":
            out_dir_arg = Path(flags["out"])
            models.save_candidates(_fake_candidates(3, out_dir_arg), out_dir_arg / "candidates.json")
            return MagicMock(returncode=0)
        raise subprocess.CalledProcessError(1, cmd)

    with patch("backend.cli.subprocess.run", side_effect=failing_side_effect):
        with pytest.raises(subprocess.CalledProcessError):
            cli._run_pipeline(video_path, "a cat", out_dir)

    timing_path = Path("data") / "work" / "run1" / "timing.json"
    assert timing_path.exists()
    timing = json.loads(timing_path.read_text())
    assert set(timing.keys()) == {"stage1"}  # only the completed stage's timing survives


def test_run_pipeline_stops_on_subprocess_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    calls_log: list[list[str]] = []

    def failing_side_effect(cmd, check=True):
        calls_log.append(cmd)
        subcommand = cmd[3]
        flags = _parse_cmd_flags(cmd)
        if subcommand == "extract":
            out_dir_arg = Path(flags["out"])
            models.save_candidates(_fake_candidates(3, out_dir_arg), out_dir_arg / "candidates.json")
            return MagicMock(returncode=0)
        if subcommand == "dedup":
            raise subprocess.CalledProcessError(1, cmd)
        raise AssertionError(f"should not reach subcommand {subcommand!r} after dedup failed")

    with patch("backend.cli.subprocess.run", side_effect=failing_side_effect):
        with pytest.raises(subprocess.CalledProcessError):
            cli._run_pipeline(video_path, "a cat", out_dir)

    assert [cmd[3] for cmd in calls_log] == ["extract", "dedup"]
