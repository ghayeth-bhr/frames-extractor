"""Tests for cli.py: argument parsing, dispatch, and _run_pipeline's
orchestration. subprocess.run is always mocked here -- no real subprocess,
GPU, or Ollama call is ever spawned by this file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frames_extractor import cli, models
from frames_extractor.models import Candidate, ReviewDecision, VerifiedFrame

# --- argument parsing ---


def test_parse_extract():
    args = cli.build_parser().parse_args(["extract", "--video", "clip.mp4", "--out", "out1"])
    assert args.command == "extract"
    assert args.video == Path("clip.mp4")
    assert args.out == Path("out1")


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


def test_parse_missing_required_flag_raises_system_exit():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["extract", "--video", "clip.mp4"])  # missing --out


# --- dispatch ---


def test_main_dispatches_extract():
    with patch("frames_extractor.cli.stage1_extract.extract") as mock_extract:
        cli.main(["extract", "--video", "clip.mp4", "--out", "out1"])
    mock_extract.assert_called_once_with(Path("clip.mp4"), Path("out1"))


def test_main_dispatches_dedup():
    with patch("frames_extractor.cli.stage2_dedup.dedup") as mock_dedup:
        cli.main(["dedup", "--in", "stage1", "--out", "stage2"])
    mock_dedup.assert_called_once_with(Path("stage1"), Path("stage2"))


def test_main_dispatches_rank():
    with patch("frames_extractor.cli.stage3_rank.rank") as mock_rank:
        cli.main(["rank", "--in", "stage2", "--query", "a cat", "--out", "stage3"])
    mock_rank.assert_called_once_with(Path("stage2"), Path("stage3"), "a cat")


def test_main_dispatches_verify():
    with patch("frames_extractor.cli.stage4_verify.verify") as mock_verify:
        cli.main(["verify", "--in", "stage3", "--query", "a cat", "--out", "stage4"])
    mock_verify.assert_called_once_with(Path("stage3"), Path("stage4"), "a cat")


def test_main_dispatches_review():
    with patch("frames_extractor.cli.stage5_review.review") as mock_review:
        cli.main(["review", "--in", "stage4", "--query", "a cat", "--out", "stage5"])
    mock_review.assert_called_once_with(Path("stage4"), Path("stage5"), "a cat")


def test_main_dispatches_run():
    with patch("frames_extractor.cli._run_pipeline") as mock_run:
        cli.main(["run", "--video", "clip.mp4", "--query", "a cat", "--out", "output1"])
    mock_run.assert_called_once_with(Path("clip.mp4"), "a cat", Path("output1"))


def test_main_dispatches_export():
    with patch("frames_extractor.cli.export.export") as mock_export:
        cli.main(["export", "--in", "stage5", "--out", "output1"])
    mock_export.assert_called_once_with(Path("stage5"), Path("output1"))


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


def _parse_cmd_flags(cmd: list[str]) -> dict[str, str]:
    flags: dict[str, str] = {}
    args = iter(cmd[4:])  # skip [sys.executable, "-m", "frames_extractor", subcommand]
    for token in args:
        if token.startswith("--"):
            flags[token[2:]] = next(args)
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
        elif subcommand == "rank":
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
    with patch("frames_extractor.cli.subprocess.run", side_effect=_make_subprocess_side_effect(calls_log)):
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


def test_run_pipeline_prints_funnel_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")
    out_dir = tmp_path / "data" / "output" / "run1"

    with patch("frames_extractor.cli.subprocess.run", side_effect=_make_subprocess_side_effect([])):
        cli._run_pipeline(video_path, "a cat", out_dir)

    out = capsys.readouterr().out
    assert "stage 1 done: 3 candidates" in out
    assert "stage 2 done: 3 -> 2 candidates" in out
    assert "stage 3 done: 2 -> 2 candidates" in out
    assert "stage 4 done: 2 -> 2 verified (1 yes, 1 no, 0 error)" in out
    assert "stage 5 done: 1 kept, 0 discarded" in out
    assert "[run] export -- writing final output to" in out
    assert "stage 6" not in out  # export is not a numbered stage


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

    with patch("frames_extractor.cli.subprocess.run", side_effect=failing_side_effect):
        with pytest.raises(subprocess.CalledProcessError):
            cli._run_pipeline(video_path, "a cat", out_dir)

    assert [cmd[3] for cmd in calls_log] == ["extract", "dedup"]
