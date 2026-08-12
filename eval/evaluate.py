"""Scores a completed pipeline run against eval/ground_truth.json, per
SPEC.md's end-to-end verification section.

Reads stage 5's manifest directly (has timestamp_ms already, unlike the
COCO export which strips it), filters to decision == "keep", and matches
each kept frame's timestamp against ground-truth event windows (with a
tolerance, since a real event rarely lands on an exact frame).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from frames_extractor import io_utils, models

# Tolerance around each ground-truth event's [start_ts, end_ts] window used
# to match a kept frame's timestamp_ms. Named, not a magic number, since it
# materially changes the recall/precision numbers -- widening it makes
# recall easier and precision harder, and vice versa.
DEFAULT_TOLERANCE_MS = 2000.0


@dataclass(kw_only=True)
class GroundTruthEvent:
    start_ts: float
    end_ts: float
    description: str


@dataclass(kw_only=True)
class EvaluationReport:
    recall: float
    precision: float
    events_total: int
    events_recalled: int
    kept_total: int
    kept_hits: int
    tolerance_ms: float
    funnel: dict[str, int | None]
    stage_elapsed_sec: dict[str, float]
    stage_elapsed_is_approximate: bool
    total_elapsed_sec: float
    kept_no_verdict_frame_indices: list[int]


def load_ground_truth(path: Path) -> list[GroundTruthEvent]:
    return [GroundTruthEvent(**d) for d in json.loads(path.read_text())]


def _within_any_event(timestamp_ms: float, events: list[GroundTruthEvent], tolerance_ms: float) -> bool:
    return any(event.start_ts - tolerance_ms <= timestamp_ms <= event.end_ts + tolerance_ms for event in events)


def _reconstruct_stage_elapsed_from_timestamps(work_dir: Path) -> dict[str, float]:
    """Best-effort fallback when timing.json isn't present (e.g. a run from
    before this was added, or individual stage subcommands run by hand):
    each stage directory's creation time -> its own candidates.json mtime.

    This is only an approximation, and it degrades silently if a manifest is
    edited after the stage actually completed (e.g. a manual correction to a
    decision) -- the reconstructed "elapsed time" then includes however long
    passed until that edit, not the real processing time. Caller-visible via
    stage_elapsed_is_approximate; there's no way to detect this corruption
    from the file timestamps alone, so it's flagged unconditionally whenever
    this fallback path is used at all, not just when corruption is detected.
    """
    elapsed: dict[str, float] = {}
    for n in range(1, 6):
        stage_dir = work_dir / f"stage{n}"
        manifest = stage_dir / "candidates.json"
        if not stage_dir.exists() or not manifest.exists():
            continue
        elapsed[f"stage{n}"] = os.path.getmtime(manifest) - os.path.getctime(stage_dir)
    return elapsed


def _load_stage_elapsed(work_dir: Path) -> tuple[dict[str, float], bool]:
    timing_path = work_dir / "timing.json"
    if timing_path.exists():
        return json.loads(timing_path.read_text()), False
    return _reconstruct_stage_elapsed_from_timestamps(work_dir), True


def evaluate(
    work_dir: Path,
    ground_truth_path: Path,
    video_path: Path | None = None,
    tolerance_ms: float = DEFAULT_TOLERANCE_MS,
) -> EvaluationReport:
    events = load_ground_truth(ground_truth_path)

    stage1 = models.load_candidates(work_dir / "stage1" / "candidates.json")
    stage2 = models.load_candidates(work_dir / "stage2" / "candidates.json")
    stage3 = models.load_candidates(work_dir / "stage3" / "candidates.json")
    stage4 = models.load_verified_frames(work_dir / "stage4" / "candidates.json")
    stage5 = models.load_review_decisions(work_dir / "stage5" / "candidates.json")

    kept = [d for d in stage5 if d.decision == "keep"]

    events_recalled = sum(
        1
        for event in events
        if any(
            event.start_ts - tolerance_ms <= frame.timestamp_ms <= event.end_ts + tolerance_ms for frame in kept
        )
    )
    recall = events_recalled / len(events) if events else 0.0

    kept_hits = sum(1 for frame in kept if _within_any_event(frame.timestamp_ms, events, tolerance_ms))
    precision = kept_hits / len(kept) if kept else 0.0

    raw_frame_count = None
    if video_path is not None:
        cap = io_utils.open_video(video_path)
        try:
            raw_frame_count = io_utils.get_video_metadata(cap).frame_count
        finally:
            cap.release()

    funnel: dict[str, int | None] = {
        "raw": raw_frame_count,
        "stage1": len(stage1),
        "stage2": len(stage2),
        "stage3": len(stage3),
        "stage4_verified": len(stage4),
        "stage5_kept": len(kept),
    }

    stage_elapsed, is_approximate = _load_stage_elapsed(work_dir)
    total_elapsed_sec = sum(stage_elapsed.values())

    kept_no_verdict_frame_indices = sorted(d.frame_index for d in kept if d.verdict == "no")

    return EvaluationReport(
        recall=recall,
        precision=precision,
        events_total=len(events),
        events_recalled=events_recalled,
        kept_total=len(kept),
        kept_hits=kept_hits,
        tolerance_ms=tolerance_ms,
        funnel=funnel,
        stage_elapsed_sec=stage_elapsed,
        stage_elapsed_is_approximate=is_approximate,
        total_elapsed_sec=total_elapsed_sec,
        kept_no_verdict_frame_indices=kept_no_verdict_frame_indices,
    )


def print_report(report: EvaluationReport) -> None:
    print(f"Recall:    {report.events_recalled}/{report.events_total} = {report.recall:.1%}")
    print(f"Precision: {report.kept_hits}/{report.kept_total} = {report.precision:.1%}")
    print(f"(tolerance_ms = {report.tolerance_ms:.0f})")

    print("\nFunnel:")
    for stage, count in report.funnel.items():
        print(f"  {stage}: {count if count is not None else 'n/a (no --video given)'}")

    approx_note = (
        " -- APPROXIMATE: reconstructed from file timestamps (no timing.json found); "
        "invalidated by any manifest edit made after a stage actually completed"
        if report.stage_elapsed_is_approximate
        else ""
    )
    print(f"\nStage elapsed times{approx_note}:")
    for stage, secs in report.stage_elapsed_sec.items():
        print(f"  {stage}: {secs:.1f}s")
    print(f"  total: {report.total_elapsed_sec:.1f}s ({report.total_elapsed_sec / 60:.1f} min)")

    print("\nDiagnostic -- kept frames where the VLM said \"no\" (human overrode the model):")
    if report.kept_no_verdict_frame_indices:
        print(f"  {len(report.kept_no_verdict_frame_indices)} frame(s): {report.kept_no_verdict_frame_indices}")
    else:
        print("  none")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="e.g. data/work/run1")
    parser.add_argument("--ground-truth", type=Path, default=Path(__file__).parent / "ground_truth.json")
    parser.add_argument("--video", type=Path, default=None, help="original video, for the raw frame count")
    parser.add_argument("--tolerance-ms", type=float, default=DEFAULT_TOLERANCE_MS)
    args = parser.parse_args(argv)

    report = evaluate(args.work_dir, args.ground_truth, args.video, args.tolerance_ms)
    print_report(report)


if __name__ == "__main__":
    main()
