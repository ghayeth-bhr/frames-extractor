"""CLI entrypoints per SPEC.md's shown shape: run, extract, dedup, rank,
verify, review, export.

No CLI flags for internal stage tunables -- every stage runs with its
dataclass defaults (already recall-biased), matching SPEC.md's own shown
examples, which only ever pass --video/--in/--out/--query.

`run` chains all 5 stages plus export as separate subprocesses (each
invoked as `sys.executable -m frames_extractor <stage> ...`), not
in-process calls. A stage-3 (SigLIP2/PyTorch) subprocess exiting
guarantees the OS reclaims 100% of its VRAM before stage 4 (Ollama)
starts -- see SPEC.md/the approved plan for why explicit in-process
`torch.cuda.empty_cache()` cleanup was rejected: this project has hit
VRAM-contention crashes between these two GPU consumers three times, and
`empty_cache()` is a known-incomplete mitigation (fragmentation, CUDA
context overhead that only clears on process exit).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from . import export, models, stage1_extract, stage2_dedup, stage3_rank, stage4_verify, stage5_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frames_extractor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run")
    run_p.add_argument("--video", type=Path, required=True)
    run_p.add_argument("--query", type=str, required=True)
    run_p.add_argument("--out", type=Path, required=True)

    extract_p = subparsers.add_parser("extract")
    extract_p.add_argument("--video", type=Path, required=True)
    extract_p.add_argument("--out", type=Path, required=True)

    dedup_p = subparsers.add_parser("dedup")
    dedup_p.add_argument("--in", dest="in_dir", type=Path, required=True)
    dedup_p.add_argument("--out", type=Path, required=True)

    rank_p = subparsers.add_parser("rank")
    rank_p.add_argument("--in", dest="in_dir", type=Path, required=True)
    rank_p.add_argument("--query", type=str, required=True)
    rank_p.add_argument("--out", type=Path, required=True)

    verify_p = subparsers.add_parser("verify")
    verify_p.add_argument("--in", dest="in_dir", type=Path, required=True)
    verify_p.add_argument("--query", type=str, required=True)
    verify_p.add_argument("--out", type=Path, required=True)

    review_p = subparsers.add_parser("review")
    review_p.add_argument("--in", dest="in_dir", type=Path, required=True)
    review_p.add_argument("--query", type=str, required=True)
    review_p.add_argument("--out", type=Path, required=True)

    export_p = subparsers.add_parser("export")
    export_p.add_argument("--in", dest="in_dir", type=Path, required=True)
    export_p.add_argument("--out", type=Path, required=True)

    return parser


def _run_stage_subprocess(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "frames_extractor", *args], check=True)


def _run_pipeline(video: Path, query: str, out: Path) -> None:
    run_id = out.name
    work_dir = Path("data/work") / run_id
    stage_dirs = {n: work_dir / f"stage{n}" for n in range(1, 6)}

    print(f"[run] stage 1/5: extract -- {video}")
    t0 = time.time()
    _run_stage_subprocess(["extract", "--video", str(video), "--out", str(stage_dirs[1])])
    count1 = len(models.load_candidates(stage_dirs[1] / "candidates.json"))
    print(f"[run] stage 1 done: {count1} candidates ({time.time() - t0:.1f}s)")

    print("[run] stage 2/5: dedup")
    t0 = time.time()
    _run_stage_subprocess(["dedup", "--in", str(stage_dirs[1]), "--out", str(stage_dirs[2])])
    count2 = len(models.load_candidates(stage_dirs[2] / "candidates.json"))
    print(f"[run] stage 2 done: {count1} -> {count2} candidates ({time.time() - t0:.1f}s)")

    print("[run] stage 3/5: rank")
    t0 = time.time()
    _run_stage_subprocess(["rank", "--in", str(stage_dirs[2]), "--query", query, "--out", str(stage_dirs[3])])
    count3 = len(models.load_candidates(stage_dirs[3] / "candidates.json"))
    print(f"[run] stage 3 done: {count2} -> {count3} candidates ({time.time() - t0:.1f}s)")

    print("[run] stage 4/5: verify")
    t0 = time.time()
    _run_stage_subprocess(["verify", "--in", str(stage_dirs[3]), "--query", query, "--out", str(stage_dirs[4])])
    verified = models.load_verified_frames(stage_dirs[4] / "candidates.json")
    yes_count = sum(1 for v in verified if v.verdict == "yes")
    no_count = sum(1 for v in verified if v.verdict == "no")
    error_count = sum(1 for v in verified if v.verdict == "error")
    print(
        f"[run] stage 4 done: {count3} -> {len(verified)} verified "
        f"({yes_count} yes, {no_count} no, {error_count} error) ({time.time() - t0:.1f}s)"
    )

    print("[run] stage 5/5: review -- opens an interactive window, decide with k/d/space/q")
    _run_stage_subprocess(["review", "--in", str(stage_dirs[4]), "--query", query, "--out", str(stage_dirs[5])])
    decisions = models.load_review_decisions(stage_dirs[5] / "candidates.json")
    kept = sum(1 for d in decisions if d.decision == "keep")
    print(f"[run] stage 5 done: {kept} kept, {len(decisions) - kept} discarded")

    print(f"[run] export -- writing final output to {out}")
    _run_stage_subprocess(["export", "--in", str(stage_dirs[5]), "--out", str(out)])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "extract":
        stage1_extract.extract(args.video, args.out)
    elif args.command == "dedup":
        stage2_dedup.dedup(args.in_dir, args.out)
    elif args.command == "rank":
        stage3_rank.rank(args.in_dir, args.out, args.query)
    elif args.command == "verify":
        stage4_verify.verify(args.in_dir, args.out, args.query)
    elif args.command == "review":
        stage5_review.review(args.in_dir, args.out, args.query)
    elif args.command == "export":
        export.export(args.in_dir, args.out)
    elif args.command == "run":
        _run_pipeline(args.video, args.query, args.out)


if __name__ == "__main__":
    main()
