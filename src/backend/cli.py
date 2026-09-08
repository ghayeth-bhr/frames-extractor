"""CLI entrypoints per SPEC.md's shown shape: run, extract, dedup, rank,
verify, review, export.

No CLI flags for internal stage tunables -- every stage runs with its
dataclass defaults (already recall-biased), matching SPEC.md's own shown
examples, which only ever pass --video/--in/--out/--query.

`run` chains all 5 stages plus export as separate subprocesses (each
invoked as `sys.executable -m backend <stage> ...`), not
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
import json
import subprocess
import sys
import time
from pathlib import Path

from . import export, models, stage1_extract, stage2_dedup, stage3_rank, stage4_verify, stage5_review
from vectordb import multi_source_index


def _parse_mask_region(value: str) -> tuple[int, int, int, int]:
    """Parses "x,y,w,h" -- matches Stage1Config.mask_regions' real format
    exactly (a top-left corner + width/height, not two corner points)."""
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"mask region must be 'x,y,w,h' (4 comma-separated ints), got {value!r}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"mask region values must be integers: {value!r}") from e
    return (x, y, w, h)


def _warn_if_event_duration_bounds_violated(min_val: float | None, max_val: float | None) -> None:
    """Validation-only, per the approved plan: --max-event-duration-sec does
    not currently drive floor-sampling (a longer event doesn't need denser
    sampling -- motion detection already catches it regardless of the floor
    interval). This just warns, it never raises -- a violated bound isn't
    fatal, it just means the caller's own bookkeeping is inconsistent."""
    if min_val is not None and max_val is not None and min_val > max_val:
        print(
            f"[extract] warning: --min-event-duration-sec ({min_val}) > "
            f"--max-event-duration-sec ({max_val}) -- ignoring, proceeding with min"
        )


def _warn_if_sampling_mode_flags_conflict(
    sampling_mode: str | None, sample_interval_sec: float | None, min_event_duration_sec: float | None
) -> None:
    """Validation-only, mirrors _warn_if_event_duration_bounds_violated: these
    flags answer deliberately different questions (see Stage1Config's
    sample_interval_sec docstring) and are never conflated in code, but a
    caller mixing them is very likely a mistake worth flagging -- never
    fatal, since neither combination actually breaks anything."""
    if sample_interval_sec is not None and sampling_mode != "fixed-fps":
        print(
            f"[extract] warning: --sample-interval-sec ({sample_interval_sec}) has no effect "
            "unless --sampling-mode fixed-fps is also given -- ignoring"
        )
    if sampling_mode == "fixed-fps" and min_event_duration_sec is not None:
        print(
            f"[extract] warning: --min-event-duration-sec ({min_event_duration_sec}) is scoped to "
            "motion mode's floor-sampling safety net and has no effect in --sampling-mode fixed-fps "
            "-- use --sample-interval-sec instead"
        )


def _add_extract_flags(p: argparse.ArgumentParser) -> None:
    """Shared by extract/run/index -- every flag that shapes Stage1Config."""
    p.add_argument("--mask-regions", dest="mask_regions", type=_parse_mask_region, nargs="+", default=None)
    p.add_argument("--auto-mask", dest="auto_mask", action="store_true", default=False)
    p.add_argument(
        "--min-event-area-ratio",
        dest="min_event_area_ratio",
        type=float,
        default=None,
        help=(
            "Override Stage1Config.min_blob_area_ratio (default 0.0054, "
            "calibrated against large-body-motion door entry/exit events). "
            "Pass a smaller value for queries targeting fine-grained motion "
            "(e.g. a hand gesture, a spill) that a door-scale gate could "
            "filter out -- this default has not been validated against "
            "small-gesture events."
        ),
    )
    p.add_argument(
        "--min-event-duration-sec",
        dest="min_event_duration_sec",
        type=float,
        default=None,
        help=(
            "Shortest real event duration you care about catching via floor "
            "sampling (motion detection is duration-independent and already "
            "catches an event regardless of this). Derives "
            "Stage1Config.floor_interval_sec = min_event_duration_sec / 2, "
            "clamped to never exceed the proven-safe default (5.0s)."
        ),
    )
    p.add_argument(
        "--max-event-duration-sec",
        dest="max_event_duration_sec",
        type=float,
        default=None,
        help=(
            "VALIDATION-ONLY for now: checked against --min-event-duration-sec "
            "(warns if min > max). Does NOT currently drive the floor-sampling "
            "interval -- a longer event doesn't need denser sampling, since "
            "motion detection already catches it regardless of floor interval."
        ),
    )
    p.add_argument(
        "--downscale-factor",
        dest="downscale_factor",
        type=float,
        default=None,
        help=(
            "Override Stage1Config.downscale_factor (default 1.0, off). "
            "Downscales the frame fed to MOG2/blob-detection only -- saved "
            "candidate images stay full resolution. 0.5 validated against "
            "data/raw/7min.mp4: identical motion candidates on all 6 known "
            "ground-truth events; see stage1_extract.Stage1Config's docstring "
            "for the boundary-region caveat before using a different value."
        ),
    )
    p.add_argument(
        "--sampling-mode",
        dest="sampling_mode",
        choices=["motion", "fixed-fps"],
        default=None,
        help=(
            "'motion' (default, unchanged): MOG2 + blob-gate detection, with "
            "floor sampling as a recall safety net alongside it. 'fixed-fps': "
            "for continuous/state-based queries (e.g. 'is this worker wearing "
            "a mask correctly') where the condition persists regardless of "
            "motion -- bypasses MOG2/blob-gate detection ENTIRELY and samples "
            "every --sample-interval-sec instead. WARNING: fixed-fps produces "
            "a MUCH larger candidate count reaching stage 2/3 than motion mode "
            "on the same clip, since nothing is filtered by activity."
        ),
    )
    p.add_argument(
        "--sample-interval-sec",
        dest="sample_interval_sec",
        type=float,
        default=None,
        help=(
            "Sampling interval for --sampling-mode fixed-fps ONLY (default "
            "2.0s) -- has no effect in motion mode. Deliberately separate "
            "from --min-event-duration-sec, which stays scoped to motion "
            "mode's floor-sampling safety net and answers a different "
            "question (shortest event that might otherwise be missed between "
            "motion detections, not how densely to sample a condition that "
            "never triggers motion at all)."
        ),
    )


def _add_dedup_flags(p: argparse.ArgumentParser) -> None:
    """Shared by dedup/run/index -- every flag that shapes Stage2Config."""
    p.add_argument(
        "--dedup-hamming-threshold",
        dest="dedup_hamming_threshold",
        type=int,
        default=None,
        help=(
            "Override Stage2Config.hamming_threshold (default 8, recall-biased -- "
            "only collapses very close pHash matches). A production-speed run can "
            "raise this to merge more near-duplicates and shrink stage 3/4's input, "
            "trading recall for speed deliberately."
        ),
    )
    p.add_argument(
        "--dedup-window-size",
        dest="dedup_window_size",
        type=int,
        default=None,
        help="Override Stage2Config.window_size (default 5, the sliding dedup comparison window).",
    )


def _add_rank_flags(p: argparse.ArgumentParser) -> None:
    """Shared by rank/run/search -- every flag that shapes Stage3Config."""
    p.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help=(
            "Override Stage3Config.top_k (default 50, a hard count cap not a "
            "similarity cutoff). A production-speed run can lower this to shrink "
            "stage 4/5's input, trading recall for speed deliberately."
        ),
    )


def _add_verify_flags(p: argparse.ArgumentParser) -> None:
    """Shared by verify/run/search -- every flag that shapes Stage4Config."""
    p.add_argument(
        "--skip-vlm",
        dest="skip_vlm",
        action="store_true",
        default=False,
        help=(
            "Skip stage 4's VLM verification entirely -- zero Ollama calls, every "
            "candidate gets verdict='skipped' (distinct from 'error': no call was "
            "ever attempted). For a fast 'help me find candidates' production mode, "
            "not the recall-critical default -- stage 5 still reviews every frame."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frames_extractor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run")
    run_p.add_argument("--video", type=Path, required=True)
    run_p.add_argument("--query", type=str, required=True)
    run_p.add_argument("--out", type=Path, required=True)
    _add_extract_flags(run_p)
    _add_dedup_flags(run_p)
    _add_rank_flags(run_p)
    _add_verify_flags(run_p)

    extract_p = subparsers.add_parser("extract")
    extract_p.add_argument("--video", type=Path, required=True)
    extract_p.add_argument("--out", type=Path, required=True)
    _add_extract_flags(extract_p)

    dedup_p = subparsers.add_parser("dedup")
    dedup_p.add_argument("--in", dest="in_dir", type=Path, required=True)
    dedup_p.add_argument("--out", type=Path, required=True)
    _add_dedup_flags(dedup_p)

    rank_p = subparsers.add_parser("rank")
    rank_p.add_argument("--in", dest="in_dir", type=Path, required=True)
    rank_p.add_argument("--query", type=str, required=True)
    rank_p.add_argument("--out", type=Path, required=True)
    _add_rank_flags(rank_p)

    verify_p = subparsers.add_parser("verify")
    verify_p.add_argument("--in", dest="in_dir", type=Path, required=True)
    verify_p.add_argument("--query", type=str, required=True)
    verify_p.add_argument("--out", type=Path, required=True)
    _add_verify_flags(verify_p)

    review_p = subparsers.add_parser("review")
    review_p.add_argument("--in", dest="in_dir", type=Path, required=True)
    review_p.add_argument("--query", type=str, required=True)
    review_p.add_argument("--out", type=Path, required=True)

    export_p = subparsers.add_parser("export")
    export_p.add_argument("--in", dest="in_dir", type=Path, required=True)
    export_p.add_argument("--out", type=Path, required=True)

    index_p = subparsers.add_parser("index")
    index_p.add_argument("--video", type=Path, required=True)
    index_p.add_argument("--out", type=Path, required=True)
    _add_extract_flags(index_p)
    _add_dedup_flags(index_p)

    search_p = subparsers.add_parser("search")
    search_p.add_argument("--index", dest="index_dir", type=Path, required=True)
    search_p.add_argument("--query", type=str, required=True)
    search_p.add_argument("--out", type=Path, required=True)
    _add_rank_flags(search_p)
    _add_verify_flags(search_p)

    # Single-stage counterpart to "search", exactly as "rank" is to "run" --
    # search's own subprocess orchestration (_search_pipeline) dispatches to
    # this internally; not usually invoked directly by a human, but follows
    # the same "every stage runnable in isolation" contract as every other
    # single-stage subcommand.
    search_index_p = subparsers.add_parser("search-index")
    search_index_p.add_argument("--index", dest="index_dir", type=Path, required=True)
    search_index_p.add_argument("--query", type=str, required=True)
    search_index_p.add_argument("--out", type=Path, required=True)
    _add_rank_flags(search_index_p)

    # For the webapp's multi-source assets (see multi_source_index.py) --
    # embeds one source's stage2 output and appends it to a combined index.
    # Invoked as a subprocess by the webapp's background worker, the same
    # way `run` invokes its own stages, so the SigLIP2 CUDA context this
    # spawns is guaranteed released on process exit.
    add_to_index_p = subparsers.add_parser("add-to-index")
    add_to_index_p.add_argument("--index", dest="index_dir", type=Path, required=True)
    add_to_index_p.add_argument("--stage2-dir", dest="stage2_dir", type=Path, required=True)
    add_to_index_p.add_argument("--source-id", dest="source_id", type=str, required=True)
    add_to_index_p.add_argument("--cross-source-dedup", action="store_true", default=False)
    add_to_index_p.add_argument("--cross-source-hamming-threshold", type=int, default=8)

    return parser


def _run_stage_subprocess(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "backend", *args], check=True)


def _run_pipeline(
    video: Path,
    query: str,
    out: Path,
    mask_regions: list[tuple[int, int, int, int]] | None = None,
    auto_mask: bool = False,
    min_event_area_ratio: float | None = None,
    min_event_duration_sec: float | None = None,
    max_event_duration_sec: float | None = None,
    downscale_factor: float | None = None,
    sampling_mode: str | None = None,
    sample_interval_sec: float | None = None,
    dedup_hamming_threshold: int | None = None,
    dedup_window_size: int | None = None,
    top_k: int | None = None,
    skip_vlm: bool = False,
) -> None:
    run_id = out.name
    work_dir = Path("data/work") / run_id
    stage_dirs = {n: work_dir / f"stage{n}" for n in range(1, 6)}
    timing_path = work_dir / "timing.json"
    stage_elapsed_sec: dict[str, float] = {}

    def _record_elapsed(stage_key: str, elapsed: float) -> None:
        # Incremental write (same pattern as every stage's own manifest) so a
        # later stage's failure doesn't lose already-completed stages' timing.
        stage_elapsed_sec[stage_key] = elapsed
        work_dir.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(stage_elapsed_sec, indent=2))

    print(f"[run] stage 1/5: extract -- {video}")
    t0 = time.time()
    extract_args = ["extract", "--video", str(video), "--out", str(stage_dirs[1])]
    if mask_regions:
        extract_args.append("--mask-regions")
        extract_args.extend(f"{x},{y},{w},{h}" for x, y, w, h in mask_regions)
    if auto_mask:
        extract_args.append("--auto-mask")
    if min_event_area_ratio is not None:
        extract_args.extend(["--min-event-area-ratio", str(min_event_area_ratio)])
    if min_event_duration_sec is not None:
        extract_args.extend(["--min-event-duration-sec", str(min_event_duration_sec)])
    if max_event_duration_sec is not None:
        extract_args.extend(["--max-event-duration-sec", str(max_event_duration_sec)])
    if downscale_factor is not None:
        extract_args.extend(["--downscale-factor", str(downscale_factor)])
    if sampling_mode is not None:
        extract_args.extend(["--sampling-mode", sampling_mode])
    if sample_interval_sec is not None:
        extract_args.extend(["--sample-interval-sec", str(sample_interval_sec)])
    _run_stage_subprocess(extract_args)
    _record_elapsed("stage1", time.time() - t0)
    count1 = len(models.load_candidates(stage_dirs[1] / "candidates.json"))
    print(f"[run] stage 1 done: {count1} candidates ({stage_elapsed_sec['stage1']:.1f}s)")

    print("[run] stage 2/5: dedup")
    t0 = time.time()
    dedup_args = ["dedup", "--in", str(stage_dirs[1]), "--out", str(stage_dirs[2])]
    if dedup_hamming_threshold is not None:
        dedup_args.extend(["--dedup-hamming-threshold", str(dedup_hamming_threshold)])
    if dedup_window_size is not None:
        dedup_args.extend(["--dedup-window-size", str(dedup_window_size)])
    _run_stage_subprocess(dedup_args)
    _record_elapsed("stage2", time.time() - t0)
    count2 = len(models.load_candidates(stage_dirs[2] / "candidates.json"))
    print(f"[run] stage 2 done: {count1} -> {count2} candidates ({stage_elapsed_sec['stage2']:.1f}s)")

    print("[run] stage 3/5: rank")
    t0 = time.time()
    rank_args = ["rank", "--in", str(stage_dirs[2]), "--query", query, "--out", str(stage_dirs[3])]
    if top_k is not None:
        rank_args.extend(["--top-k", str(top_k)])
    _run_stage_subprocess(rank_args)
    _record_elapsed("stage3", time.time() - t0)
    count3 = len(models.load_candidates(stage_dirs[3] / "candidates.json"))
    print(f"[run] stage 3 done: {count2} -> {count3} candidates ({stage_elapsed_sec['stage3']:.1f}s)")

    print("[run] stage 4/5: verify -- SKIPPED (--skip-vlm)" if skip_vlm else "[run] stage 4/5: verify")
    t0 = time.time()
    verify_args = ["verify", "--in", str(stage_dirs[3]), "--query", query, "--out", str(stage_dirs[4])]
    if skip_vlm:
        verify_args.append("--skip-vlm")
    _run_stage_subprocess(verify_args)
    _record_elapsed("stage4", time.time() - t0)
    verified = models.load_verified_frames(stage_dirs[4] / "candidates.json")
    yes_count = sum(1 for v in verified if v.verdict == "yes")
    no_count = sum(1 for v in verified if v.verdict == "no")
    error_count = sum(1 for v in verified if v.verdict == "error")
    skipped_count = sum(1 for v in verified if v.verdict == "skipped")
    print(
        f"[run] stage 4 done: {count3} -> {len(verified)} verified "
        f"({yes_count} yes, {no_count} no, {error_count} error, {skipped_count} skipped) "
        f"({stage_elapsed_sec['stage4']:.1f}s)"
    )

    print("[run] stage 5/5: review -- opens an interactive window, decide with k/d/space/q")
    t0 = time.time()
    _run_stage_subprocess(["review", "--in", str(stage_dirs[4]), "--query", query, "--out", str(stage_dirs[5])])
    _record_elapsed("stage5", time.time() - t0)
    decisions = models.load_review_decisions(stage_dirs[5] / "candidates.json")
    kept = sum(1 for d in decisions if d.decision == "keep")
    print(f"[run] stage 5 done: {kept} kept, {len(decisions) - kept} discarded ({stage_elapsed_sec['stage5']:.1f}s)")

    print(f"[run] export -- writing final output to {out}")
    _run_stage_subprocess(["export", "--in", str(stage_dirs[5]), "--out", str(out)])


def _stage1_config_from_flags(
    mask_regions: list[tuple[int, int, int, int]] | None,
    auto_mask: bool,
    min_event_area_ratio: float | None,
    min_event_duration_sec: float | None,
    max_event_duration_sec: float | None,
    downscale_factor: float | None,
    sampling_mode: str | None,
    sample_interval_sec: float | None,
) -> "stage1_extract.Stage1Config":
    """Shared by main()'s "extract" dispatch and _index_pipeline -- both
    build a Stage1Config from the identical flag set."""
    config_kwargs = {"mask_regions": mask_regions, "run_auto_detect": auto_mask}
    if min_event_area_ratio is not None:
        config_kwargs["min_blob_area_ratio"] = min_event_area_ratio
    _warn_if_event_duration_bounds_violated(min_event_duration_sec, max_event_duration_sec)
    _warn_if_sampling_mode_flags_conflict(sampling_mode, sample_interval_sec, min_event_duration_sec)
    if min_event_duration_sec is not None:
        config_kwargs["floor_interval_sec"] = stage1_extract.derive_floor_interval_sec(min_event_duration_sec)
    if downscale_factor is not None:
        config_kwargs["downscale_factor"] = downscale_factor
    if sampling_mode is not None:
        config_kwargs["sampling_mode"] = sampling_mode
    if sample_interval_sec is not None:
        config_kwargs["sample_interval_sec"] = sample_interval_sec
    return stage1_extract.Stage1Config(**config_kwargs)


def _stage2_config_from_flags(
    dedup_hamming_threshold: int | None, dedup_window_size: int | None
) -> "stage2_dedup.Stage2Config | None":
    """Shared by main()'s "dedup" dispatch and _index_pipeline."""
    config_kwargs = {}
    if dedup_hamming_threshold is not None:
        config_kwargs["hamming_threshold"] = dedup_hamming_threshold
    if dedup_window_size is not None:
        config_kwargs["window_size"] = dedup_window_size
    return stage2_dedup.Stage2Config(**config_kwargs) if config_kwargs else None


def _index_pipeline(
    video: Path,
    out: Path,
    mask_regions: list[tuple[int, int, int, int]] | None = None,
    auto_mask: bool = False,
    min_event_area_ratio: float | None = None,
    min_event_duration_sec: float | None = None,
    max_event_duration_sec: float | None = None,
    downscale_factor: float | None = None,
    sampling_mode: str | None = None,
    sample_interval_sec: float | None = None,
    dedup_hamming_threshold: int | None = None,
    dedup_window_size: int | None = None,
) -> None:
    """Builds a reusable "build once, search many times" index: stage 1
    (extract) -> stage 2 (dedup) -> stage 3's query-independent embedding
    half (build_index). Runs all three IN-PROCESS, unlike _run_pipeline's
    per-stage subprocess isolation -- that isolation exists specifically to
    protect stage 3 (SigLIP2) and stage 4 (Ollama) from a VRAM-contention
    crash when handing off between two DIFFERENT GPU consumers; index never
    reaches stage 4 at all, so there's no cross-model GPU handoff here to
    protect against.

    `out` is a plain, user-named, REUSABLE directory -- not a
    data/work/<run_id> run artifact -- since the whole point is reusing it
    across many later `search` calls against different queries. The final
    index (candidates.json + embeddings.npy + copied images) ends up sitting
    directly IN `out` itself (dedup's own output dir), so `search --index`
    is given the exact same path this command's `--out` was.
    """
    stage1_dir = out / "stage1"  # debug parity only -- never skip writing intermediate output
    timing_path = out / "timing.json"
    stage_elapsed_sec: dict[str, float] = {}

    def _record_elapsed(stage_key: str, elapsed: float) -> None:
        stage_elapsed_sec[stage_key] = elapsed
        out.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(stage_elapsed_sec, indent=2))

    print(f"[index] stage 1/3: extract -- {video}")
    t0 = time.time()
    config1 = _stage1_config_from_flags(
        mask_regions, auto_mask, min_event_area_ratio, min_event_duration_sec,
        max_event_duration_sec, downscale_factor, sampling_mode, sample_interval_sec,
    )
    stage1_extract.extract(video, stage1_dir, config1)
    _record_elapsed("stage1", time.time() - t0)
    count1 = len(models.load_candidates(stage1_dir / "candidates.json"))
    print(f"[index] stage 1 done: {count1} candidates ({stage_elapsed_sec['stage1']:.1f}s)")

    print("[index] stage 2/3: dedup")
    t0 = time.time()
    config2 = _stage2_config_from_flags(dedup_hamming_threshold, dedup_window_size)
    stage2_dedup.dedup(stage1_dir, out, config2)
    _record_elapsed("stage2", time.time() - t0)
    count2 = len(models.load_candidates(out / "candidates.json"))
    print(f"[index] stage 2 done: {count1} -> {count2} candidates ({stage_elapsed_sec['stage2']:.1f}s)")

    print("[index] stage 3/3: build_index -- embedding every candidate (query-independent, no top_k cut)")
    t0 = time.time()
    stage3_rank.build_index(out)
    _record_elapsed("stage3", time.time() - t0)
    print(f"[index] stage 3 done: {count2} candidates embedded ({stage_elapsed_sec['stage3']:.1f}s)")
    print(f'[index] index ready -- run `search --index {out} --query "..." --out <output>` to use it')


def _search_pipeline(
    index_dir: Path,
    query: str,
    out: Path,
    top_k: int | None = None,
    skip_vlm: bool = False,
) -> None:
    """Structurally parallel to _run_pipeline, but starts from a
    build_index() output instead of a video -- never re-runs stage 1/2,
    never re-embeds an image. Uses the SAME subprocess-per-stage isolation
    as _run_pipeline between its own stage-3-equivalent (search-index,
    SigLIP2) and stage 4 (verify, Ollama) -- identical VRAM-contention risk
    to run's stage3->stage4 handoff.
    """
    run_id = out.name
    work_dir = Path("data/work") / run_id
    stage_dirs = {n: work_dir / f"stage{n}" for n in range(3, 6)}
    timing_path = work_dir / "timing.json"
    stage_elapsed_sec: dict[str, float] = {}

    def _record_elapsed(stage_key: str, elapsed: float) -> None:
        stage_elapsed_sec[stage_key] = elapsed
        work_dir.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(stage_elapsed_sec, indent=2))

    print(f"[search] stage 3/5: search-index -- {index_dir}")
    t0 = time.time()
    search_args = ["search-index", "--index", str(index_dir), "--query", query, "--out", str(stage_dirs[3])]
    if top_k is not None:
        search_args.extend(["--top-k", str(top_k)])
    _run_stage_subprocess(search_args)
    _record_elapsed("stage3", time.time() - t0)
    count3 = len(models.load_candidates(stage_dirs[3] / "candidates.json"))
    print(f"[search] stage 3 done: {count3} candidates ({stage_elapsed_sec['stage3']:.1f}s)")

    print("[search] stage 4/5: verify -- SKIPPED (--skip-vlm)" if skip_vlm else "[search] stage 4/5: verify")
    t0 = time.time()
    verify_args = ["verify", "--in", str(stage_dirs[3]), "--query", query, "--out", str(stage_dirs[4])]
    if skip_vlm:
        verify_args.append("--skip-vlm")
    _run_stage_subprocess(verify_args)
    _record_elapsed("stage4", time.time() - t0)
    verified = models.load_verified_frames(stage_dirs[4] / "candidates.json")
    yes_count = sum(1 for v in verified if v.verdict == "yes")
    no_count = sum(1 for v in verified if v.verdict == "no")
    error_count = sum(1 for v in verified if v.verdict == "error")
    skipped_count = sum(1 for v in verified if v.verdict == "skipped")
    print(
        f"[search] stage 4 done: {count3} -> {len(verified)} verified "
        f"({yes_count} yes, {no_count} no, {error_count} error, {skipped_count} skipped) "
        f"({stage_elapsed_sec['stage4']:.1f}s)"
    )

    print("[search] stage 5/5: review -- opens an interactive window, decide with k/d/space/q")
    t0 = time.time()
    _run_stage_subprocess(["review", "--in", str(stage_dirs[4]), "--query", query, "--out", str(stage_dirs[5])])
    _record_elapsed("stage5", time.time() - t0)
    decisions = models.load_review_decisions(stage_dirs[5] / "candidates.json")
    kept = sum(1 for d in decisions if d.decision == "keep")
    print(f"[search] stage 5 done: {kept} kept, {len(decisions) - kept} discarded ({stage_elapsed_sec['stage5']:.1f}s)")

    print(f"[search] export -- writing final output to {out}")
    _run_stage_subprocess(["export", "--in", str(stage_dirs[5]), "--out", str(out)])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "extract":
        config = _stage1_config_from_flags(
            args.mask_regions, args.auto_mask, args.min_event_area_ratio,
            args.min_event_duration_sec, args.max_event_duration_sec,
            args.downscale_factor, args.sampling_mode, args.sample_interval_sec,
        )
        stage1_extract.extract(args.video, args.out, config)
    elif args.command == "dedup":
        dedup_config = _stage2_config_from_flags(args.dedup_hamming_threshold, args.dedup_window_size)
        stage2_dedup.dedup(args.in_dir, args.out, dedup_config)
    elif args.command == "rank":
        rank_config = stage3_rank.Stage3Config(top_k=args.top_k) if args.top_k is not None else None
        stage3_rank.rank(args.in_dir, args.out, args.query, rank_config)
    elif args.command == "verify":
        verify_config = stage4_verify.Stage4Config(skip=args.skip_vlm) if args.skip_vlm else None
        stage4_verify.verify(args.in_dir, args.out, args.query, verify_config)
    elif args.command == "review":
        stage5_review.review(args.in_dir, args.out, args.query)
    elif args.command == "export":
        export.export(args.in_dir, args.out)
    elif args.command == "run":
        _run_pipeline(
            args.video,
            args.query,
            args.out,
            args.mask_regions,
            args.auto_mask,
            args.min_event_area_ratio,
            args.min_event_duration_sec,
            args.max_event_duration_sec,
            args.downscale_factor,
            args.sampling_mode,
            args.sample_interval_sec,
            args.dedup_hamming_threshold,
            args.dedup_window_size,
            args.top_k,
            args.skip_vlm,
        )
    elif args.command == "index":
        _index_pipeline(
            args.video,
            args.out,
            args.mask_regions,
            args.auto_mask,
            args.min_event_area_ratio,
            args.min_event_duration_sec,
            args.max_event_duration_sec,
            args.downscale_factor,
            args.sampling_mode,
            args.sample_interval_sec,
            args.dedup_hamming_threshold,
            args.dedup_window_size,
        )
    elif args.command == "search":
        _search_pipeline(args.index_dir, args.query, args.out, args.top_k, args.skip_vlm)
    elif args.command == "search-index":
        rank_config = stage3_rank.Stage3Config(top_k=args.top_k) if args.top_k is not None else None
        stage3_rank.search_index(args.index_dir, args.out, args.query, rank_config)
    elif args.command == "add-to-index":
        multi_source_index.add_to_index(
            args.index_dir,
            args.stage2_dir,
            args.source_id,
            cross_source_dedup=args.cross_source_dedup,
            cross_source_hamming_threshold=args.cross_source_hamming_threshold,
        )


if __name__ == "__main__":
    main()
