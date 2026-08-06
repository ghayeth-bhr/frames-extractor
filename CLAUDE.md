# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Pre-implementation. The repository currently contains only `SPEC.md` — no
`pyproject.toml`, no `src/`, no tests exist yet. There are no build/lint/test
commands to run until the codebase described below is scaffolded.

## What this is

A self-operated CLI pipeline (v1, correctness/recall proof — not a polished
product) that finds specific moments in long static-camera CCTV footage from
a plain-English query and exports the matching frames as a Roboflow-ready
labeled dataset. Full scope, locked decisions, and stage-by-stage design
rationale live in `SPEC.md` — read it before implementing any stage.

## Architecture

The pipeline is six sequential CLI subcommands, each reading the previous
stage's output directory and writing its own, so any single stage can be
run and debugged in isolation:

```
extract → dedup → rank → verify → review → export
```

(`frames_extractor run` chains all six in one invocation.)

- **stage1_extract.py** — OpenCV MOG2 background subtraction (slow learning
  rate) finds motion candidates, plus a fixed-interval floor sample as a
  recall safety net; never returns an empty candidate set.
- **stage2_dedup.py** — 64-bit pHash, Hamming distance vs. a sliding
  temporal window (not the whole corpus) to avoid O(n²) cost.
- **stage3_rank.py** — SigLIP 2 (`transformers`, batched on GPU) cosine
  similarity of frame embeddings vs. the text query; returns top-K, not a
  hard cutoff.
- **stage4_verify.py** — one Claude Haiku vision call per shortlisted frame
  (yes/no + reasoning + confidence); results cached to a JSON manifest keyed
  by frame id so reruns after a network blip only re-call uncached frames.
- **stage5_review.py** — OpenCV keyboard review window (`k`/`d`/`space`/`n`/`q`);
  decisions persist to a JSON manifest, so review is resumable across
  sessions.
- **export.py** — writes `images/` + `_annotations.coco.json` in a
  Roboflow-importable layout (no annotations populated, no API upload).

Shared modules: `models.py` (Frame/Candidate/VerifiedFrame/ReviewDecision
dataclasses), `config.py` (all tunable thresholds, defaults biased toward
recall), `io_utils.py` (video I/O, frame save/load, timestamp helpers).

### Cross-cutting decisions that shape every stage

- **Recall over precision everywhere.** Every stage's default thresholds
  err loose — a missed real event is worse than an extra frame for a human
  to reject in stage 5. When tuning any stage, bias toward under-filtering.
- **Timestamps are read per-frame via `cap.get(cv2.CAP_PROP_POS_MSEC)`**,
  never derived from `frame_index / nominal_fps` — CCTV exports have
  variable frame rate and dropped frames, and index-based timestamps can
  silently drift over a long clip. `io_utils.py` also cross-checks
  `CAP_PROP_FRAME_COUNT / fps` against reported duration on load.
- **Burned-in clock/timestamp overlays** are a known CCTV hazard for MOG2
  (a ticking region reads as constant motion everywhere) — mask them out
  before background subtraction in stage 1.
- **Run granularity is one query, one video per invocation** — no
  multi-query or multi-video batching in v1.
- Stage 4 does not reduce stage 5's review volume in v1 (nothing is
  auto-discarded on VLM verification) — the pipeline's "time saved" story
  is the stage 1→3 funnel, not a smaller review set. Don't read this as
  underperformance when evaluating results.

### Explicitly out of scope for v1

PPE/fine-grained compliance detection, multi-query/multi-video runs, any UI
beyond the stage 5 OpenCV window, moving/PTZ camera support, automatic
Roboflow API upload, FiftyOne/MongoDB/persistent DB layer, multi-provider
VLM abstraction or cost caps, multi-GPU/distributed processing.

## Development

Always check `SPEC.md` before implementing a stage — it is the source of
truth for scope, file layout, and the recall-bias default.

### Python env

Use `uv` — activate `.venv` / run via `uv run` before running anything.

### GPU

Single consumer GPU available locally: NVIDIA GeForce RTX 4050, 6GB VRAM
(driver 577.05, CUDA 12.9). SigLIP 2 batch sizes and checkpoint choice
(`so400m` vs. `large-patch16` fallback) should be sized against this.

### Intermediate output

Every stage writes its output to `data/work/<run_id>/stageN/` per
`SPEC.md`'s CLI shape — never skip writing intermediate output, even when
chaining stages via `run`, since debugging a single stage depends on it.
