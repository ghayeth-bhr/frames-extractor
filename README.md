# frames-extractor

A self-operated CLI pipeline that finds specific moments in long
static-camera CCTV footage from a plain-English query (e.g. *"forklift
passing close to a pedestrian"*) and exports the matching frames as a
Roboflow-ready labeled dataset.

This is a **v1 correctness/recall proof**, run by hand against real footage
to validate the approach — not a polished product or a UI for non-technical
users. Full design rationale and locked scoping decisions live in
[`SPEC.md`](SPEC.md); read that before changing any stage.

## How it works

The query is answered by funneling a video through six stages, each a
narrowing pass over the previous stage's output. Every stage reads the prior
stage's output directory and writes its own, so any single stage can be run
and debugged in isolation:

```
extract → dedup → rank → verify → review → export
 (1000s)  (100s)  (top-K)  (VLM)  (human)  (COCO)
```

1. **`extract` (stage 1)** — OpenCV `MOG2` background subtraction (slow
   learning rate) flags frames with real motion, plus a fixed-interval floor
   sample as a recall safety net so slow/subtle motion isn't missed. Never
   returns an empty candidate set. Burned-in CCTV clock/timestamp overlays
   can be masked out first (`--mask-regions` / `--auto-mask`) since a ticking
   clock reads as constant motion to MOG2.
2. **`dedup` (stage 2)** — 64-bit pHash, Hamming distance against a sliding
   temporal window (not the whole corpus), to collapse near-duplicate frames
   without O(n²) cost.
3. **`rank` (stage 3)** — SigLIP 2 (`transformers`, batched on GPU) embeds
   every deduped frame and the text query, then ranks by cosine similarity.
   Returns the top-K (default 50) rather than a hard cutoff.
4. **`verify` (stage 4)** — one vision call per shortlisted frame to a local
   `qwen3-vl:4b` model via Ollama (`localhost:11434`), returning a
   yes/no verdict + reasoning + confidence against the query. Nothing is
   auto-discarded here — verdicts are advisory context for stage 5, cached
   to a resumable JSON manifest.
5. **`review` (stage 5)** — a small OpenCV window shows each frame with the
   VLM's verdict/reasoning overlaid; you decide with the keyboard
   (`k` keep, `d` discard, `space`/`n` skip, `q` quit-and-save). Decisions
   persist to a JSON manifest, so review is resumable across sessions.
6. **`export`** — writes kept frames to a Roboflow-importable folder:
   `images/` + `_annotations.coco.json` (images populated, no annotations
   yet — upload via the `roboflow` SDK is a manual follow-up).

Every stage's default thresholds err loose ("recall over precision") —
missing a real event is worse than showing a human an extra frame to
reject in stage 5.

## Requirements

- Python 3.11+, [`uv`](https://docs.astral.sh/uv/)
- NVIDIA GPU with CUDA (stage 3's SigLIP 2 inference; tuned against a 6GB
  card) — see `pyproject.toml` for the pinned `torch`/CUDA versions
- [Ollama](https://ollama.com) running locally with `qwen3-vl:4b` pulled,
  for stage 4 (`ollama pull qwen3-vl:4b`)

## Setup

```
uv sync
```

## Usage

Run the full pipeline against one video with one query:

```
uv run frames_extractor run \
    --video data/raw/clip1.mp4 \
    --query "person falls near the packing line" \
    --out data/output/run1
```

`run` chains the stages below as separate subprocesses (so stage 3's
PyTorch/GPU memory is fully released before stage 4's Ollama call starts)
and writes each stage's intermediate output to `data/work/run1/stageN/` —
useful for debugging a single stage without rerunning the whole pipeline:

```
uv run frames_extractor extract --video data/raw/clip1.mp4 --out data/work/run1/stage1
uv run frames_extractor dedup   --in data/work/run1/stage1 --out data/work/run1/stage2
uv run frames_extractor rank    --in data/work/run1/stage2 --query "..." --out data/work/run1/stage3
uv run frames_extractor verify  --in data/work/run1/stage3 --query "..." --out data/work/run1/stage4
uv run frames_extractor review  --in data/work/run1/stage4 --query "..." --out data/work/run1/stage5
uv run frames_extractor export  --in data/work/run1/stage5 --out data/output/run1
```

One query, one video per invocation — no multi-query or multi-video
batching in v1.

## Evaluating against ground truth

`eval/ground_truth.json` is a manually authored answer key (timestamped
real occurrences of a query) for one sample clip. After running the
pipeline against that clip:

```
uv run python eval/evaluate.py --work-dir data/work/run1 --video data/raw/clip1.mp4
```

This reports recall/precision (kept frames vs. ground-truth event windows),
the stage-by-stage funnel (raw frame count → stage 1 → ... → stage 5 kept),
and per-stage/total wall-clock time.

## Project layout

```
src/frames_extractor/
  cli.py              # entrypoints: extract, dedup, rank, verify, review, export, run
  config.py           # aggregates each stage's tunable dataclass, recall-biased defaults
  models.py           # shared dataclasses: Frame, Candidate, VerifiedFrame, ReviewDecision
  io_utils.py          # video I/O, frame save/load, per-frame timestamp helpers
  stage1_extract.py     # MOG2 motion detection + floor sampling
  stage2_dedup.py         # pHash dedup, sliding-window comparison
  stage3_rank.py            # SigLIP 2 embeddings + cosine ranking
  stage4_verify.py           # Ollama VLM verification
  stage5_review.py            # OpenCV keyboard review UI
  export.py                    # Roboflow-importable COCO export
eval/
  ground_truth.json              # answer key for the sample clip
  evaluate.py                      # scores a run against ground_truth.json
data/                               # gitignored: raw/ inputs, work/<run_id>/ per-stage output, output/<run_id>/ exports
```

## Explicitly out of scope for v1

PPE/fine-grained compliance detection, multi-query/multi-video runs, any UI
beyond the stage 5 OpenCV window, moving/PTZ camera support, automatic
Roboflow API upload, a persistent dataset/database layer, multi-provider
VLM abstraction or cost caps, multi-GPU/distributed processing. See
`SPEC.md` for the full rationale.
