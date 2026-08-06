# Frame-Mining Tool — v1 Spec

## Context

Ops teams (safety/QA/manufacturing) need to find specific moments in long raw
CCTV footage by describing them in plain English ("forklift passing close to
a pedestrian", "person falls near the packing line"), then export the
matching frames as a labeled training dataset. Today this means manually
scrubbing hours of footage.

This v1 is a self-operated CLI pipeline, run by the tool's author against
real Absar footage, to prove the approach works before investing in any UI
for non-technical ops users. It is explicitly a correctness/recall proof,
not a polished product.

Decisions locked in during scoping (see prior research pass for the
technical justification behind each):

| Question | Decision |
|---|---|
| Interface | CLI script. Stage 5 review is a small OpenCV window with keyboard shortcuts — no web server. |
| Stage 1 approach | Absar footage is static fixed CCTV → motion detection (OpenCV MOG2 background subtraction), **not** shot/scene-cut detection. |
| Stage 3 model | SigLIP 2 (so400m or ViT-L/16 checkpoint), GPU available (single consumer GPU) so batched inference is the default. |
| Stage 4 model | Claude Haiku via the Anthropic API (`ANTHROPIC_API_KEY` env var), single provider, no abstraction layer for v1. |
| Ranking bias | Recall over precision — loose thresholds at stage 1/3, let stage 4 (VLM) + stage 5 (human) do the filtering. Missing a real safety event is worse than reviewing extra frames. |
| Scale | A handful of clips, 10-60 min each, 1080p or lower, single static camera. Design must not assume the whole video fits in memory, but doesn't need multi-hour/multi-camera throughput yet. |
| PPE / fine-grained compliance (hairnets etc.) | **Out of scope for v1.** CLIP/SigLIP + VLM verification are not reliable at this granularity; needs a dedicated detect-then-classify step later. |
| Run granularity | One query, one video per invocation. No batching of multiple queries or multiple videos in v1. |
| Definition of done | You run the CLI yourself against Absar footage and judge the output. Not required to be operable by a non-technical ops person yet. |
| Ground truth for verification | You manually scrub one sample clip once and write down every real occurrence of the test query as a timestamped answer key. |
| Frame timestamps | Read per-frame via `cap.get(cv2.CAP_PROP_POS_MSEC)`, **not** derived from `frame_index / nominal_fps` — CCTV exports are prone to variable frame rate and dropped frames, and index-based timestamps can silently drift seconds-to-minutes over a 60-minute clip, corrupting the recall/precision scoring without any visible error. `io_utils.py` also sanity-checks `CAP_PROP_FRAME_COUNT / fps` against the video's actual reported duration on load and warns if they diverge meaningfully. |
| SigLIP 2 loading library | `transformers` (`AutoModel`/`AutoProcessor`, e.g. `google/siglip2-so400m-patch14-384`) — official HF-native support, avoids `open_clip`'s separate pretrained-tag lookup for checkpoint availability. Confirm the exact checkpoint id is available at implementation time; fall back to the `large-patch16` variant if `so400m` proves too slow on the target GPU. |

## Pipeline & modules

```
frames-extractor/
  pyproject.toml
  SPEC.md
  src/frames_extractor/
    __init__.py
    cli.py              # entrypoints: extract, dedup, rank, verify, review, export, run (chains all)
    config.py           # dataclass of all tunable thresholds/paths, defaults biased toward recall
    models.py            # shared dataclasses: Frame, Candidate, VerifiedFrame, ReviewDecision
    io_utils.py           # video reading (cv2.VideoCapture), frame save/load, timestamp helpers (per-frame CAP_PROP_POS_MSEC, not index/fps)
    stage1_extract.py     # MOG2 motion detection + fixed-interval floor sampling
    stage2_dedup.py        # imagehash phash, Hamming distance, temporal-window comparison
    stage3_rank.py          # SigLIP2 embeddings, cosine similarity vs text query, top-K shortlist
    stage4_verify.py        # Claude Haiku vision call per shortlisted frame, yes/no + reasoning + confidence
    stage5_review.py         # OpenCV keyboard review UI, resumable JSON manifest of decisions
    export.py                 # writes images/ + _annotations.coco.json in Roboflow-importable layout
  eval/
    ground_truth.json          # manually authored answer key for the sample Absar clip
    evaluate.py                 # scores pipeline output against ground_truth.json
  data/                          # gitignored
    raw/                          # input videos
    work/<run_id>/                # intermediate per-stage output (for debugging each stage independently)
    output/<run_id>/               # final export folder
```

### CLI shape

```
frames_extractor run --video data/raw/clip1.mp4 \
                      --query "person falls near the packing line" \
                      --out data/output/run1

# or step-by-step for debugging a single stage:
frames_extractor extract --video data/raw/clip1.mp4 --out data/work/run1/stage1
frames_extractor dedup   --in data/work/run1/stage1 --out data/work/run1/stage2
frames_extractor rank    --in data/work/run1/stage2 --query "..." --out data/work/run1/stage3
frames_extractor verify  --in data/work/run1/stage3 --query "..." --out data/work/run1/stage4
frames_extractor review  --in data/work/run1/stage4 --out data/work/run1/stage5
frames_extractor export  --in data/work/run1/stage5 --out data/output/run1
```

### Stage 1 — coarse extraction (`stage1_extract.py`)
OpenCV `MOG2` background subtraction per video, tuned with a slow learning
rate so gradual lighting drift doesn't get flagged as motion; frames where
foreground blob area exceeds a threshold are candidates. A fixed-interval
floor sample (e.g. one frame every N seconds regardless of motion) is added
as a recall safety net, since MOG2 can miss slow/subtle motion. If the
whole clip has zero motion above threshold, floor samples are still
emitted — stage 1 never returns an empty candidate set for an active,
recall-biased run.

**Before tuning thresholds against the sample Absar clip**, check for a
burned-in timestamp/clock overlay (very common in CCTV exports). A ticking
on-screen clock is a small region of constant per-second pixel change that
MOG2 can misread as motion everywhere, every frame — flooding stage 1 with
junk candidates, or worse, getting absorbed into the background model in a
way that masks real motion nearby it. If present, mask that region out of
the frame before running background subtraction.

### Stage 2 — dedup (`stage2_dedup.py`)
64-bit pHash (`imagehash`) per candidate frame, Hamming distance threshold
(~8, tunable). Near-duplicates are temporally local in video, so each frame
is only compared against a sliding window of recent frames (not the whole
corpus) — avoids O(n²) cost without needing a BK-tree at this scale.
Same recall bias as stages 1 and 3 applies here: default to erring loose
(a lower Hamming threshold, i.e. only collapsing very close matches) —
under-merging costs a few extra frames downstream, over-merging can
silently drop the one frame that captured the actual event.

### Stage 3 — semantic ranking (`stage3_rank.py`)
SigLIP 2 image encoder, loaded via `transformers` (see decision table),
batched on GPU, embeds every deduped frame; the text query is embedded
once; frames ranked by cosine similarity. Recall bias means this stage
returns the top-K by score (default K=50) rather than applying a hard
similarity cutoff that could silently exclude a true positive.

### Stage 4 — VLM verification (`stage4_verify.py`)
Claude Haiku, one vision call per stage-3 shortlisted frame: given the
frame and the original natural-language query, returns yes/no + short
reasoning + confidence. Nothing is auto-discarded here — low-confidence
"no" frames still pass through to stage 5 so a human sees borderline
cases, consistent with the recall bias. This means v1's "time saved"
story is really the stage 1–3 funnel (thousands of raw frames → ~50
shortlisted), not a reduction in stage 5 review volume — worth being
clear-eyed about when reading the eval results against the original
"1200 calls → 30-50 confirmed" framing; that narrowing is a later
optimization, not something v1 measures.

Results are cached to a JSON manifest keyed by frame id as each call
completes (same resumability pattern as stage 5's decision manifest) —
a re-run after a network blip or rate limit only re-calls frames without
a cached verdict, rather than re-spending all ~50 API calls.

### Stage 5 — human review (`stage5_review.py`)
Standalone OpenCV window (`cv2.imshow`), one frame at a time, with the
VLM's reasoning/confidence and the original query overlaid as text.
Keybindings: `k` keep, `d` discard, `space`/`n` next (skip without
deciding), `q` quit-and-save. Decisions persist to a JSON manifest as you
go, so review is resumable across sessions.

### Export (`export.py`)
Kept frames written to a Roboflow-importable layout:
```
data/output/<run_id>/
  images/frame_000001.jpg ...
  _annotations.coco.json   # images populated, "annotations": [] (no labels yet)
```
Actual upload via the `roboflow` SDK is not built in v1 — the output folder
is structured so a manual `project.upload()` call works without
reformatting.

## Explicitly out of scope for v1
- PPE/fine-grained compliance detection (hairnets, gloves, etc.)
- Multiple queries or multiple videos in a single run
- Any UI beyond the stage 5 OpenCV review window (no Streamlit/Gradio/web app)
- Moving/handheld/PTZ camera support in stage 1 (static-camera assumption only)
- Automatic Roboflow API upload (folder is upload-ready; the call itself is a manual follow-up)
- FiftyOne, MongoDB, or any persistent dataset/database layer
- Multi-provider VLM abstraction, cost/budget caps (single hardcoded Claude Haiku call path)
- Multi-GPU / distributed / multi-camera-simultaneous processing

## End-to-end verification

1. Pick one sample Absar clip (10-60 min, static camera).
2. Manually scrub it once; write `eval/ground_truth.json` — every real
   occurrence of the test query as `{start_ts, end_ts, description}`.
3. Run the full pipeline via `frames_extractor run` against that clip with
   the matching query.
4. Run `eval/evaluate.py`, which:
   - Matches each **kept** (stage 5) frame's timestamp against
     `ground_truth.json` windows (± tolerance, e.g. 2s) to compute
     **recall** (fraction of ground-truth events with ≥1 matching kept
     frame) and **precision** (fraction of kept frames matching a real
     event).
   - Reports the stage-by-stage funnel (raw frame count → stage 1 →
     stage 2 → stage 3 shortlist → stage 4 verified → stage 5 kept) so
     it's visible where events are gained or lost.
   - Reports **time saved**: wall-clock pipeline run time + your stage 5
     review time, vs. the time you spent manually scrubbing the same clip
     in step 2. Note this is really measuring stage 1–3's search/shortlist
     value (thousands of raw frames narrowed to ~50) — stage 4 doesn't
     reduce stage 5's review volume in v1 (see stage 4 section), so don't
     read a modest time-saved number here as the pipeline underperforming
     the original pitch; that's a v2 optimization, not a v1 target.
5. Success is demonstrating actual recall/precision numbers plus a
   Roboflow-importable output folder — not hitting a specific numeric bar
   yet, since this is the first real test against Absar footage.
