# Frame-Extractor Optimization Plan — 6 Steps

## How to work through this file

Work through the steps below **in order**. Each step ends with a
**STOP GATE** — do not proceed to the next step until that gate's
validation has run and passed. Report the results of each step's gate
back to the user and wait for explicit confirmation ("continue") before
starting the next step.

This ordering is deliberate: later steps assume earlier steps' changes
are already correct and recall-safe. Skipping a gate means any later
step could be silently built on top of a broken assumption.

**The one rule that overrides everything else below:** every change in
this plan must be validated by running the full pipeline against
`data/raw/7min.mp4` through `export`, then running
`eval/evaluate.py` against `eval/ground_truth.json`. **Recall must stay
at 6/6 after every single change.** A faster pipeline that misses a real
event is a regression, not an optimization — treat any recall drop as a
hard blocker, not a tradeoff to accept.

---

## Step 1 — Fix MOG2 calibration ✅ DONE

**Status: COMPLETE.** Root causes found and fixed in `stage1_extract.py`:

- **Warm-up bug** (real, independent of tuning): passing a fixed
  `learningRate` to `bg_subtractor.apply()` overrode MOG2's own
  fast-then-decaying auto schedule, misclassifying most of the
  `history`-length warm-up window as motion. Fixed: use `learningRate=-1`
  (auto) while `frames_decoded <= config.mog2_history`, then switch to
  the deliberately slow `config.mog2_learning_rate` for steady state.
- **`varThreshold` was not the lever** — raising it from 16→100 only cut
  steady-state candidates 99.9%→96.0% on `7min.mp4`. Left at its default.
- **New `min_blob_area_ratio` gate** (default 0.0054, ~20,000px at
  2560x1440): gates on the single largest connected-component blob, not
  total flagged-area ratio. Calibrated against this project's only real
  ground truth (large-body-motion door events, minimum observed blob
  ≥31,900px → >2x safety margin). **Caveat, still standing:** not
  validated against small-gesture queries (e.g. "put a lid on a cup") —
  a query targeting fine-grained motion should pass a smaller value via
  `--min-event-area-ratio`.
- New `--auto-mask`, `--min-event-area-ratio` CLI flags added, threaded
  through `Stage1Config` and `_run_pipeline`.

**STOP GATE result:** Full pipeline re-run on `7min.mp4` as `run2`
(query: `"a person entering or leaving the coffee shop through the
door"`), **genuine blind human stage-5 review** (a Claude-Code-performed
review was done once by mistake during investigation and is preserved
read-only at `data/work/run2/stage5_claude_DO_NOT_USE_contaminated` —
never use it; this is exactly why the standing CLAUDE.md rule below
exists). Result: **Recall 6/6, Precision 19/19.** Stage 1 candidates:
6,043 (vs. run1's 12,592) — floor sampling also now actually fires (42
times) instead of never, confirming the warm-up/gate fix is working, not
just coincidentally recall-safe.

**Standing rule added to CLAUDE.md as a direct result of this step:**
Stage 5 review MUST be performed by the human via the actual interactive
tool, never reconstructed by Claude Code, even temporarily.

---

## Step 2 — Dynamic floor-sampling frequency from event duration ✅ DONE

**Status: COMPLETE.** Added `--min-event-duration-sec` to `run`/`extract`.
Derives `floor_interval_sec = min_event_duration_sec / 2` (new
`stage1_extract.derive_floor_interval_sec()`), clamped to never exceed
today's proven-safe default (5.0s) via `min(derived, Stage1Config.floor_interval_sec)`.
`--max-event-duration-sec` added as validation-only (warns if
`min > max`, does not drive sampling — a longer event doesn't need denser
sampling since motion detection already catches it regardless).

**Why /2, not exact Nyquist:** an event of duration D is guaranteed to
contain at least one floor sample if the interval is strictly less than
D — that's the bare minimum, and it's phase-dependent (a sample spaced
at exactly D apart can still straddle the event in the worst-case
alignment, since this isn't reconstructing a periodic signal, it's
guaranteeing a hit on a one-off window at an unknown phase). Halving
gives 2x margin below that bare minimum.

**Pre-implementation check:** the 46-47s "man leaves" event in run2's
stage1 manifest was caught **entirely by motion detection**, not floor
sampling — confirming floor sampling wasn't load-bearing for any of the
6 real events in this specific clip, before implementing anything.

**STOP GATE result:** `run3` (same query, `min_event_duration_sec=1.0` →
derived `floor_interval_sec=0.5`), genuine blind human review. **Recall
6/6, Precision 17/17.** Stage 1 candidates: 6,444 vs run2's 6,043 —
motion-detected candidates identical (6,001 both runs, confirming
floor_interval doesn't touch MOG2 at all); the +401 delta is entirely
additional floor samples (443 vs 42). Isolated clamp check:
`derive_floor_interval_sec(15.0) == 5.0` (7.5 correctly clamped down).

**Honest caveat, stated plainly and still true:** this result demonstrates
the change is **recall-safe** on this clip, not that it changed this
clip's outcome — floor sampling wasn't load-bearing for any of the 6 real
events here (confirmed in the pre-implementation check). The real test of
this feature is a future clip with a genuinely quiet stretch long enough
that motion detection alone would miss a short real event.

---

## Step 3 — Stage 2 dedup: FAISS binary Hamming index + vectorized hashing ✅ DONE — REJECTED, REVERTED

**Status: COMPLETE. Final verdict: explored, measured, and reverted.**
Stage 2 currently takes ~509.8s (run1 baseline) / 279.1s (run3, same
implementation, different candidate count) via a pure-Python per-frame
sliding-window pHash comparison loop. This step explored replacing it;
**the pure-Python loop is what actually runs today**, unchanged from
before this step.

**Pre-implementation profiling (300 real images) found the premise
wrong:** the pHash *computation* itself (DCT + median threshold) is only
**0.35%** of stage2's per-image cost. The real cost is image load (34%)
and PIL's full-resolution (2560x1440) grayscale+resize preprocessing
(65%, split across BGR→RGB conversion and the PIL `.convert('L').resize()`
call). "Vectorizing the hash computation," taken literally, could not
have delivered a meaningful speedup — there was almost nothing to
vectorize that mattered.

**The only real lever found (resize-first via cv2, before grayscale
conversion, so all later steps operate on a 32x32 array instead of
2560x1440) was validated and rejected on correctness grounds, with real
numbers, not a synthetic sample:**

- An average-bit-diff sanity check (300 images) looked safe: max 4 bits
  of divergence from `imagehash.phash()`'s exact output, well under the
  `hamming_threshold=8` cutoff.
- A **boundary-focused validation** — replaying run3's actual dedup
  cascade over its full ~6,444-candidate stage1 output, recording every
  one of the **14,645 real comparisons** the sliding-window algorithm
  actually made (not synthetic all-pairs) — told a different story:
  5,357 comparisons fell within ±2 of the threshold, and **297 of those
  flip decision in the dangerous direction** (a real "keep both,"
  distance 10, becomes an incorrect "merge," distance ≤8 under the fast
  hash — silently dropping a real frame). That's roughly **1 in 20** of
  all real comparisons, not a rare tail case. This directly violates
  SPEC.md's stated priority that over-merging is worse than
  under-merging. **Rejected.** See `stage2_dedup.compute_phash`'s
  docstring for this finding, so it isn't rediscovered from scratch.

**FAISS (`faiss.IndexBinaryIDMap` over `IndexBinaryFlat`) was implemented
as a structural swap regardless** — replacing the Hamming-search data
structure only, keeping `compute_phash` bit-exact — since it was
correctness-preserving (full test suite passed unchanged, kept-frame
count identical) and satisfied the original ask. **Measured result: it
made stage 2 slower, not faster** — 776.9s vs. 279.1s (run3, same
candidate count) / 509.8s (run1 baseline). Cause: the design searched the
*entire* growing index (up to 161 kept frames) on every one of 6,444
candidates via `range_search`, post-filtering to the real `window_size=5`
window in Python, rather than keeping the index itself bounded — combined
with FAISS's per-call Python-binding overhead across 6,444 individual
calls not being amortized at this tiny scale (a 5-entry window is not
where FAISS's advantage lives). **Reverted.** The `faiss-cpu` dependency
was removed from `pyproject.toml` entirely rather than left in as an
unused, misleading "we use FAISS" signal.

**STOP GATE result (post-revert):** reverted implementation confirmed to
reproduce `run4`'s exact kept-frame set (161 candidates, same frame
indices) at 279.1s-class wall-clock, full test suite green, `eval/evaluate.py`
re-confirmed **Recall 6/6, Precision 20/20** (unchanged from the FAISS
version, as expected — the revert removes an already-measured-neutral
structural wrapper, not a behavior change).

**Net conclusion: no speedup is available for stage 2 via hash-computation
or Hamming-search optimization at this scale.** I/O and image
preprocessing dominate cost; the only fast preprocessing path found is
measurably unsafe on this project's real footage.

### Step 3.5 (flagged, not pursued now) — reduce image I/O cost directly

If stage 2 speed ever needs revisiting, the next candidate isn't hash
computation — it's the fact that every stage round-trips through JPEG at
its boundary (stage 1 writes JPEGs, stage 2 reads them back via
`io_utils.load_frame_image`, which is itself ~34% of stage 2's cost per
the profiling above). Caching decoded arrays across stages (in-process,
or a lightweight on-disk format cheaper to decode than JPEG) instead of
re-decoding JPEG at every stage boundary is the more promising direction
— but this is a cross-stage architectural change (affects stage 1's
output contract, not just stage 2's internals), materially bigger in
scope than this step, and not validated at all yet. Flagged for later,
not something to pursue as part of Step 3.

---

## Known fragile ground-truth event — check here before treating a 5/6 as a new regression

**Event 4** ("a man leaves the coffee shop," 174–176s, frames ~5181/5194)
has now been the **sole point of recall failure in 2 of ~5 independent
end-to-end evaluations** across this project — the original `run2`
query-wording miss (Step 1's gate investigation) and `run5`'s stage-5
review variance (Step 4) — for two entirely different underlying
reasons, converging on the same event both times.

Why this specific event is fragile, structurally:
- It sits close to stage 3's SigLIP2 top-50 similarity cutoff (scores
  ~0.110, within ~0.02 of the observed cutoff in multiple runs) — small,
  otherwise-harmless changes upstream (query wording, dedup's exact
  survivor set) can push it across the boundary either way.
- It is **visually ambiguous even under careful human review**: the
  frame shows a man standing at the door, not clearly having crossed
  the threshold. Three independent blind reviews (`run2`, `run3`,
  `run4`) each kept at least one of the two frames; a fourth (`run5`)
  discarded both. This is genuine, defensible reviewer disagreement on
  an ambiguous frame, not an error either way.

**If a future run lands on 5/6 and the miss is specifically event 4,
check whether it traces to this same combination (near-cutoff stage 3
score + genuinely ambiguous framing) before treating it as a new
regression from whatever change is being validated.** It has never
once been attributable to a stage 1 or stage 2 change in this
project's history — both prior instances were stage 3 (query wording)
and stage 5 (review variance) respectively.

---

## Step 4 — Stage 1 decode: hardware-accelerated video decode (NVDEC) ✅ DONE — NVDEC REJECTED, PIVOTED TO RESOLUTION REDUCTION

**Status: COMPLETE.** NVDEC decode was evaluated and rejected; the real
lever turned out to be resolution, not decode acceleration.

**Smoke test: all three candidates fail cleanly without heavy setup.**
- `decord`: installs as a genuine Windows wheel, but the PyPI build is
  CPU-only — `decord.gpu(0)` fails with `CUDA not enabled`. GPU support
  requires building from source with CUDA, exactly the heavy setup to
  avoid.
- `PyNvVideoCodec`: installs but fails to even *import* — `DLL load
  failed` (missing NVIDIA Video Codec SDK runtime DLLs, not bundled
  with the pip wheel).
- `ffmpegcv`: installs but fails to import — requires an external
  `ffmpeg` binary, and this machine has none on PATH at all.

None viable. Per the documented fallback, no dependency was added; all
three were installed only for the smoke test and removed afterward
(`pyproject.toml` untouched).

**Profiling pivot (2000 real frames from `data/raw/7min.mp4`) found the
premise wrong, the same way Step 3's did:** decode (`grab`+`retrieve`)
is only **22.4%** of stage 1's per-frame cost. `MOG2.apply()` +
threshold/morph + `connectedComponents` is **77.6%**, and all three
scale with pixel count. Even a perfect NVDEC integration would have
capped out around a 22% reduction — the real lever is reducing the
pixel count fed to MOG2, not accelerating decode.

**Pivoted to `Stage1Config.downscale_factor`**: resizes the frame fed to
MOG2/blob-detection only (saved candidate images always stay full
resolution) — every frame is still processed in order, nothing skipped,
so MOG2's background-model continuity is fully preserved (unlike
frame-skipping, which would reintroduce the exact warm-up-corruption
hazard Step 1 already fixed once).

**Validation, in order, before touching `stage1_extract.py`:**
1. Full real-clip comparison (all 12,600 frames, 0.5x downscale) against
   the un-downscaled baseline: **every one of the 6 ground-truth
   events' motion candidates identical by exact frame_index** — not
   just matching counts.
2. Boundary-focused check (frames with full-res `largest_blob_ratio`
   within 0.004–0.007 of the 0.0054 threshold, matching the discipline
   Step 3 established): a real **~11% decision-flip rate** among those
   1,224 borderline frames (134 flips), split 68 dangerous-direction
   (full-res correctly flags motion, half-res misses it) vs. 66 safe.
   Of the 68 dangerous flips, **2 landed inside a real event's
   tolerance-widened zone** (not its exact labeled window) — narrow and
   bounded to the threshold boundary, not a general degradation, and
   confirmed not to change that event's actual recall outcome in the
   full pipeline run.
3. Only after both checks: implemented for real, existing stage 1 test
   suite passes unchanged (19/19), full suite green.

**STOP GATE result (`run5`, `min_event_duration_sec=1.0` +
`downscale_factor=0.5`, same query as `run2`–`run4`):**

| | value |
|---|---|
| Stage 1 wall-clock | **402.5s** |
| vs. run1's original baseline (910.6s) | **2.26x** |
| vs. run3/run4's full-res, same `min_event_duration_sec=1.0` (~1179–1237s) | **~2.9–3.1x** |
| Stage 1 candidates | 6,448 (6,005 motion, 443 floor) vs. run3's 6,444 (6,001 motion, 443 floor) — floor count identical (time-based, resolution-independent by design) |
| Recall | **5/6** |
| Precision | **15/15 = 100%** |

The 5/6 is real and was investigated, not smoothed over — see "Known
fragile ground-truth event" above. It traces entirely to stage 5 (this
run's blind reviewer discarded both event-4 candidate frames, which
survived stage 1–3 with byte-identical frame indices and similarity
scores to prior runs), not to `downscale_factor`. Per the project's
standing rule, this was not re-litigated with another review pass.

**Default recommendation: keep `downscale_factor=1.0` (off) as the
`Stage1Config` default**, consistent with this project's established
pattern for every other calibrated-but-narrow-scope knob
(`min_blob_area_ratio`, `min_event_duration_sec` before it) — but **0.5
is the validated, recommended value for real runs on similar footage**,
given the boundary risk is real (11% flip rate near the threshold) even
though it wasn't shown to affect this clip's actual recall outcome. Pass
`--downscale-factor 0.5` explicitly, the same way a query targeting
fine-grained motion should pass `--min-event-area-ratio` explicitly.

---

## Step 5 — Stage 4 VLM throughput: the strategic decision

**Status: COMPLETE. (a) tested and REVERTED (modest gain, measurable
reliability cost). (b) infrastructure feasibility CONFIRMED, but the
actual migration attempt hit a hard, unconditional environment blocker
(CUDA UVA unavailable under WSL2 GPU passthrough) — REJECTED for this
specific vLLM release. Sequential Ollama remains stage 4's
implementation.**

**(a) result:** raised `OLLAMA_NUM_PARALLEL=4` (required a full server
restart with the env var set — Ollama doesn't support changing this on
a running server), fired 4-way concurrent verification requests against
all 50 real `run5`/stage4 frames using stage4_verify's exact request
logic, compared against the already-known sequential baseline (both
throughput and, critically, per-frame verdict accuracy — not just
whether it crashed):

| | value |
|---|---|
| Concurrent wall-clock (4 workers) | 539.8s |
| Sequential baseline (run5) | 872.0s |
| Speedup | **1.62x** — far short of ~4x naive linear scaling |
| GPU during test | 96% util, 4.9GB/6.1GB used — little headroom |
| Yes/no verdict flips | **0** — every frame's semantic verdict matched exactly |
| Error-rate change | 8→10 errors (2 frames that verified cleanly sequentially became parse-failure errors under concurrency) — same known JSON-parse/context-budget failure mode, triggered on 2 more frames |

**Verdict: reverted.** 1.62x is a modest gain, not this step's hoped-for
biggest win, and it came with a measured reliability cost (25% relative
increase in error rate) on a card already pinned near its VRAM ceiling.
Ollama was restarted back to its default (no `OLLAMA_NUM_PARALLEL`
override) — the sequential path remains the validated, more reliable
one for actual pipeline runs.

**(b) feasibility check** (per the STOP GATE: investigate feasibility
before committing real implementation time to a migration):
- WSL2: present (`Ubuntu-22.04` and `docker-desktop` distros, both
  WSL version 2).
- Docker Desktop: installed, daemon reachable (required a manual
  restart mid-session — it wasn't running at first check).
- **GPU passthrough confirmed working**: `docker run --gpus all
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` correctly sees the
  RTX 4050 (driver 577.05, CUDA 12.9) from inside a container.
- VRAM headroom note: at test time the card showed only ~1.2GB free
  (4.9GB held by Ollama's currently-loaded `llama-server.exe`) — not a
  permanent constraint, since a real migration retires Ollama, freeing
  the full ~6.1GB for the new server + AWQ model.

All three infrastructure feasibility questions came back positive.

**The actual migration attempt, once undertaken, hit a genuine
environment blocker — reported here in full, including two mistakes
made along the way and corrected:**

1. **First mistake:** pulled `vllm/vllm-openai:latest` (30.8GB) without
   checking its CUDA requirement first. It requires CUDA ≥13.0; this
   machine's driver only supports CUDA 12.9 (`nvidia-container-cli:
   requirement error: unsatisfied condition: cuda>=13.0`). Removed. The
   correct fix: found `v0.27.1-cu129-ubuntu2404`, a versioned tag
   explicitly built for CUDA 12.9, via the Docker Hub tags API —
   verified via `docker manifest inspect` (11.44GB compressed, ~26GB on
   disk) and the model repo's real size (3.42GB, confirmed via the HF
   API) *before* pulling, per the explicit instruction not to repeat
   the compatibility mistake.
2. **Disk space detour:** removing the bad image only freed space at
   Docker's internal accounting level, not the Windows host filesystem
   — WSL2's backing `.vhdx` doesn't auto-shrink. `Optimize-VHD`
   (Hyper-V) isn't available at all on this machine (Windows 11 Home
   has no Hyper-V). WSL's own built-in sparse-VHD compaction
   (`wsl --manage --set-sparse`) is blocked by Microsoft's own
   corruption-risk warning without `--allow-unsafe` — not used, given
   the blast radius (all Docker data, not just this test). Resolved via
   Docker Desktop's own "Reset to factory defaults" (Settings →
   Troubleshoot), which reclaimed the disk safely (17GB → 130GB free)
   since there was nothing left worth keeping in Docker's store at that
   point.
3. **VRAM debugging:** the correct `cu129` image still failed to start
   vLLM — `ValueError: Free memory on device cuda:0 (4.96/6.0 GiB) ...
   is less than desired GPU memory utilization (0.85, 5.1 GiB)`. This
   was **misread once as "4.96GB used"** when it actually means "4.96GB
   free" (only ~1.04GB reserved, almost certainly Windows' own WDDM
   display overhead, not Ollama) — corrected, and lowering
   `--gpu-memory-utilization` to 0.75 resolved it cleanly. (An Ollama
   `llama-server.exe` VRAM-lingering issue was also hit and fixed along
   the way — `ollama stop` didn't fully release VRAM; killing the
   process directly did — but this turned out to be a red herring for
   the actual blocker.)
4. **The real, unconditional blocker:** with memory sized correctly,
   vLLM's engine failed with `RuntimeError: UVA is not available` inside
   `GPUModelRunnerV2`'s `StagedWriteTensor` / `UvaBuffer` init — CUDA
   Unified Virtual Addressing, which this vLLM release's "V2 Model
   Runner" requires unconditionally. Tried `--enforce-eager` (disables
   CUDA graph capture/compilation) as a targeted attempt to route around
   it — **identical failure, same code path**, confirming this isn't a
   config-tunable issue. WSL2's GPU passthrough layer doesn't provide
   full parity with bare-metal Linux CUDA for every advanced memory-
   management feature, and UVA is one of the gaps; the earlier plain
   `nvidia-smi` passthrough smoke test didn't exercise this specific
   capability, so it wasn't and couldn't have been caught earlier.

**Verdict: REJECTED for this environment, this vLLM release.** The
migration is not a config problem to keep tuning — it's a genuine
WSL2-vs-bare-metal CUDA capability gap. Per the explicit lesson from
this step (verify compatibility before pulling anything else heavy),
this was not chased further into unverified territory: an older vLLM
tag (whose logs suggest it may have used a different, non-UVA-dependent
model runner, since "V2 Model Runner" appears to be a recent addition)
or SGLang are the next unverified candidates *if* this is ever
revisited — genuinely unverified, not assumed to work, and each would
need the same compatibility-first discipline applied before any further
heavy pull.

**Stage 4's implementation is unchanged: sequential Ollama calls to
qwen3-vl:4b**, exactly as validated through runs 2–5. No new dependency
was added to `pyproject.toml`; all Docker images pulled during this
investigation were smoke-test artifacts, not a persisted part of the
pipeline.

Stage 4 currently takes 661.3s (~13.2s/frame, strictly sequential
Ollama calls). Two paths, evaluate the cheap one first:

**(a) Lower risk:** raise `OLLAMA_NUM_PARALLEL` and fire concurrent
requests to the existing Ollama server. Vision concurrency is reported
flaky/VRAM-gated on a 6GB card — must be measured, not assumed.

**(b) Higher reward, higher setup cost:** migrate to a
continuous-batching server (vLLM or SGLang) with an AWQ-quantized small
VLM — **Qwen2.5-VL-3B-AWQ specifically, not Qwen3-VL** (no published
AWQ accuracy numbers exist for Qwen3-VL yet). Likely needs WSL2 or
Docker on this Windows machine.

**Before committing to either:**
1. Test (a) first, since it's cheap: raise `OLLAMA_NUM_PARALLEL`, fire
   4–8 concurrent verification requests against real Absar frames
   (reuse existing frames from `data/work/run1/stage3` or `stage4`),
   measure real wall-clock throughput vs. today's sequential baseline.
   Check whether verdicts stay accurate under concurrency — compare
   against the already-known verdicts from the completed real run, not
   just "did it crash."
2. Only if (a)'s gain is small/unstable, investigate (b)'s setup
   feasibility on this machine (WSL2 availability? Docker installed?)
   before committing real implementation time.
3. Whichever path is chosen, re-check specifically the two frames
   (`5181`, `12053`) flagged earlier in this project as needing manual
   verification — a different model/quantization could change those
   verdicts, and that's worth knowing explicitly, not discovered later.

**Note from Step 1-3's work:** stage 4's ~8/50-frame deterministic
context-budget error rate (qwen3-vl's "thinking" chain exhausting
`num_ctx=4096` before emitting the JSON answer, reproducible at
`temperature=0`) is a separate, already-observed reliability issue
worth folding into this step's investigation — it doesn't block recall
(stage 5 shows every candidate regardless of stage 4's verdict) but it
undermines stage 4's usefulness as a review aid.

**STOP GATE:** Report the concurrency smoke test (a) result **first** —
do not plan a vLLM/SGLang migration until it's known whether the cheap
option is "good enough." If a path is implemented: re-run the full
pipeline + `eval/evaluate.py`, confirm recall/precision hasn't
regressed, report new stage 4 wall-clock time. **Wait for confirmation
before Step 6.**

---

## Step 6 — Multiprocessing across Stage 1/2 (only if still needed)

Check `timing.json` from the latest full run first — only proceed if
stage 1 and/or stage 2 are *still* meaningful bottlenecks after steps
1–4. If they're no longer the dominant cost, skip this step and go to
the final comparison instead.

If still needed: split the video into time-chunks processed by
separate worker processes (multiprocessing, not threading — GIL
prevents true parallelism for CPU-bound work), using a
producer-consumer queue so decode and compute overlap.

**Critical correctness concern:** MOG2 needs a continuous frame stream
to maintain its background model — splitting into chunks means each
worker's MOG2 instance starts cold at its chunk boundary. Plan a
short warm-up overlap before each chunk's "real" start (discarded from
output) so chunking doesn't introduce false motion at boundaries or
miss real motion right after one.

**STOP GATE:** Implement, re-run the full pipeline + `eval/evaluate.py`
against `data/raw/7min.mp4`. Recall must not regress — specifically
check no ground-truth event near a chunk boundary was affected. Report
new stage 1/2 wall-clock times. **Wait for confirmation before the
final comparison.**

---

## Final comparison

Run the full pipeline against `data/raw/7min.mp4` one more time end to
end, using `timing.json` for exact per-stage timings (not reconstructed
approximations).

Produce a before/after table:

| Stage | Original | After Step 1-3 | After Step 4 (run5) |
|---|---|---|---|
| Stage 1 | 910.6s | ~1179-1237s (min_blob_area_ratio gate adds connected-components cost, no speed optimization yet) | **402.5s** (downscale_factor=0.5 -- 2.26x vs original, ~2.9-3.1x vs the Step 1-3 baseline at the same min_event_duration_sec=1.0) |
| Stage 2 | 509.8s | 279.1s (pure-Python loop, unchanged from original design -- FAISS explored and reverted, see Step 3) | 259.0s (unchanged implementation; input candidate count shifted slightly with downscaled stage 1) |
| Stage 3 | 50.3s | ~45-95s | 49.5s |
| Stage 4 | 661.3s | ~830-2075s (variance from Ollama backend reliability, see Step 5's note) | 872.0s (same variance, not yet addressed -- that's Step 5) |
| Stage 5 (review) | -- | -- | 46.6s (this pass; varies by reviewer/session) |
| Total (incl. review) | ~54 min | ~41-54 min | **~27.2 min** |

Recall/precision after Step 4: 5/6, 15/15 -- investigated and root-caused
to stage 5 review variance on a known-fragile event (see "Known fragile
ground-truth event" above), not a regression from Step 4's actual change.
Stage 1's own claim (downscaling doesn't change which frames get flagged
as motion) was independently verified before any human review happened.

Confirm `eval/evaluate.py`'s final recall/precision against
`eval/ground_truth.json` is unchanged or improved, not regressed.

Extrapolate the new per-stage times linearly to a 60-minute clip and
report the projected total — this is the number that answers whether
"8min clip in 30+ min" has actually become practical for 1hr+ footage.
