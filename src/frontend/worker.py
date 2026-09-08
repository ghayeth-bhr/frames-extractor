"""Single-worker GPU-job lane.

Serializes EVERY CUDA-touching operation this server can trigger --
embedding during upload processing, and search -- through one
concurrent.futures.ThreadPoolExecutor(max_workers=1), so two never run
concurrently. This project has a documented history of VRAM-contention
crashes between concurrent GPU consumers (see cli.py's module docstring);
a bare FastAPI BackgroundTasks call gives no such serialization on its own.

Upload processing (_process_upload) runs stage 1 (extract) and stage 2
(dedup) IN-PROCESS -- both are CPU-only, no CUDA risk -- but shells out to
the `add-to-index` CLI subcommand as a SEPARATE SUBPROCESS for the
embedding step, reusing the exact isolation mechanism `run`/`index` already
rely on: this server process can stay alive for hours/days across many
uploads, and this project has explicitly rejected in-process
torch.cuda.empty_cache() as an incomplete mitigation for VRAM fragmentation
(only a process exit reliably clears it) -- an untested risk this avoids
entirely by never loading SigLIP2 in the server's own long-lived process.

Search, in contrast, DOES load the model in-process (via
multi_source_index.search_multi_source_index) -- it's a short-lived,
single call rather than a repeated-forever-in-a-long-lived-process
pattern, so the fragmentation concern above doesn't apply the same way;
it still goes through the same single-worker executor purely for
serialization against upload jobs, not for the same process-lifetime reason.
"""

from __future__ import annotations

import subprocess
import sys
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from backend import models, stage1_extract, stage2_dedup
from vectordb import multi_source_index

from . import storage

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")


def _run_add_to_index_subprocess(
    index_dir: Path, stage2_dir: Path, source_id: str, *, cross_source_dedup: bool = False
) -> None:
    args = [
        sys.executable,
        "-m",
        "backend",
        "add-to-index",
        "--index",
        str(index_dir),
        "--stage2-dir",
        str(stage2_dir),
        "--source-id",
        source_id,
    ]
    if cross_source_dedup:
        args.append("--cross-source-dedup")
    subprocess.run(args, check=True)


def _stage1_already_complete(stage1_dir: Path) -> bool:
    """True only if stage1's candidates.json exists AND actually loads as
    real Candidate objects -- mirrors the corrupted-frame incident earlier
    in this project (a same-byte-count-but-garbled file from an interrupted
    job): existence alone was NOT trustworthy there, and isn't here either.
    A parse failure is treated identically to the file never having existed
    -- extract() gets re-run from scratch, no partial trust."""
    candidates_path = stage1_dir / "candidates.json"
    if not candidates_path.exists():
        return False
    try:
        models.load_candidates(candidates_path)
        return True
    except Exception:
        return False


def _process_upload(
    asset_id: str,
    source_id: str,
    video_path: Path,
    stage1_config: "stage1_extract.Stage1Config",
    stage2_config: "stage2_dedup.Stage2Config",
    *,
    cross_source_dedup: bool = False,
) -> None:
    """Never raises out of the worker thread -- any failure is recorded as
    this source's status=error with the exception details, never silently
    dropped, matching this project's recall-biased "never silently drop"
    ethos applied here to job failures instead of frames.

    stage1_config/stage2_config are already fully resolved by the caller
    (app.py, reusing cli.py's own _stage1_config_from_flags/
    _stage2_config_from_flags rather than re-deriving them here) -- this
    function just uses them, it doesn't build them.

    Also doubles as the retry path (see app.py's retry_source endpoint):
    if stage1_dir already has a genuinely complete candidates.json (e.g.
    this source previously got interrupted mid-dedup, after extract() had
    already finished), extract() is skipped and dedup() runs straight
    against the existing stage1 output -- dedup() is cheap and
    deterministic to redo, and add_to_index() is already idempotent per
    source_id, so nothing downstream needs special-casing for a retry."""
    try:
        stage1_dir = storage.source_stage1_dir(asset_id, source_id)
        if not _stage1_already_complete(stage1_dir):
            storage.update_source(asset_id, source_id, status="extracting")
            stage1_extract.extract(video_path, stage1_dir, stage1_config)

        storage.update_source(asset_id, source_id, status="deduping")
        stage2_dir = storage.source_stage2_dir(asset_id, source_id)
        candidates = stage2_dedup.dedup(stage1_dir, stage2_dir, stage2_config)

        storage.update_source(asset_id, source_id, status="embedding", candidate_count=len(candidates))
        idx_dir = storage.index_dir(asset_id)
        _run_add_to_index_subprocess(idx_dir, stage2_dir, source_id, cross_source_dedup=cross_source_dedup)

        storage.update_source(asset_id, source_id, status="ready")
    except Exception as e:  # noqa: BLE001 -- must never crash the worker thread silently
        storage.update_source(
            asset_id,
            source_id,
            status="error",
            error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def submit_upload_job(
    asset_id: str,
    source_id: str,
    video_path: Path,
    stage1_config: "stage1_extract.Stage1Config",
    stage2_config: "stage2_dedup.Stage2Config",
    *,
    cross_source_dedup: bool = False,
) -> None:
    """Fire-and-forget -- the upload endpoint does not wait on this; status
    is polled separately via GET /assets/{id}/sources/{id}/status."""
    _executor.submit(
        _process_upload,
        asset_id,
        source_id,
        video_path,
        stage1_config,
        stage2_config,
        cross_source_dedup=cross_source_dedup,
    )


def submit_search_job(index_dir: Path, query: str, top_k: int) -> Future:
    """Returns a Future -- the search endpoint DOES await this (via
    asyncio.wrap_future), since a search request must return results
    synchronously, unlike fire-and-forget upload processing."""
    return _executor.submit(multi_source_index.search_multi_source_index, index_dir, query, top_k)
