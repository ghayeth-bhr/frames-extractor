"""FastAPI app for the "assets" webapp: create assets, upload videos into
them, browse extracted frames, and search a combined multi-source index.

See worker.py for why upload processing and search both go through a
single-worker GPU-job lane, and multi_source_index.py for the combined
index itself.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import re
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import cli, models, stage1_extract, stage2_dedup

from . import storage, worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A source's status is just a string on disk -- nothing re-submits an
    # in-flight job on startup, so anything left mid-pipeline from a prior
    # process (crash, restart, Ctrl+C) is orphaned: no thread anywhere is
    # ever going to touch it again. Without this, it sits there forever
    # looking like it's still progressing, indistinguishable from a real
    # hang (see the incident this was written to fix).
    recovered = storage.mark_orphaned_sources_as_error()
    if recovered:
        print(f"[startup] marked {len(recovered)} orphaned source(s) as error (interrupted by a prior restart):")
        for asset_id, source_id in recovered:
            print(f"  asset={asset_id} source={source_id}")
    yield


app = FastAPI(title="frames-extractor", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def ui_asset_list() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/assets/{asset_id}/ui")
def ui_asset_detail(asset_id: str) -> FileResponse:
    # Deliberately does NOT validate asset_id here -- always serves the same
    # static shell; a bad/missing id surfaces as a 404 from the page's own
    # client-side fetch(GET /assets/{id}) call, handled in JS, not blocked
    # at this route.
    return FileResponse(STATIC_DIR / "asset.html")


_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_FRAME_FILENAME_RE = re.compile(r"^frame_\d{6}\.jpg$")


def _validate_id(value: str, kind: str) -> str:
    """asset_id/source_id are URL path segments that feed directly into
    filesystem paths elsewhere (esp. get_frame_image) -- reject anything
    that isn't exactly a uuid4().hex before it's used for a lookup."""
    if not _ID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"invalid {kind}: {value!r}")
    return value


def _require_asset(asset_id: str) -> storage.AssetMeta:
    _validate_id(asset_id, "asset_id")
    try:
        return storage.load_asset_meta(asset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no asset {asset_id!r}")


def _asset_frame_count(asset_id: str) -> int:
    embeds_path = storage.index_dir(asset_id) / "embeddings.npy"
    if not embeds_path.exists():
        return 0
    return int(np.load(embeds_path, mmap_mode="r").shape[0])


class CreateAssetRequest(BaseModel):
    name: str


class AssetSummary(BaseModel):
    asset_id: str
    name: str
    created_at: str
    source_count: int
    frame_count: int


class SourceStatusResponse(BaseModel):
    source_id: str
    filename: str
    status: str
    added_at: str
    error_message: str | None = None
    candidate_count: int | None = None
    settings: dict | None = None


class AssetDetailResponse(BaseModel):
    asset_id: str
    name: str
    created_at: str
    sources: list[SourceStatusResponse]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 50


class SearchResultItem(BaseModel):
    source_id: str
    frame_index: int
    timestamp_ms: float
    similarity_score: float | None
    image_url: str


class FrameItem(BaseModel):
    source_id: str
    frame_index: int
    timestamp_ms: float
    reason: str
    image_url: str


class SourceSummary(BaseModel):
    source_id: str
    status: str
    filename: str
    frame_count: int


class FramesResponse(BaseModel):
    sources: list[SourceSummary]
    frames: list[FrameItem]
    page: int
    page_size: int
    total_frames: int


class FrameRef(BaseModel):
    source_id: str
    filename: str


class DownloadFramesRequest(BaseModel):
    frames: list[FrameRef]


@app.post("/assets", response_model=AssetSummary)
def create_asset(body: CreateAssetRequest) -> AssetSummary:
    meta = storage.create_asset(body.name)
    return AssetSummary(
        asset_id=meta.asset_id, name=meta.name, created_at=meta.created_at, source_count=0, frame_count=0
    )


@app.get("/assets", response_model=list[AssetSummary])
def list_assets() -> list[AssetSummary]:
    return [
        AssetSummary(
            asset_id=meta.asset_id,
            name=meta.name,
            created_at=meta.created_at,
            source_count=len(meta.sources),
            frame_count=_asset_frame_count(meta.asset_id),
        )
        for meta in storage.list_assets()
    ]


@app.get("/assets/{asset_id}", response_model=AssetDetailResponse)
def get_asset(asset_id: str) -> AssetDetailResponse:
    meta = _require_asset(asset_id)
    return AssetDetailResponse(
        asset_id=meta.asset_id,
        name=meta.name,
        created_at=meta.created_at,
        sources=[SourceStatusResponse(**s.__dict__) for s in meta.sources],
    )


@app.post("/assets/{asset_id}/upload", response_model=SourceStatusResponse)
async def upload_source(
    asset_id: str,
    file: UploadFile = File(...),
    downscale_factor: float | None = Form(None),
    min_event_area_ratio: float | None = Form(None),
    auto_mask: bool = Form(False),
    dedup_hamming_threshold: int | None = Form(None),
    dedup_window_size: int | None = Form(None),
    cross_source_dedup: bool = Form(False),
) -> SourceStatusResponse:
    """Production-speed knobs, all optional (unset = today's exact default
    behavior) -- reuses cli.py's own _stage1_config_from_flags/
    _stage2_config_from_flags (the same functions `extract`/`dedup`/`index`
    already build their configs with) rather than re-deriving this logic a
    third time. The CLI flags this form doesn't expose (mask_regions,
    min/max_event_duration_sec, sampling_mode, sample_interval_sec) are
    always passed as None/not-set here -- not reachable from this endpoint.

    NOTE: Stage3Config.top_k is deliberately NOT accepted here -- it has no
    effect on indexing (add_to_index()/build_index() embed everything, no
    truncation), it only matters at search time, where
    POST /assets/{id}/search already accepts it per-request.

    cross_source_dedup (default False): passed straight through to
    add_to_index() -- see its docstring. Recorded in `settings` so a later
    retry of THIS source reuses the same choice rather than defaulting back
    to off.
    """
    _require_asset(asset_id)

    stage1_config = cli._stage1_config_from_flags(
        mask_regions=None,
        auto_mask=auto_mask,
        min_event_area_ratio=min_event_area_ratio,
        min_event_duration_sec=None,
        max_event_duration_sec=None,
        downscale_factor=downscale_factor,
        sampling_mode=None,
        sample_interval_sec=None,
    )
    # Always resolved to a concrete object (never left None) -- both so the
    # worker always has a real config to pass to dedup(), and so `settings`
    # below records the actual values used, not a null.
    stage2_config = cli._stage2_config_from_flags(dedup_hamming_threshold, dedup_window_size) or (
        stage2_dedup.Stage2Config()
    )

    ext = Path(file.filename or "").suffix or ".mp4"
    source = storage.add_source(
        asset_id,
        file.filename or "upload",
        settings={
            "stage1_config": dataclasses.asdict(stage1_config),
            "stage2_config": dataclasses.asdict(stage2_config),
            "cross_source_dedup": cross_source_dedup,
        },
    )
    video_dir = storage.source_dir(asset_id, source.source_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"video{ext}"
    contents = await file.read()
    video_path.write_bytes(contents)

    worker.submit_upload_job(
        asset_id, source.source_id, video_path, stage1_config, stage2_config, cross_source_dedup=cross_source_dedup
    )

    return SourceStatusResponse(**source.__dict__)


@app.get("/assets/{asset_id}/sources/{source_id}/status", response_model=SourceStatusResponse)
def get_source_status(asset_id: str, source_id: str) -> SourceStatusResponse:
    _require_asset(asset_id)
    _validate_id(source_id, "source_id")
    try:
        source = storage.get_source(asset_id, source_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no source {source_id!r} in asset {asset_id!r}")
    return SourceStatusResponse(**source.__dict__)


@app.post("/assets/{asset_id}/sources/{source_id}/retry", response_model=SourceStatusResponse)
def retry_source(asset_id: str, source_id: str) -> SourceStatusResponse:
    """Reuses the exact upload-processing path (_process_upload), pointed
    at the already-on-disk video file instead of a fresh one, and rebuilds
    the ORIGINAL Stage1Config/Stage2Config from `settings` (recorded at
    upload time) rather than asking the caller to re-specify them. If
    stage1 already completed before the interruption, _process_upload's own
    check skips re-running extract() -- see worker._stage1_already_complete.
    """
    _require_asset(asset_id)
    _validate_id(source_id, "source_id")
    try:
        source = storage.get_source(asset_id, source_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no source {source_id!r} in asset {asset_id!r}")

    if source.status != "error":
        raise HTTPException(status_code=400, detail="can only retry a source currently in 'error' status")

    video_dir = storage.source_dir(asset_id, source_id)
    video_candidates = sorted(video_dir.glob("video.*"))
    if not video_candidates:
        raise HTTPException(
            status_code=500, detail="original video file missing on disk -- cannot retry, re-upload instead"
        )
    video_path = video_candidates[0]

    settings = source.settings or {}
    stage1_config = stage1_extract.Stage1Config(**settings.get("stage1_config", {}))
    stage2_config = stage2_dedup.Stage2Config(**settings.get("stage2_config", {}))
    cross_source_dedup = settings.get("cross_source_dedup", False)

    storage.reset_source_for_retry(asset_id, source_id)
    worker.submit_upload_job(
        asset_id, source_id, video_path, stage1_config, stage2_config, cross_source_dedup=cross_source_dedup
    )

    return SourceStatusResponse(**storage.get_source(asset_id, source_id).__dict__)


@app.get("/assets/{asset_id}/frames", response_model=FramesResponse)
def list_frames(asset_id: str, page: int = 1, page_size: int = 50) -> FramesResponse:
    """Every source appears in `sources` regardless of status, so the
    frontend can show "still processing" for anything not yet ready --
    but only sources whose stage2 output already exists on disk (dedup
    done, real images present -- checked by file existence, not by
    trusting the possibly-stale status string) contribute frames, so
    browsing never waits on the combined index/embedding step finishing.
    """
    meta = _require_asset(asset_id)
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    if page_size < 1 or page_size > 500:
        raise HTTPException(status_code=400, detail="page_size must be between 1 and 500")

    all_frames: list[FrameItem] = []
    source_summaries: list[SourceSummary] = []
    for source in meta.sources:
        stage2_dir = storage.source_stage2_dir(asset_id, source.source_id)
        candidates_path = stage2_dir / "candidates.json"
        frame_count = 0
        if candidates_path.exists():
            candidates = sorted(models.load_candidates(candidates_path), key=lambda c: c.frame_index)
            frame_count = len(candidates)
            for c in candidates:
                all_frames.append(
                    FrameItem(
                        source_id=source.source_id,
                        frame_index=c.frame_index,
                        timestamp_ms=c.timestamp_ms,
                        reason=c.reason,
                        image_url=f"/frames/{asset_id}/{source.source_id}/{c.image_path.name}",
                    )
                )
        source_summaries.append(
            SourceSummary(
                source_id=source.source_id,
                status=source.status,
                filename=source.filename,
                frame_count=frame_count,
            )
        )

    total = len(all_frames)
    start = (page - 1) * page_size
    page_frames = all_frames[start : start + page_size]

    return FramesResponse(
        sources=source_summaries,
        frames=page_frames,
        page=page,
        page_size=page_size,
        total_frames=total,
    )


@app.post("/assets/{asset_id}/search", response_model=list[SearchResultItem])
async def search_asset(asset_id: str, body: SearchRequest) -> list[SearchResultItem]:
    _require_asset(asset_id)
    idx_dir = storage.index_dir(asset_id)
    if not (idx_dir / "embeddings.npy").exists():
        raise HTTPException(
            status_code=400,
            detail="asset has no searchable index yet -- upload and wait for at least one source to finish embedding",
        )

    # Routed through the same single-worker GPU lane as upload processing
    # (see worker.py) -- never concurrent with an embedding job.
    future = worker.submit_search_job(idx_dir, body.query, body.top_k)
    try:
        results = await asyncio.wrap_future(future)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [
        SearchResultItem(
            source_id=f.source_id,
            frame_index=f.frame_index,
            timestamp_ms=f.timestamp_ms,
            similarity_score=f.similarity_score,
            image_url=f"/frames/{asset_id}/{f.source_id}/{f.image_path.name}",
        )
        for f in results
    ]


@app.get("/frames/{asset_id}/{source_id}/{frame_filename}")
def get_frame_image(asset_id: str, source_id: str, frame_filename: str) -> FileResponse:
    _validate_id(asset_id, "asset_id")
    _validate_id(source_id, "source_id")
    if not _FRAME_FILENAME_RE.match(frame_filename):
        raise HTTPException(status_code=404, detail="not found")

    stage2_dir = storage.source_stage2_dir(asset_id, source_id).resolve()
    candidate_path = (stage2_dir / frame_filename).resolve()
    # Defense in depth: the regex above already rejects "..", but confirm
    # the resolved path is still exactly inside stage2_dir before serving.
    if candidate_path.parent != stage2_dir or not candidate_path.is_file():
        raise HTTPException(status_code=404, detail="not found")

    return FileResponse(candidate_path)


_MAX_DOWNLOAD_FRAMES = 500


@app.post("/assets/{asset_id}/frames/download")
def download_frames(asset_id: str, body: DownloadFramesRequest) -> StreamingResponse:
    """Bundles the caller's selected frames (gallery or search results) into
    a single in-memory zip -- same per-frame path validation as
    get_frame_image (regex + resolved-path containment check), just looped
    over a list instead of one URL's path segments."""
    _require_asset(asset_id)
    if not body.frames:
        raise HTTPException(status_code=400, detail="no frames selected")
    if len(body.frames) > _MAX_DOWNLOAD_FRAMES:
        raise HTTPException(
            status_code=400, detail=f"too many frames selected (max {_MAX_DOWNLOAD_FRAMES})"
        )

    buf = io.BytesIO()
    seen_arcnames: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for ref in body.frames:
            _validate_id(ref.source_id, "source_id")
            if not _FRAME_FILENAME_RE.match(ref.filename):
                raise HTTPException(status_code=400, detail=f"invalid frame filename: {ref.filename!r}")

            stage2_dir = storage.source_stage2_dir(asset_id, ref.source_id).resolve()
            frame_path = (stage2_dir / ref.filename).resolve()
            if frame_path.parent != stage2_dir or not frame_path.is_file():
                raise HTTPException(status_code=404, detail=f"frame not found: {ref.source_id}/{ref.filename}")

            arcname = f"{ref.source_id}_{ref.filename}"
            if arcname in seen_arcnames:
                continue  # caller sent a duplicate selection -- include it once, not twice
            seen_arcnames.add(arcname)
            zf.write(frame_path, arcname=arcname)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="frames_{asset_id}.zip"'},
    )
