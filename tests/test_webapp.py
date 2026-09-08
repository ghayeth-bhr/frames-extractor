"""API tests for the webapp -- FastAPI TestClient, NO real GPU/Ollama
dependency: the worker's expensive steps (extract/dedup/add-to-index
subprocess, and multi_source_index.search_multi_source_index) are mocked,
matching this project's "mock what's expensive, test the real logic
underneath" testing philosophy (see test_stage4_verify.py's own framing).
Real logic under test: asset/source lifecycle, status transitions,
meta.json persistence, pagination, search request/response contract, and
path-traversal rejection for image serving.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import models
from backend.models import Candidate, IndexedFrame
from frontend import storage, worker
from frontend.app import app, get_frame_image


@pytest.fixture(autouse=True)
def _isolated_assets_root(tmp_path, monkeypatch):
    # Every test gets its own data/assets root -- never touches the real one.
    monkeypatch.setattr(storage, "ASSETS_ROOT", tmp_path / "assets")
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _make_upload_run_synchronously(monkeypatch):
    """Patches submit_upload_job to run _process_upload inline in the
    calling thread instead of the real background executor -- needed so
    tests can assert on post-processing state deterministically without a
    real wait. The pipeline steps themselves are mocked separately per test
    (no real GPU/CPU work happens either way)."""

    def fake_submit(asset_id, source_id, video_path, stage1_config, stage2_config, *, cross_source_dedup=False):
        worker._process_upload(
            asset_id, source_id, video_path, stage1_config, stage2_config, cross_source_dedup=cross_source_dedup
        )

    monkeypatch.setattr("frontend.app.worker.submit_upload_job", fake_submit)


# --- asset lifecycle ---


def test_create_asset(client):
    resp = client.post("/assets", json={"name": "my clip"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "my clip"
    assert body["source_count"] == 0
    assert body["frame_count"] == 0
    assert len(body["asset_id"]) == 32


def test_list_assets(client):
    client.post("/assets", json={"name": "a"})
    client.post("/assets", json={"name": "b"})
    resp = client.get("/assets")
    assert resp.status_code == 200
    names = {a["name"] for a in resp.json()}
    assert names == {"a", "b"}


def test_get_asset_detail(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    resp = client.get(f"/assets/{asset_id}")
    assert resp.status_code == 200
    assert resp.json()["sources"] == []


def test_get_asset_404_for_unknown_id(client):
    resp = client.get("/assets/" + "0" * 32)
    assert resp.status_code == 404


def test_get_asset_400_for_malformed_id(client):
    resp = client.get("/assets/not-a-valid-id")
    assert resp.status_code == 400


# --- upload + status transitions ---


def test_upload_returns_queued_immediately(client, monkeypatch):
    # Real (non-test) behavior: submit_upload_job is fire-and-forget, so the
    # response reflects the state at submission time -- "queued" -- not
    # whatever the background job eventually reaches. Only patch the
    # pipeline internals here, NOT submit_upload_job itself, to prove this.
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]

    def fake_submit(asset_id, source_id, video_path, stage1_config, stage2_config, *, cross_source_dedup=False):
        pass  # never actually runs the job -- simulates "still queued"

    monkeypatch.setattr("frontend.app.worker.submit_upload_job", fake_submit)

    resp = client.post(f"/assets/{asset_id}/upload", files={"file": ("clip.mp4", b"fake", "video/mp4")})
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


def test_upload_processes_through_to_ready(client, monkeypatch):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]

    def fake_extract(video, out, config):
        out.mkdir(parents=True, exist_ok=True)
        models.save_candidates([], out / "candidates.json")

    def fake_dedup(in_dir, out, config=None):
        out.mkdir(parents=True, exist_ok=True)
        models.save_candidates([], out / "candidates.json")
        return []

    def fake_add_to_index_subprocess(index_dir, stage2_dir, source_id, *, cross_source_dedup=False):
        pass  # simulates the real `add-to-index` subprocess succeeding

    monkeypatch.setattr("frontend.worker.stage1_extract.extract", fake_extract)
    monkeypatch.setattr("frontend.worker.stage2_dedup.dedup", fake_dedup)
    monkeypatch.setattr("frontend.worker._run_add_to_index_subprocess", fake_add_to_index_subprocess)
    _make_upload_run_synchronously(monkeypatch)

    resp = client.post(f"/assets/{asset_id}/upload", files={"file": ("clip.mp4", b"fake", "video/mp4")})
    source_id = resp.json()["source_id"]

    status_resp = client.get(f"/assets/{asset_id}/sources/{source_id}/status")
    assert status_resp.json()["status"] == "ready"
    assert status_resp.json()["candidate_count"] == 0


def test_upload_records_error_status_on_failure(client, monkeypatch):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]

    def fake_extract_raises(video, out, config):
        raise ValueError("corrupt video")

    monkeypatch.setattr("frontend.worker.stage1_extract.extract", fake_extract_raises)
    _make_upload_run_synchronously(monkeypatch)

    resp = client.post(f"/assets/{asset_id}/upload", files={"file": ("clip.mp4", b"x", "video/mp4")})
    source_id = resp.json()["source_id"]

    status_resp = client.get(f"/assets/{asset_id}/sources/{source_id}/status")
    assert status_resp.json()["status"] == "error"
    assert "corrupt video" in status_resp.json()["error_message"]


def test_upload_to_missing_asset_404(client):
    resp = client.post(f"/assets/{'0' * 32}/upload", files={"file": ("clip.mp4", b"x", "video/mp4")})
    assert resp.status_code == 404


# --- orphaned-job recovery (server restart interrupting an in-flight source) ---


def test_mark_orphaned_sources_as_error_flags_non_terminal_statuses(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    s_queued = storage.add_source(asset_id, "a.mp4")
    s_extracting = storage.add_source(asset_id, "b.mp4")
    storage.update_source(asset_id, s_extracting.source_id, status="extracting")
    s_ready = storage.add_source(asset_id, "c.mp4")
    storage.update_source(asset_id, s_ready.source_id, status="ready")
    s_error = storage.add_source(asset_id, "d.mp4")
    storage.update_source(asset_id, s_error.source_id, status="error", error_message="pre-existing failure")

    recovered = storage.mark_orphaned_sources_as_error()

    recovered_ids = {source_id for _asset_id, source_id in recovered}
    assert recovered_ids == {s_queued.source_id, s_extracting.source_id}

    meta = storage.load_asset_meta(asset_id)
    by_id = {s.source_id: s for s in meta.sources}
    assert by_id[s_queued.source_id].status == "error"
    assert storage.ORPHANED_ERROR_MESSAGE in by_id[s_queued.source_id].error_message
    assert by_id[s_extracting.source_id].status == "error"
    # untouched -- already terminal
    assert by_id[s_ready.source_id].status == "ready"
    assert by_id[s_error.source_id].error_message == "pre-existing failure"


def test_mark_orphaned_sources_as_error_returns_empty_when_nothing_to_recover(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    storage.update_source(asset_id, source.source_id, status="ready")

    assert storage.mark_orphaned_sources_as_error() == []


def test_server_startup_recovers_orphaned_sources(tmp_path, monkeypatch):
    # Distinct from the direct-call test above: this confirms the FastAPI
    # lifespan hook itself actually invokes the recovery function, using a
    # real TestClient context-manager entry (which is what triggers
    # lifespan startup/shutdown events -- a bare TestClient(app) does not).
    monkeypatch.setattr(storage, "ASSETS_ROOT", tmp_path / "assets")
    with TestClient(app) as setup_client:
        asset_id = setup_client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    storage.update_source(asset_id, source.source_id, status="deduping")

    with TestClient(app) as c:
        status_resp = c.get(f"/assets/{asset_id}/sources/{source.source_id}/status")

    assert status_resp.json()["status"] == "error"
    assert storage.ORPHANED_ERROR_MESSAGE in status_resp.json()["error_message"]


# --- production-speed knobs on upload ---


def test_upload_with_no_knobs_builds_default_configs(client, monkeypatch):
    # Regression check: omitting every knob must produce bit-identical
    # default Stage1Config/Stage2Config to today's hardcoded behavior.
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]

    captured = {}

    def fake_submit(asset_id, source_id, video_path, stage1_config, stage2_config, *, cross_source_dedup=False):
        captured["stage1_config"] = stage1_config
        captured["stage2_config"] = stage2_config

    monkeypatch.setattr("frontend.app.worker.submit_upload_job", fake_submit)

    client.post(f"/assets/{asset_id}/upload", files={"file": ("clip.mp4", b"fake", "video/mp4")})

    from backend.stage1_extract import Stage1Config
    from backend.stage2_dedup import Stage2Config

    assert captured["stage1_config"] == Stage1Config()
    assert captured["stage2_config"] == Stage2Config()


def test_upload_with_knobs_builds_custom_configs(client, monkeypatch):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]

    captured = {}

    def fake_submit(asset_id, source_id, video_path, stage1_config, stage2_config, *, cross_source_dedup=False):
        captured["stage1_config"] = stage1_config
        captured["stage2_config"] = stage2_config

    monkeypatch.setattr("frontend.app.worker.submit_upload_job", fake_submit)

    client.post(
        f"/assets/{asset_id}/upload",
        files={"file": ("clip.mp4", b"fake", "video/mp4")},
        data={
            "downscale_factor": "0.5",
            "min_event_area_ratio": "0.008",
            "auto_mask": "true",
            "dedup_hamming_threshold": "12",
            "dedup_window_size": "10",
        },
    )

    stage1_config = captured["stage1_config"]
    stage2_config = captured["stage2_config"]
    assert stage1_config.downscale_factor == 0.5
    assert stage1_config.min_blob_area_ratio == 0.008
    assert stage1_config.run_auto_detect is True
    assert stage2_config.hamming_threshold == 12
    assert stage2_config.window_size == 10


def test_upload_records_settings_in_meta(client, monkeypatch):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    monkeypatch.setattr("frontend.app.worker.submit_upload_job", lambda *a, **k: None)

    resp = client.post(
        f"/assets/{asset_id}/upload",
        files={"file": ("clip.mp4", b"fake", "video/mp4")},
        data={"downscale_factor": "0.25", "dedup_hamming_threshold": "12"},
    )
    body = resp.json()
    assert body["settings"]["stage1_config"]["downscale_factor"] == 0.25
    assert body["settings"]["stage2_config"]["hamming_threshold"] == 12
    # Fields NOT overridden still show their real resolved defaults, not null.
    assert body["settings"]["stage2_config"]["window_size"] == 5

    status_resp = client.get(f"/assets/{asset_id}/sources/{body['source_id']}/status")
    assert status_resp.json()["settings"]["stage1_config"]["downscale_factor"] == 0.25


def test_upload_top_k_not_accepted_as_a_form_field(client, monkeypatch):
    # top_k has no effect on indexing -- confirm it's simply not part of
    # this endpoint's contract (an unknown form field is silently ignored
    # by FastAPI/Starlette's form parsing, not rejected -- this test exists
    # to document that intentional omission, not to assert an error).
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    monkeypatch.setattr("frontend.app.worker.submit_upload_job", lambda *a, **k: None)

    resp = client.post(
        f"/assets/{asset_id}/upload",
        files={"file": ("clip.mp4", b"fake", "video/mp4")},
        data={"top_k": "5"},
    )
    assert resp.status_code == 200
    assert "top_k" not in resp.json()["settings"]["stage1_config"]
    assert "top_k" not in resp.json()["settings"]["stage2_config"]


# --- frames pagination ---


def test_list_frames_includes_all_sources_but_only_ready_ones_contribute_frames(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]

    source_ready = storage.add_source(asset_id, "a.mp4")
    storage.update_source(asset_id, source_ready.source_id, status="ready")
    stage2_ready = storage.source_stage2_dir(asset_id, source_ready.source_id)
    stage2_ready.mkdir(parents=True)
    candidates = [
        Candidate(frame_index=i, timestamp_ms=i * 100.0, image_path=stage2_ready / f"frame_{i:06d}.jpg", reason="floor")
        for i in range(3)
    ]
    models.save_candidates(candidates, stage2_ready / "candidates.json")

    source_processing = storage.add_source(asset_id, "b.mp4")
    storage.update_source(asset_id, source_processing.source_id, status="extracting")
    # no stage2 output yet -- still mid-pipeline

    resp = client.get(f"/assets/{asset_id}/frames?page=1&page_size=50")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_frames"] == 3  # only source_ready contributes frames
    assert len(body["sources"]) == 2  # BOTH sources are listed, not just ready ones
    statuses = {s["source_id"]: s["status"] for s in body["sources"]}
    assert statuses[source_ready.source_id] == "ready"
    assert statuses[source_processing.source_id] == "extracting"  # surfaced, not hidden or omitted
    frame_counts = {s["source_id"]: s["frame_count"] for s in body["sources"]}
    assert frame_counts[source_ready.source_id] == 3
    assert frame_counts[source_processing.source_id] == 0


def test_list_frames_pagination_window(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    stage2_dir = storage.source_stage2_dir(asset_id, source.source_id)
    stage2_dir.mkdir(parents=True)
    candidates = [
        Candidate(frame_index=i, timestamp_ms=i * 100.0, image_path=stage2_dir / f"frame_{i:06d}.jpg", reason="floor")
        for i in range(10)
    ]
    models.save_candidates(candidates, stage2_dir / "candidates.json")

    resp = client.get(f"/assets/{asset_id}/frames?page=2&page_size=4")
    body = resp.json()
    assert body["total_frames"] == 10
    assert [f["frame_index"] for f in body["frames"]] == [4, 5, 6, 7]


def test_list_frames_404_for_missing_asset(client):
    resp = client.get(f"/assets/{'0' * 32}/frames")
    assert resp.status_code == 404


# --- search ---


def test_search_returns_ranked_results(client, monkeypatch):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    idx_dir = storage.index_dir(asset_id)
    idx_dir.mkdir(parents=True)
    np.save(idx_dir / "embeddings.npy", np.zeros((2, 4), dtype=np.float32))

    fake_results = [
        IndexedFrame(
            source_id="src1",
            frame_index=1,
            timestamp_ms=100.0,
            image_path=Path("data/assets/x/sources/src1/stage2/frame_000001.jpg"),
            reason="floor",
            similarity_score=0.9,
        ),
        IndexedFrame(
            source_id="src2",
            frame_index=2,
            timestamp_ms=200.0,
            image_path=Path("data/assets/x/sources/src2/stage2/frame_000002.jpg"),
            reason="motion",
            similarity_score=0.5,
        ),
    ]

    monkeypatch.setattr(
        "frontend.worker.multi_source_index.search_multi_source_index",
        lambda index_dir, query, top_k: fake_results,
    )

    resp = client.post(f"/assets/{asset_id}/search", json={"query": "a cat", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["source_id"] == "src1"
    assert body[0]["similarity_score"] == 0.9
    assert body[0]["image_url"] == f"/frames/{asset_id}/src1/frame_000001.jpg"


def test_search_without_index_yet_400(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    resp = client.post(f"/assets/{asset_id}/search", json={"query": "a cat"})
    assert resp.status_code == 400


def test_search_missing_asset_404(client):
    resp = client.post(f"/assets/{'0' * 32}/search", json={"query": "a cat"})
    assert resp.status_code == 404


# --- worker._stage1_already_complete: existence alone is not enough ---


def test_stage1_already_complete_false_when_missing(tmp_path):
    assert worker._stage1_already_complete(tmp_path / "stage1") is False


def test_stage1_already_complete_true_for_valid_manifest(tmp_path):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    models.save_candidates(
        [Candidate(frame_index=0, timestamp_ms=0.0, image_path=stage1_dir / "frame_000000.jpg", reason="floor")],
        stage1_dir / "candidates.json",
    )
    assert worker._stage1_already_complete(stage1_dir) is True


def test_stage1_already_complete_false_for_corrupt_manifest(tmp_path):
    # Mirrors the earlier real incident: a same-byte-count-but-garbled file
    # from an interrupted job -- existence alone must not be trusted.
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    (stage1_dir / "candidates.json").write_text("{not valid json at all")
    assert worker._stage1_already_complete(stage1_dir) is False


def test_stage1_already_complete_false_for_wrong_shape_json(tmp_path):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    (stage1_dir / "candidates.json").write_text('{"not": "a list of candidates"}')
    assert worker._stage1_already_complete(stage1_dir) is False


# --- retry ---


def test_retry_skips_extract_when_stage1_already_complete(client, monkeypatch):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]

    def fake_extract(video, out, config):
        raise AssertionError("extract() must not be called when stage1 output is already valid")

    def fake_dedup(in_dir, out, config=None):
        out.mkdir(parents=True, exist_ok=True)
        models.save_candidates([], out / "candidates.json")
        return []

    monkeypatch.setattr("frontend.worker.stage1_extract.extract", fake_extract)
    monkeypatch.setattr("frontend.worker.stage2_dedup.dedup", fake_dedup)
    monkeypatch.setattr("frontend.worker._run_add_to_index_subprocess", lambda *a, **k: None)
    _make_upload_run_synchronously(monkeypatch)

    # First upload fails (simulating an interruption AFTER stage1 finished --
    # e.g. mid-dedup), but with real, valid stage1 output already on disk.
    real_extract_call_count = {"n": 0}

    def fake_extract_first_time(video, out, config):
        real_extract_call_count["n"] += 1
        out.mkdir(parents=True, exist_ok=True)
        models.save_candidates(
            [Candidate(frame_index=0, timestamp_ms=0.0, image_path=out / "frame_000000.jpg", reason="floor")],
            out / "candidates.json",
        )

    monkeypatch.setattr("frontend.worker.stage1_extract.extract", fake_extract_first_time)
    monkeypatch.setattr(
        "frontend.worker.stage2_dedup.dedup",
        lambda in_dir, out, config=None: (_ for _ in ()).throw(ValueError("interrupted mid-dedup")),
    )
    resp = client.post(f"/assets/{asset_id}/upload", files={"file": ("clip.mp4", b"fake", "video/mp4")})
    source_id = resp.json()["source_id"]
    assert client.get(f"/assets/{asset_id}/sources/{source_id}/status").json()["status"] == "error"
    assert real_extract_call_count["n"] == 1

    # Now retry -- extract() must NOT run again (fake_extract above raises
    # AssertionError if it's called), dedup() must run and succeed this time
    # (restored to the succeeding fake_dedup -- the mid-dedup-failure lambda
    # above was only for simulating the FIRST, interrupted attempt).
    monkeypatch.setattr("frontend.worker.stage1_extract.extract", fake_extract)
    monkeypatch.setattr("frontend.worker.stage2_dedup.dedup", fake_dedup)
    retry_resp = client.post(f"/assets/{asset_id}/sources/{source_id}/retry")
    assert retry_resp.status_code == 200

    status_resp = client.get(f"/assets/{asset_id}/sources/{source_id}/status")
    assert status_resp.json()["status"] == "ready"


def test_retry_reruns_extract_when_stage1_missing(client, monkeypatch):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]

    monkeypatch.setattr(
        "frontend.worker.stage1_extract.extract",
        lambda video, out, config: (_ for _ in ()).throw(ValueError("still broken")),
    )
    _make_upload_run_synchronously(monkeypatch)

    resp = client.post(f"/assets/{asset_id}/upload", files={"file": ("clip.mp4", b"fake", "video/mp4")})
    source_id = resp.json()["source_id"]
    assert client.get(f"/assets/{asset_id}/sources/{source_id}/status").json()["status"] == "error"

    extract_calls = []

    def fake_extract(video, out, config):
        extract_calls.append(video)
        out.mkdir(parents=True, exist_ok=True)
        models.save_candidates([], out / "candidates.json")

    def fake_dedup(in_dir, out, config=None):
        out.mkdir(parents=True, exist_ok=True)
        models.save_candidates([], out / "candidates.json")
        return []

    monkeypatch.setattr("frontend.worker.stage1_extract.extract", fake_extract)
    monkeypatch.setattr("frontend.worker.stage2_dedup.dedup", fake_dedup)
    monkeypatch.setattr("frontend.worker._run_add_to_index_subprocess", lambda *a, **k: None)

    retry_resp = client.post(f"/assets/{asset_id}/sources/{source_id}/retry")
    assert retry_resp.status_code == 200
    assert len(extract_calls) == 1  # extract() genuinely re-ran, since stage1 never completed

    status_resp = client.get(f"/assets/{asset_id}/sources/{source_id}/status")
    assert status_resp.json()["status"] == "ready"


def test_retry_reuses_original_settings(client, monkeypatch):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]

    monkeypatch.setattr(
        "frontend.worker.stage1_extract.extract",
        lambda video, out, config: (_ for _ in ()).throw(ValueError("boom")),
    )
    _make_upload_run_synchronously(monkeypatch)

    resp = client.post(
        f"/assets/{asset_id}/upload",
        files={"file": ("clip.mp4", b"fake", "video/mp4")},
        data={"downscale_factor": "0.25", "dedup_hamming_threshold": "12"},
    )
    source_id = resp.json()["source_id"]

    captured_configs = {}

    def fake_extract(video, out, config):
        captured_configs["stage1"] = config
        out.mkdir(parents=True, exist_ok=True)
        models.save_candidates([], out / "candidates.json")

    def fake_dedup(in_dir, out, config=None):
        captured_configs["stage2"] = config
        out.mkdir(parents=True, exist_ok=True)
        models.save_candidates([], out / "candidates.json")
        return []

    monkeypatch.setattr("frontend.worker.stage1_extract.extract", fake_extract)
    monkeypatch.setattr("frontend.worker.stage2_dedup.dedup", fake_dedup)
    monkeypatch.setattr("frontend.worker._run_add_to_index_subprocess", lambda *a, **k: None)

    client.post(f"/assets/{asset_id}/sources/{source_id}/retry")

    assert captured_configs["stage1"].downscale_factor == 0.25
    assert captured_configs["stage2"].hamming_threshold == 12


def test_retry_rejects_non_error_source(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    storage.update_source(asset_id, source.source_id, status="ready")

    resp = client.post(f"/assets/{asset_id}/sources/{source.source_id}/retry")
    assert resp.status_code == 400


def test_retry_missing_video_file_500(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    storage.update_source(asset_id, source.source_id, status="error", error_message="x")
    # No video file ever written to disk for this source.

    resp = client.post(f"/assets/{asset_id}/sources/{source.source_id}/retry")
    assert resp.status_code == 500


def test_retry_missing_source_404(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    resp = client.post(f"/assets/{asset_id}/sources/{'0' * 32}/retry")
    assert resp.status_code == 404


# --- frame image serving ---


def test_get_frame_image_serves_file(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    stage2_dir = storage.source_stage2_dir(asset_id, source.source_id)
    stage2_dir.mkdir(parents=True)
    (stage2_dir / "frame_000005.jpg").write_bytes(b"fake jpeg bytes")

    resp = client.get(f"/frames/{asset_id}/{source.source_id}/frame_000005.jpg")
    assert resp.status_code == 200
    assert resp.content == b"fake jpeg bytes"


def test_get_frame_image_404_when_missing(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    stage2_dir = storage.source_stage2_dir(asset_id, source.source_id)
    stage2_dir.mkdir(parents=True)

    resp = client.get(f"/frames/{asset_id}/{source.source_id}/frame_000005.jpg")
    assert resp.status_code == 404


def test_get_frame_image_rejects_non_frame_filename(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    stage2_dir = storage.source_stage2_dir(asset_id, source.source_id)
    stage2_dir.mkdir(parents=True)
    (stage2_dir / "candidates.json").write_text("[]")  # a real, existing file in that dir

    resp = client.get(f"/frames/{asset_id}/{source.source_id}/candidates.json")
    assert resp.status_code == 404


def test_get_frame_image_rejects_path_traversal_filename():
    # Calling the handler directly (not via TestClient/HTTP) sidesteps URL-
    # encoding ambiguity around embedded "/" and tests the actual
    # security-relevant regex rejection directly.
    with pytest.raises(HTTPException) as exc_info:
        get_frame_image("0" * 32, "1" * 32, "../../../etc/passwd")
    assert exc_info.value.status_code in (400, 404)


def test_get_frame_image_rejects_malformed_asset_id():
    with pytest.raises(HTTPException) as exc_info:
        get_frame_image("not-a-valid-id", "1" * 32, "frame_000001.jpg")
    assert exc_info.value.status_code == 400


# --- download selected frames as a zip ---


def test_download_frames_returns_zip_with_requested_frames(client):
    import zipfile
    from io import BytesIO

    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    stage2_dir = storage.source_stage2_dir(asset_id, source.source_id)
    stage2_dir.mkdir(parents=True)
    (stage2_dir / "frame_000001.jpg").write_bytes(b"jpeg one")
    (stage2_dir / "frame_000002.jpg").write_bytes(b"jpeg two")

    resp = client.post(
        f"/assets/{asset_id}/frames/download",
        json={
            "frames": [
                {"source_id": source.source_id, "filename": "frame_000001.jpg"},
                {"source_id": source.source_id, "filename": "frame_000002.jpg"},
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"

    zf = zipfile.ZipFile(BytesIO(resp.content))
    names = sorted(zf.namelist())
    assert names == sorted(
        [f"{source.source_id}_frame_000001.jpg", f"{source.source_id}_frame_000002.jpg"]
    )
    assert zf.read(f"{source.source_id}_frame_000001.jpg") == b"jpeg one"
    assert zf.read(f"{source.source_id}_frame_000002.jpg") == b"jpeg two"


def test_download_frames_dedupes_repeated_selection(client):
    import zipfile
    from io import BytesIO

    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    stage2_dir = storage.source_stage2_dir(asset_id, source.source_id)
    stage2_dir.mkdir(parents=True)
    (stage2_dir / "frame_000001.jpg").write_bytes(b"jpeg one")

    ref = {"source_id": source.source_id, "filename": "frame_000001.jpg"}
    resp = client.post(f"/assets/{asset_id}/frames/download", json={"frames": [ref, ref]})
    assert resp.status_code == 200

    zf = zipfile.ZipFile(BytesIO(resp.content))
    assert zf.namelist() == [f"{source.source_id}_frame_000001.jpg"]


def test_download_frames_rejects_empty_selection(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    resp = client.post(f"/assets/{asset_id}/frames/download", json={"frames": []})
    assert resp.status_code == 400


def test_download_frames_rejects_too_many_frames(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    frames = [{"source_id": "0" * 32, "filename": "frame_000001.jpg"} for _ in range(501)]
    resp = client.post(f"/assets/{asset_id}/frames/download", json={"frames": frames})
    assert resp.status_code == 400


def test_download_frames_404_for_missing_frame_file(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    stage2_dir = storage.source_stage2_dir(asset_id, source.source_id)
    stage2_dir.mkdir(parents=True)

    resp = client.post(
        f"/assets/{asset_id}/frames/download",
        json={"frames": [{"source_id": source.source_id, "filename": "frame_000005.jpg"}]},
    )
    assert resp.status_code == 404


def test_download_frames_rejects_non_frame_filename(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    source = storage.add_source(asset_id, "a.mp4")
    stage2_dir = storage.source_stage2_dir(asset_id, source.source_id)
    stage2_dir.mkdir(parents=True)
    (stage2_dir / "candidates.json").write_text("[]")

    resp = client.post(
        f"/assets/{asset_id}/frames/download",
        json={"frames": [{"source_id": source.source_id, "filename": "candidates.json"}]},
    )
    assert resp.status_code == 400


def test_download_frames_rejects_malformed_source_id(client):
    asset_id = client.post("/assets", json={"name": "a"}).json()["asset_id"]
    resp = client.post(
        f"/assets/{asset_id}/frames/download",
        json={"frames": [{"source_id": "not-a-valid-id", "filename": "frame_000001.jpg"}]},
    )
    assert resp.status_code == 400


def test_download_frames_404_for_missing_asset(client):
    resp = client.post(
        f"/assets/{'0' * 32}/frames/download",
        json={"frames": [{"source_id": "1" * 32, "filename": "frame_000001.jpg"}]},
    )
    assert resp.status_code == 404
