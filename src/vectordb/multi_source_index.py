"""Appendable, multi-source combined SigLIP2 index -- for the webapp's
"assets" concept (one asset can hold several uploaded videos, searchable
as one combined index), NOT part of the numbered CLI pipeline stages.

Deliberately a separate module from stage3_rank.py: that file is scoped to
the single-video CLI stage 3 (its own docstring says so). This one reuses
its embedding primitives (_load_model/_embed_candidate_images/_embed_query)
rather than duplicating them, but owns its own record type (IndexedFrame,
which carries source_id -- Candidate has no such field, so stage3_rank's
own search_index()/build_index() cannot be reused unchanged here: their
metadata loader (models.load_candidates -> Candidate.from_dict) would raise
on an unexpected source_id key. The ranking MATH is identical either way
(pure numpy cosine similarity over pre-normalized embeddings) -- only the
metadata schema differs, which is why _rank_by_cosine_similarity below is
the one piece actually shared in spirit with stage3_rank's search_index().

Unlike single-video build_index() (in-place on ONE stage2 directory, since
that directory IS the index), a combined index spans MULTIPLE sources'
stage2 directories -- so it stores its own candidates.json/embeddings.npy
in index_dir, with each entry's image_path still pointing at its ORIGINAL
source's stage2 directory (no image copying/duplication at index time).
search_multi_source_index() correspondingly does not copy any images either
(there's no "next pipeline stage" to hand a fresh directory to here -- the
webapp serves images directly from each source's own stage2 directory).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import imagehash
import numpy as np

from backend import io_utils, models, stage2_dedup
from backend.models import IndexedFrame
from backend.stage3_rank import Stage3Config, _embed_candidate_images, _embed_query, _load_model


def _rank_by_cosine_similarity(embeds: np.ndarray, query_embed: np.ndarray) -> list[tuple[int, float]]:
    """Returns (row_index, score) pairs sorted descending. Pure numpy, no
    knowledge of what a row represents -- both sides must already be
    L2-normalized (true of every embedding this project produces), so the
    dot product IS the cosine similarity."""
    sims = (embeds @ query_embed.T).reshape(-1)
    return sorted(enumerate(sims.tolist()), key=lambda pair: pair[1], reverse=True)


def add_to_index(
    index_dir: Path,
    stage2_dir: Path,
    source_id: str,
    config: Stage3Config | None = None,
    *,
    cross_source_dedup: bool = False,
    cross_source_hamming_threshold: int = 8,
) -> None:
    """Embeds one source's deduped candidates (stage2_dir, unchanged stage 2
    output) and appends them to index_dir's combined index, creating it
    fresh if it doesn't exist yet.

    Idempotent per source_id: re-adding a source_id already present in the
    index REPLACES its rows rather than duplicating them -- needed because
    a retried/re-run upload job must not double-count a source's frames.

    cross_source_dedup (opt-in, default False): for two uploaded videos in
    the same asset that cover overlapping real-world footage -- before
    embedding this source's candidates, drop any whose pHash is within
    cross_source_hamming_threshold of an already-indexed frame from a
    DIFFERENT source_id, so the shared search index doesn't carry two
    visually-duplicate entries for the same real moment. The dropped
    frame's image/candidate row is untouched in its OWN source's stage2
    output/gallery -- only its entry in the SHARED index is skipped.

    Left False by default deliberately: this is the first place in this
    project that actively removes a frame instead of keeping it, which
    works against the recall-biased default used everywhere else -- a
    caller has to opt in per upload, not get it silently applied.

    Only ever compares against other sources' frames that were THEMSELVES
    indexed with cross_source_dedup=True (their IndexedFrame.phash is only
    populated in that case) -- a source added without this flag leaves no
    comparison history for a later source to dedup against, and pays zero
    added cost (no extra image loads/hashing) when left off here.
    """
    config = config or Stage3Config()
    index_dir.mkdir(parents=True, exist_ok=True)

    new_candidates = sorted(
        models.load_candidates(stage2_dir / "candidates.json"), key=lambda c: c.frame_index
    )

    candidates_path = index_dir / "candidates.json"
    embeds_path = index_dir / "embeddings.npy"
    if candidates_path.exists() and embeds_path.exists():
        existing_frames = models.load_indexed_frames(candidates_path)
        existing_embeds = np.load(embeds_path)
    else:
        existing_frames = []
        existing_embeds = None

    # Drop any prior rows for this source_id before appending its new ones --
    # replace, not accumulate. Everything left in existing_frames afterward
    # is, by construction, from a DIFFERENT source_id.
    keep_mask = [f.source_id != source_id for f in existing_frames]
    if not all(keep_mask):
        existing_frames = [f for f, keep in zip(existing_frames, keep_mask) if keep]
        if existing_embeds is not None:
            existing_embeds = existing_embeds[keep_mask]

    keep_flags = [True] * len(new_candidates)
    new_phashes: list[imagehash.ImageHash | None] = [None] * len(new_candidates)
    if cross_source_dedup:
        other_hashes = [
            imagehash.hex_to_hash(f.phash) for f in existing_frames if f.phash is not None
        ]
        for i, c in enumerate(new_candidates):
            phash = stage2_dedup.compute_phash(io_utils.load_frame_image(c.image_path))
            new_phashes[i] = phash
            if any(phash - other <= cross_source_hamming_threshold for other in other_hashes):
                keep_flags[i] = False

        dropped = keep_flags.count(False)
        if dropped:
            print(
                f"[add_to_index] source {source_id}: excluded {dropped} frame(s) as cross-source "
                "duplicates of frames already indexed from another source"
            )

    filtered_candidates = [c for c, keep in zip(new_candidates, keep_flags) if keep]
    filtered_phashes = [p for p, keep in zip(new_phashes, keep_flags) if keep]

    if not filtered_candidates:
        # Nothing left to embed for this source (stage2 produced no
        # candidates at all, or cross_source_dedup excluded every one) --
        # still persist the (possibly source_id-purged) existing state so
        # the index stays internally consistent.
        models.save_indexed_frames(existing_frames, candidates_path)
        if existing_embeds is not None:
            np.save(embeds_path, existing_embeds)
        elif not embeds_path.exists():
            np.save(embeds_path, np.zeros((0, 0), dtype=np.float32))
        return

    model, processor = _load_model(config.checkpoint_id)
    new_embeds = _embed_candidate_images(filtered_candidates, model, processor, config.batch_size)

    new_frames = [
        IndexedFrame(
            source_id=source_id,
            frame_index=c.frame_index,
            timestamp_ms=c.timestamp_ms,
            image_path=c.image_path,
            reason=c.reason,
            motion_score=c.motion_score,
            phash=str(p) if p is not None else None,
        )
        for c, p in zip(filtered_candidates, filtered_phashes)
    ]

    combined_frames = existing_frames + new_frames
    combined_embeds = (
        np.concatenate([existing_embeds, new_embeds], axis=0)
        if existing_embeds is not None and existing_embeds.shape[0] > 0
        else new_embeds
    )

    models.save_indexed_frames(combined_frames, candidates_path)
    np.save(embeds_path, combined_embeds)


def search_multi_source_index(
    index_dir: Path, query: str, top_k: int, config: Stage3Config | None = None
) -> list[IndexedFrame]:
    """Ranks a combined multi-source index against a query -- same
    cosine-similarity ranking math as stage3_rank.search_index(), operating
    on IndexedFrame records instead of plain Candidate. Never copies an
    image; returns ranked IndexedFrames with similarity_score populated and
    image_path still pointing at each frame's original source directory.
    """
    config = config or Stage3Config()

    embeds_path = index_dir / "embeddings.npy"
    if not embeds_path.exists():
        raise FileNotFoundError(
            f"{index_dir} has no embeddings.npy -- add at least one source via add_to_index() first"
        )

    frames = models.load_indexed_frames(index_dir / "candidates.json")
    embeds = np.load(embeds_path)

    if not frames:
        return []
    if len(frames) != embeds.shape[0]:
        raise ValueError(
            f"index corrupt or mismatched: {len(frames)} frames but {embeds.shape[0]} embedding rows in {index_dir}"
        )

    model, processor = _load_model(config.checkpoint_id)
    query_embed = _embed_query(query, model, processor).float().cpu().numpy()
    ranked_rows = _rank_by_cosine_similarity(embeds, query_embed)

    return [dataclasses.replace(frames[row_idx], similarity_score=score) for row_idx, score in ranked_rows[:top_k]]
