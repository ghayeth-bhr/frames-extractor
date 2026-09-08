"""Stage 3 -- SigLIP2 semantic ranking against the CLI's text query.

Every deduped frame is embedded and ranked by cosine similarity against a
single query embedding; only the top-K survive. Recall-biased: K is a hard
count cap, not a similarity threshold, so a real event can't be silently
excluded by an absolute cutoff (see SPEC.md).
"""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from . import io_utils, models
from .models import Candidate

# Smoke-tested ceiling on this 6GB GPU: batch=128 reserved 5.7GB (93% of 6.1GB
# total) for zero throughput gain over batch=64, and batch=256 reserved 9.1GB
# (> total) and silently fell back to Windows WDDM shared/system memory
# instead of raising OOM -- a 15x throughput cliff (746ms/image vs ~50ms/image),
# not a clean error. The smoke test itself flagged 128 as "not worth the risk",
# so the cap matches Stage3Config's own default rather than permitting a
# config already ruled out.
MAX_SAFE_BATCH_SIZE = 64


@dataclass(kw_only=True)
class Stage3Config:
    checkpoint_id: str = "google/siglip2-so400m-patch14-384"  # no fallback needed on this GPU
    top_k: int = 50
    batch_size: int = 64  # smoke-tested sweet spot: ~64% VRAM, throughput already plateaus here

    def __post_init__(self) -> None:
        if self.batch_size > MAX_SAFE_BATCH_SIZE:
            raise ValueError(
                f"batch_size={self.batch_size} exceeds MAX_SAFE_BATCH_SIZE={MAX_SAFE_BATCH_SIZE} "
                "(see its comment -- batch=256 hit a silent WDDM fallback on this GPU)"
            )


def _load_model(checkpoint_id: str):
    """Requires CUDA -- SPEC.md assumes GPU available, no CPU fallback in v1."""
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 3 requires a CUDA GPU (SigLIP2 batched inference); none available.")
    processor = AutoProcessor.from_pretrained(checkpoint_id)
    model = AutoModel.from_pretrained(checkpoint_id, dtype=torch.float16).to("cuda")
    model.eval()
    return model, processor


def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    # Same channel-order hazard as stage 2's pHash bug: the model's
    # normalization assumes RGB, io_utils.load_frame_image returns BGR.
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _embed_query(query: str, model, processor) -> torch.Tensor:
    with torch.no_grad():
        # padding="max_length" is required -- empirically confirmed, not assumed:
        # the SAME query text embedded with vs. without it gives a cosine
        # similarity of only 0.638 between the two embeddings (0.362 apart),
        # dwarfing the entire relevant-vs-irrelevant signal range seen in the
        # smoke test (~0.15 vs ~0.04, a 0.11 spread). Default padding silently
        # produces a near-unrelated embedding, not a slightly-off one.
        inputs = processor(text=[query], padding="max_length", return_tensors="pt").to("cuda")
        embed = model.get_text_features(**inputs).pooler_output  # BaseModelOutputWithPooling
        return embed / embed.norm(dim=-1, keepdim=True)


def _embed_images(images: list[np.ndarray], model, processor, batch_size: int) -> torch.Tensor:
    with torch.no_grad():
        batches = []
        for start in range(0, len(images), batch_size):
            pil_images = [_to_pil_rgb(img) for img in images[start : start + batch_size]]
            inputs = processor(images=pil_images, return_tensors="pt").to("cuda")
            embed = model.get_image_features(**inputs).pooler_output
            batches.append(embed / embed.norm(dim=-1, keepdim=True))
        return torch.cat(batches, dim=0)


def _embed_candidate_images(
    candidates: list[Candidate], model, processor, batch_size: int
) -> np.ndarray:
    """Query-independent: embeds every candidate's image and returns an
    L2-normalized (N, D) float32 numpy array, row-aligned to `candidates` in
    the given order. Shared by rank() (embed-then-immediately-rank) and
    build_index() (embed-then-persist-for-later-ranking) -- exactly one
    image-embedding code path for both."""
    images = [io_utils.load_frame_image(c.image_path) for c in candidates]
    embeds = _embed_images(images, model, processor, batch_size)
    return embeds.float().cpu().numpy()


def _rank_candidates(
    candidates: list[Candidate], query: str, model, processor, config: Stage3Config
) -> list[tuple[Candidate, float]]:
    """Pure ranking logic, no I/O side effects -- returns ALL candidates paired
    with their score, sorted descending. Truncation to top_k happens in rank()."""
    query_embed = _embed_query(query, model, processor).float().cpu().numpy()
    image_embeds = _embed_candidate_images(candidates, model, processor, config.batch_size)
    sims = (image_embeds @ query_embed.T).reshape(-1)
    return sorted(zip(candidates, sims.tolist()), key=lambda pair: pair[1], reverse=True)


def rank(in_dir: Path, out_dir: Path, query: str, config: Stage3Config | None = None) -> list[Candidate]:
    config = config or Stage3Config()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        models.load_candidates(in_dir / "candidates.json"), key=lambda c: c.frame_index
    )
    if not candidates:
        models.save_candidates([], out_dir / "candidates.json")
        return []

    model, processor = _load_model(config.checkpoint_id)
    ranked = _rank_candidates(candidates, query, model, processor, config)

    kept: list[Candidate] = []
    for candidate, score in ranked[: config.top_k]:
        dest_path = out_dir / candidate.image_path.name
        shutil.copy2(candidate.image_path, dest_path)
        kept.append(dataclasses.replace(candidate, image_path=dest_path, similarity_score=score))

    models.save_candidates(kept, out_dir / "candidates.json")
    return kept


def build_index(dedup_dir: Path, config: Stage3Config | None = None) -> None:
    """Query-independent half of stage 3, for "build once, search many times":
    embeds every candidate already sitting in dedup_dir (stage 2's own output
    directory -- no separate copy, the images are already there) and writes
    embeddings.npy as a sidecar next to its existing candidates.json. No
    query, no top_k truncation (which candidates matter depends on a query
    that doesn't exist yet). A later search_index() call re-ranks this cached
    matrix against a real query without re-embedding a single image or
    re-running stage 1/2.
    """
    config = config or Stage3Config()

    candidates = sorted(
        models.load_candidates(dedup_dir / "candidates.json"), key=lambda c: c.frame_index
    )
    if not candidates:
        np.save(dedup_dir / "embeddings.npy", np.zeros((0, 0), dtype=np.float32))
        return

    model, processor = _load_model(config.checkpoint_id)
    embeds = _embed_candidate_images(candidates, model, processor, config.batch_size)

    # Overwrite with this exact sorted order -- embeddings.npy's row order
    # must match candidates.json's on-disk order exactly (search_index()
    # deliberately does NOT re-sort on load), so this is the one place that
    # order is nailed down, regardless of what order dedup() happened to
    # write in.
    models.save_candidates(candidates, dedup_dir / "candidates.json")
    np.save(dedup_dir / "embeddings.npy", embeds)


def search_index(
    index_dir: Path, out_dir: Path, query: str, config: Stage3Config | None = None
) -> list[Candidate]:
    """The cheap per-query lookup half of "build once, search many times":
    loads a build_index() output, embeds ONLY the query text (near-instant --
    the one unavoidable cost is loading the model itself, same as every other
    stage 3 call), and re-ranks the cached embedding matrix via a pure numpy
    dot product (both sides already L2-normalized, so dot product = cosine
    similarity) -- never touches an image, never re-runs stage 1/2.
    """
    config = config or Stage3Config()
    out_dir.mkdir(parents=True, exist_ok=True)

    embeds_path = index_dir / "embeddings.npy"
    if not embeds_path.exists():
        raise FileNotFoundError(
            f"{index_dir} has no embeddings.npy -- run `index`, not `dedup`/`extract` alone, "
            "to build a searchable index first"
        )

    # Deliberately NOT re-sorted -- row-alignment with embeddings.npy depends
    # on preserving build_index()'s on-disk order exactly (see its docstring).
    candidates = models.load_candidates(index_dir / "candidates.json")
    embeds = np.load(embeds_path)

    if not candidates:
        models.save_candidates([], out_dir / "candidates.json")
        return []
    if len(candidates) != embeds.shape[0]:
        raise ValueError(
            f"index corrupt or mismatched: {len(candidates)} candidates but "
            f"{embeds.shape[0]} embedding rows in {index_dir}"
        )

    model, processor = _load_model(config.checkpoint_id)
    query_embed = _embed_query(query, model, processor).float().cpu().numpy()
    sims = (embeds @ query_embed.T).reshape(-1)
    ranked = sorted(zip(candidates, sims.tolist()), key=lambda pair: pair[1], reverse=True)

    kept: list[Candidate] = []
    for candidate, score in ranked[: config.top_k]:
        dest_path = out_dir / candidate.image_path.name
        shutil.copy2(candidate.image_path, dest_path)
        kept.append(dataclasses.replace(candidate, image_path=dest_path, similarity_score=score))

    models.save_candidates(kept, out_dir / "candidates.json")
    return kept
