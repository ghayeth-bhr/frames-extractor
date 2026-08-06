"""Stage 4 -- VLM verification via a local Ollama server (qwen3-vl:4b).

Every stage-3-ranked frame gets a yes/no verdict + reasoning + confidence
against the CLI's query, using Ollama's structured-output `format` (JSON
schema) support rather than free-text parsing. Local failure modes
(timeout, malformed response, server unreachable mid-batch) never raise --
they're recorded as verdict="error" and still pass through to the output
manifest, consistent with the recall bias elsewhere in the pipeline: never
silently drop a frame.

Results are written to the manifest incrementally, one frame at a time, so
a re-run after an interruption only re-calls frames without a cached
yes/no verdict (SPEC.md's resumability requirement) -- cached "error"
entries are NOT treated as done, since that's exactly the case a re-run
is meant to retry.
"""

from __future__ import annotations

import base64
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import requests

from . import io_utils, models
from .models import VerifiedFrame

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["yes", "no"]},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["verdict", "reasoning", "confidence"],
}


@dataclass(kw_only=True)
class Stage4Config:
    model: str = "qwen3-vl:4b"
    base_url: str = "http://localhost:11434"
    num_ctx: int = 4096  # smoke-tested: default 2048 fails on real 1080p footage (2105 tokens)
    temperature: float = 0.0  # deterministic verdicts -- needed for fixed-expected-answer tests
    # AND for stage 4's own recall/precision consistency across repeated pipeline runs
    keep_alive: str = "10m"  # refreshed every request -- bridges inter-frame gaps, not a batch budget
    timeout_sec: float = 60.0


def _check_ollama_reachable(base_url: str) -> None:
    try:
        resp = requests.get(f"{base_url}/api/version", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama server not reachable at {base_url} -- is it running? ({e})") from e


def _build_prompt(query: str) -> str:
    return (
        f'Does this image show: "{query}"? '
        "Respond with ONLY a JSON object matching this schema: "
        '{"verdict": "yes"|"no", "reasoning": <short string>, "confidence": <float 0-1>}.'
    )


def _encode_image(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("failed to JPEG-encode image for Ollama request")
    return base64.b64encode(buf).decode("ascii")


def _parse_verdict_response(response_body: dict) -> tuple[Literal["yes", "no"], str, float]:
    """Raises ValueError for ANY shape mismatch -- missing/wrong-typed
    envelope keys, content not valid JSON, content not an object, verdict
    not yes/no, reasoning not a string, confidence missing/null/non-numeric/
    out of range. This is the ONE place shape validation happens, so the
    caller's except clause only needs (RequestException, ValueError) to be
    provably exhaustive -- no exception-type list to keep in sync by hand.
    """
    try:
        content = response_body["message"]["content"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"unexpected Ollama response envelope shape: {e}") from e

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"verdict content is not valid JSON: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError(f"verdict content is not a JSON object: {parsed!r}")

    verdict = parsed.get("verdict")
    if verdict not in ("yes", "no"):
        raise ValueError(f"verdict field missing or not yes/no: {verdict!r}")

    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str):
        raise ValueError(f"reasoning field missing or not a string: {reasoning!r}")

    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError(f"confidence field missing or not a number: {confidence!r}")
    confidence = float(confidence)
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence out of range [0,1]: {confidence}")

    return verdict, reasoning, confidence


def _verify_frame(
    image: np.ndarray, query: str, config: Stage4Config
) -> tuple[Literal["yes", "no", "error"], str, float | None]:
    """Never raises -- local failure modes (timeout, malformed response,
    connection failure) all map to ('error', <description>, None) so the
    caller can always pass a record through (never silently drop)."""
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": _build_prompt(query), "images": [_encode_image(image)]}],
        "format": VERDICT_SCHEMA,
        "stream": False,
        "keep_alive": config.keep_alive,
        "options": {"num_ctx": config.num_ctx, "temperature": config.temperature},
    }
    try:
        resp = requests.post(f"{config.base_url}/api/chat", json=payload, timeout=config.timeout_sec)
        resp.raise_for_status()
        return _parse_verdict_response(resp.json())
    except (requests.RequestException, ValueError) as e:
        return "error", f"{type(e).__name__}: {e}", None


def verify(in_dir: Path, out_dir: Path, query: str, config: Stage4Config | None = None) -> list[VerifiedFrame]:
    config = config or Stage4Config()
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_ollama_reachable(config.base_url)  # once, up front, before any frame processing

    candidates = sorted(models.load_candidates(in_dir / "candidates.json"), key=lambda c: c.frame_index)

    manifest_path = out_dir / "candidates.json"
    verified_by_index: dict[int, VerifiedFrame] = {}
    if manifest_path.exists():
        for vf in models.load_verified_frames(manifest_path):
            if vf.verdict != "error":  # only successful verdicts count as "already done"
                verified_by_index[vf.frame_index] = vf

    for candidate in candidates:
        if candidate.frame_index in verified_by_index:
            continue  # cached from a prior run -- don't re-spend a call

        dest_path = out_dir / candidate.image_path.name
        if not dest_path.exists():
            shutil.copy2(candidate.image_path, dest_path)

        image = io_utils.load_frame_image(candidate.image_path)
        verdict, reasoning, confidence = _verify_frame(image, query, config)

        verified_by_index[candidate.frame_index] = VerifiedFrame(
            frame_index=candidate.frame_index,
            timestamp_ms=candidate.timestamp_ms,
            image_path=dest_path,
            reason=candidate.reason,
            motion_score=candidate.motion_score,
            similarity_score=candidate.similarity_score,
            verdict=verdict,
            reasoning=reasoning,
            confidence=confidence,
        )
        # Incremental write after EVERY frame (SPEC.md's resumability requirement) --
        # trivial cost at this scale (<=50 frames), rebuild full ordered list each time.
        ordered = [verified_by_index[c.frame_index] for c in candidates if c.frame_index in verified_by_index]
        models.save_candidates(ordered, manifest_path)

    return [verified_by_index[c.frame_index] for c in candidates if c.frame_index in verified_by_index]
