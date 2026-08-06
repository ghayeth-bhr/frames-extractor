"""Stage 5 -- human review via a standalone OpenCV window.

Each stage-4-verified frame is shown with the query and the VLM's verdict/
reasoning/confidence overlaid. Keybindings: k=keep, d=discard, space or
n=skip (without deciding), q=quit. Decisions persist to a JSON manifest
incrementally, so review is resumable across sessions: a frame with no
entry in the manifest is implicitly "not yet decided," whether it was
never reached or was explicitly skipped in a past session.

Only "keep" ever copies a file into out_dir. "discard" still gets a
ReviewDecision entry (so a resumed session doesn't show it again), but its
image_path stays the original stage-4 location -- out_dir's actual files
are then correct by construction (only kept frames), so a future export.py
can't accidentally include a rejected frame just by reading "everything in
out_dir" without checking `decision`.

All control flow (resumability skip, skip-without-deciding, keep/discard +
incremental write, quit-stops-early) lives in _run_review_loop, which
takes the keyboard/display as injected callables -- this is what makes it
testable without any cv2 GUI interaction. review() itself is a thin
wrapper around it plus the two real cv2 calls.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from . import io_utils, models
from .models import ReviewDecision, VerifiedFrame


@dataclass(kw_only=True)
class Stage5Config:
    window_name: str = "Stage 5 Review"


def _action_for_key(key: int) -> Literal["keep", "discard", "skip", "quit"] | None:
    if key == ord("k"):
        return "keep"
    if key == ord("d"):
        return "discard"
    if key in (ord(" "), ord("n")):
        return "skip"
    if key == ord("q"):
        return "quit"
    return None  # unrecognized key -- caller waits again


def _draw_overlay(image: np.ndarray, query: str, candidate: VerifiedFrame) -> np.ndarray:
    annotated = image.copy()
    confidence_text = f"{candidate.confidence:.2f}" if candidate.confidence is not None else "N/A"
    lines = [
        f"Query: {query}",
        f"Verdict: {candidate.verdict}  Confidence: {confidence_text}",
        f"Reasoning: {candidate.reasoning[:80]}",  # truncate, not wrap -- v1 simplicity
    ]
    y = 20
    for line in lines:
        cv2.putText(annotated, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        y += 18
    return annotated


def _run_review_loop(
    candidates: list[VerifiedFrame],
    decisions_by_index: dict[int, ReviewDecision],
    query: str,
    out_dir: Path,
    manifest_path: Path,
    get_action: Callable[[VerifiedFrame], Literal["keep", "discard", "skip", "quit"]],
    show_frame: Callable[[np.ndarray], None] = lambda image: None,
) -> None:
    for candidate in candidates:
        if candidate.frame_index in decisions_by_index:
            continue  # already decided in a prior session

        image = io_utils.load_frame_image(candidate.image_path)
        show_frame(_draw_overlay(image, query, candidate))
        action = get_action(candidate)

        if action == "quit":
            break
        if action == "skip":
            continue

        if action == "keep":
            image_path = out_dir / candidate.image_path.name
            if not image_path.exists():
                shutil.copy2(candidate.image_path, image_path)
        else:  # discard
            image_path = candidate.image_path

        decisions_by_index[candidate.frame_index] = ReviewDecision(
            frame_index=candidate.frame_index,
            timestamp_ms=candidate.timestamp_ms,
            image_path=image_path,
            reason=candidate.reason,
            motion_score=candidate.motion_score,
            similarity_score=candidate.similarity_score,
            verdict=candidate.verdict,
            reasoning=candidate.reasoning,
            confidence=candidate.confidence,
            decision=action,
        )
        # Incremental write after EVERY decision (SPEC.md's resumability
        # requirement) -- trivial cost at this scale, rebuild ordered list each time.
        ordered = [decisions_by_index[c.frame_index] for c in candidates if c.frame_index in decisions_by_index]
        models.save_candidates(ordered, manifest_path)


def review(in_dir: Path, out_dir: Path, query: str, config: Stage5Config | None = None) -> list[ReviewDecision]:
    config = config or Stage5Config()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(models.load_verified_frames(in_dir / "candidates.json"), key=lambda c: c.frame_index)
    manifest_path = out_dir / "candidates.json"
    decisions_by_index: dict[int, ReviewDecision] = {}
    if manifest_path.exists():
        for d in models.load_review_decisions(manifest_path):
            decisions_by_index[d.frame_index] = d

    def get_action_from_keyboard(candidate: VerifiedFrame) -> Literal["keep", "discard", "skip", "quit"]:
        while True:
            action = _action_for_key(cv2.waitKey(0) & 0xFF)
            if action is not None:
                return action

    def show_frame_cv2(image: np.ndarray) -> None:
        cv2.imshow(config.window_name, image)

    _run_review_loop(
        candidates, decisions_by_index, query, out_dir, manifest_path, get_action_from_keyboard, show_frame_cv2
    )
    cv2.destroyAllWindows()

    return [decisions_by_index[c.frame_index] for c in candidates if c.frame_index in decisions_by_index]
