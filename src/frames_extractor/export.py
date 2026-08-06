"""Export -- writes stage 5's kept frames to a Roboflow-importable layout:
images/ + _annotations.coco.json (images populated, annotations: []).

Stage 5's discard-never-copies design already guarantees a "keep" entry's
image_path resides under in_dir and a "discard" entry's does not (it
points at stage 4's directory instead) -- so filtering to decision ==
"keep" is sufficient on its own, no further existence-checking needed.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import io_utils, models


def export(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    decisions = models.load_review_decisions(in_dir / "candidates.json")
    kept = [d for d in decisions if d.decision == "keep"]

    now = datetime.now(timezone.utc).isoformat()
    image_entries = []
    for decision in kept:
        image = io_utils.load_frame_image(decision.image_path)
        height, width = image.shape[:2]
        dest_path = images_dir / decision.image_path.name
        shutil.copy2(decision.image_path, dest_path)
        image_entries.append(
            {
                "id": decision.frame_index,
                "license": 1,  # coupled to licenses[0]'s "id": 1 below -- trivial with
                # exactly one license entry, but nothing structurally enforces this pairing
                "file_name": decision.image_path.name,
                "height": height,
                "width": width,
                "date_captured": now,
            }
        )

    coco = {
        "info": {
            "year": str(datetime.now(timezone.utc).year),
            "version": "1",
            "description": "Exported from frames-extractor",
            "contributor": "",
            "url": "",
            "date_created": now,
        },
        "licenses": [{"id": 1, "url": "", "name": "Unknown"}],
        # UNVERIFIED ASSUMPTION -- Roboflow's own docs/forum never confirmed
        # whether an empty categories list is accepted by their importer. If a
        # real Roboflow upload of this file fails, check this first.
        "categories": [],
        "images": image_entries,
        "annotations": [],
    }
    (out_dir / "_annotations.coco.json").write_text(json.dumps(coco, indent=2))
