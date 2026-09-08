"""Aggregates each stage's tunable config into a single object.

Each stage already defines its own StageNConfig dataclass, already
recall-biased by default. This module composes them for cli.py's benefit
-- it does not redefine them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .stage1_extract import Stage1Config
from .stage2_dedup import Stage2Config
from .stage3_rank import Stage3Config
from .stage4_verify import Stage4Config
from .stage5_review import Stage5Config


@dataclass(kw_only=True)
class PipelineConfig:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    stage4: Stage4Config = field(default_factory=Stage4Config)
    stage5: Stage5Config = field(default_factory=Stage5Config)
