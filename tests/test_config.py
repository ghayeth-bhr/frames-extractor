"""Tests for config.py's PipelineConfig aggregation."""

from __future__ import annotations

from backend.config import PipelineConfig
from backend.stage1_extract import Stage1Config
from backend.stage2_dedup import Stage2Config
from backend.stage3_rank import Stage3Config
from backend.stage4_verify import Stage4Config
from backend.stage5_review import Stage5Config


def test_pipeline_config_default_aggregation():
    config = PipelineConfig()
    assert isinstance(config.stage1, Stage1Config)
    assert isinstance(config.stage2, Stage2Config)
    assert isinstance(config.stage3, Stage3Config)
    assert isinstance(config.stage4, Stage4Config)
    assert isinstance(config.stage5, Stage5Config)


def test_pipeline_config_nested_override():
    config = PipelineConfig(stage3=Stage3Config(top_k=10))
    assert config.stage3.top_k == 10
    # Other stages remain untouched defaults.
    assert config.stage1 == Stage1Config()
    assert config.stage2 == Stage2Config()
