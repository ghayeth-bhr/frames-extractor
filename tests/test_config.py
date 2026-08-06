"""Tests for config.py's PipelineConfig aggregation."""

from __future__ import annotations

from frames_extractor.config import PipelineConfig
from frames_extractor.stage1_extract import Stage1Config
from frames_extractor.stage2_dedup import Stage2Config
from frames_extractor.stage3_rank import Stage3Config
from frames_extractor.stage4_verify import Stage4Config
from frames_extractor.stage5_review import Stage5Config


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
