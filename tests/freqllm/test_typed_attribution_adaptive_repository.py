import csv

import pytest

from freqtrade.freqllm.adaptive import AdaptiveConfig, AdaptiveFeedbackManager
from freqtrade.freqllm.attribution import AttributionRecord, SimpleAttributionWriter
from freqtrade.freqllm.persistence import FreqLLMDatabase, FreqLLMModelBase


def test_attribution_writer_requires_typed_current_schema(tmp_path):
    record = AttributionRecord(
        pair="BTC/USDT:USDT",
        reference_time="2026-08-02T00:00:00+00:00",
        side="long",
        candidate_side="long",
        decision="emitted",
    )
    writer = SimpleAttributionWriter(str(tmp_path))

    path = writer.append(record)

    assert path is not None
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert AttributionRecord.from_csv_row(row).pair == record.pair
    with pytest.raises(TypeError):
        writer.append(record.to_dict())


def test_adaptive_uses_unified_database_metadata_and_repository():
    database = FreqLLMDatabase("sqlite://")
    config = AdaptiveConfig(enabled=True, mode="observe")
    manager = AdaptiveFeedbackManager(database, config)
    record = AttributionRecord(
        pair="ETH/USDT:USDT",
        reference_time="2026-08-02T00:00:00+00:00",
        side="short",
        candidate_side="short",
        decision="rejected",
    )
    try:
        manager.record_signal(record)

        assert manager.pending_signals() == [record]
        assert {
            "adaptive_signals",
            "adaptive_trades",
            "adaptive_state",
            "adaptive_updates",
        }.issubset(FreqLLMModelBase.metadata.tables)
    finally:
        database.close()


def test_adaptive_config_accepts_only_canonical_keys():
    with pytest.raises(ValueError, match=r"llm_strategy\.exit"):
        AdaptiveConfig.from_config({"llm_strategy": {"exit": {"trailing_enabled": False}}})
    with pytest.raises(ValueError, match=r"llm_strategy\.adaptive\.params"):
        AdaptiveConfig.from_config(
            {"llm_strategy": {"adaptive": {"params": {"dir_enter_threshold": 0.55}}}}
        )

    config = AdaptiveConfig.from_config(
        {
            "llm_strategy": {
                "simple_exit": {"trailing_enabled": True},
                "adaptive": {"defaults": {"dir_enter_threshold": 0.31}},
            }
        }
    )
    assert config.trailing_enabled is True
    assert config.defaults.dir_enter_threshold == 0.31
