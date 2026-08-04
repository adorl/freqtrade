"""Tests for the simplified FreqAI and LLM strategy configuration."""

import json
import os
import re
from pathlib import Path
from unittest.mock import patch

from freqtrade.freqllm.configuration import LLMStrategyConfig


REMOVED_PATHS = {
    "context.reset_price_change_pct",
    "llm.analysis_timeout",
    "llm.log_full_prompt",
    "llm.min_confidence",
    "market_data.kline_analysis_limit",
    "market_data.kline_display_candles",
    "market_data.market_feature_history_days",
    "market_data.open_interest_display_limit",
    "market_data.reference_price_display_limit",
    "market_data.spot_trade_display_limit",
    "schedule.trigger_timeframe",
}


def test_configuration_resolves_environment() -> None:
    """Environment references are resolved without changing stored config syntax."""
    with patch.dict(os.environ, {"FREQLLM_TEST_API_KEY": "test-secret"}):
        config = LLMStrategyConfig.from_config(
            {
                "llm_strategy": {
                    "llm_bypass_enabled": "false",
                    "llm": {"api_key": "$FREQLLM_TEST_API_KEY"},
                }
            }
        )

    assert config.llm_bypass_enabled is False
    assert config.db_url == "sqlite:///user_data/freqllm/freqllm.db"
    assert config.llm.api_key == "test-secret"


def test_configuration_serialization_redacts_credentials() -> None:
    """Serialization redacts both API keys and database credentials."""
    environment = {
        "FREQLLM_TEST_API_KEY": "secret",
        "FREQLLM_TEST_DB_URL": "mysql://user:password@example.invalid/database",
    }
    with patch.dict(os.environ, environment):
        config = LLMStrategyConfig.from_config(
            {
                "llm_strategy": {
                    "db_url": "$FREQLLM_TEST_DB_URL",
                    "llm": {"api_key": "$FREQLLM_TEST_API_KEY"},
                }
            }
        )

    serialized = config.to_dict()

    assert serialized["db_url"] == "***"
    assert serialized["llm"]["api_key"] == "***"


def test_configuration_rejects_unknown_options() -> None:
    """Unknown top-level and grouped settings report their full configuration path."""
    cases = (
        (
            {"llm_strategy": {"llm": {"analysis_timeout": 60}}},
            r"llm_strategy\.llm\.analysis_timeout",
        ),
        ({"llm_strategy": {"unexpected": True}}, r"llm_strategy\.unexpected"),
    )
    for raw_config, expected_path in cases:
        try:
            LLMStrategyConfig.from_config(raw_config)
        except ValueError as exc:
            assert re.search(expected_path, str(exc))
            continue
        raise AssertionError("unknown configuration option was accepted")


def test_configuration_rejects_invalid_boolean() -> None:
    """Invalid boolean spellings are rejected with their configuration path."""
    try:
        LLMStrategyConfig.from_config({"llm_strategy": {"llm_bypass_enabled": "sometimes"}})
    except ValueError as exc:
        assert re.search(r"llm_strategy\.llm_bypass_enabled", str(exc))
    else:
        raise AssertionError("invalid boolean was accepted")


def test_configuration_rejects_literal_secrets_and_internal_endpoints() -> None:
    """Secrets must use env references and custom endpoints must not target internal hosts."""
    invalid_configs = (
        {"llm_strategy": {"llm": {"api_key": "literal-secret"}}},
        {
            "llm_strategy": {
                "llm": {
                    "provider": "openai",
                    "api_base_url": "http://10.0.0.1/v1",
                }
            }
        },
    )
    for invalid_config in invalid_configs:
        try:
            LLMStrategyConfig.from_config(invalid_config)
        except ValueError:
            continue
        raise AssertionError("unsafe advisor configuration was accepted")


def test_example_contains_no_removed_options() -> None:
    """The example excludes obsolete settings and retains secure defaults."""
    config_path = Path(__file__).parents[2] / "user_data" / "freqllm" / "config_example.json"
    document = json.loads(config_path.read_text(encoding="utf-8"))

    def paths(value: object, prefix: str = "") -> set[str]:
        if isinstance(value, dict):
            result: set[str] = set()
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else key
                result.add(path)
                result.update(paths(child, path))
            return result
        if isinstance(value, list):
            return set().union(*(paths(child, prefix) for child in value))
        return set()

    assert not REMOVED_PATHS.intersection(paths(document["llm_strategy"]))
    assert document["api_server"]["listen_ip_address"] == "127.0.0.1"
    assert document["force_entry_enable"] is False
    assert document["llm_strategy"]["external_features"]["disable_in_backtest"] is True
