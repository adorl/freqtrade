"""Characterization tests for deterministic entry and exit priority."""

from importlib import import_module
from typing import Any

from freqtrade.freqllm.decision.engine import (
    FreqAIPrediction,
    LLMView,
    SimpleDecisionEngine,
    SimpleExitConfig,
    SimpleStrategyConfig,
)


pytest = import_module("pytest")


def _prediction(**overrides: Any) -> FreqAIPrediction:
    values: dict[str, Any] = {
        "available": True,
        "reason": "ok",
        "confidence": 0.90,
        "dir_class": 0.90,
        "long_probability": 0.80,
        "flat_probability": 0.10,
        "short_probability": 0.10,
        "long_edge": 0.010,
        "short_edge": 0.001,
        "long_edge_lower": 0.008,
        "short_edge_lower": 0.0,
        "long_edge_upper": 0.012,
        "short_edge_upper": 0.002,
        "long_peak_profit": 0.010,
        "short_peak_profit": 0.002,
        "long_pre_profit_drawdown": -0.001,
        "short_pre_profit_drawdown": -0.001,
        "long_post_profit_drawdown": 0.001,
        "short_post_profit_drawdown": 0.001,
        "long_early_fail_risk": 0.10,
        "short_early_fail_risk": 0.10,
        "level_event_class": 0.0,
        "upside_room_pct": 0.10,
        "downside_room_pct": 0.10,
        "upside_room_atr": 2.0,
        "downside_room_atr": 2.0,
        "strong_upside_room_atr": 2.0,
        "strong_downside_room_atr": 2.0,
        "strong_resistance_strength": 1.0,
        "strong_support_strength": 1.0,
        "round_trip_cost": 0.001,
    }
    values.update(overrides)
    return FreqAIPrediction(**values)


def _llm(**overrides: Any) -> LLMView:
    values: dict[str, Any] = {
        "available": True,
        "direction_bias": "long",
        "confidence": 0.90,
        "event_risk": 0.10,
        "avoid_trade": False,
        "leverage_cap_multiplier": 1.0,
    }
    values.update(overrides)
    return LLMView(**values)


@pytest.mark.parametrize(
    ("prediction_overrides", "llm_overrides", "expected_reason"),
    [
        pytest.param(
            {"available": False, "reason": "not_ready", "confidence": 0.0, "long_edge": 0.2},
            {},
            "freqai_not_ready",
            id="availability-before-confidence-and-outlier",
        ),
        pytest.param(
            {"confidence": 0.0, "long_edge": 0.2},
            {},
            "freqai_confidence_low",
            id="confidence-before-outlier",
        ),
        pytest.param(
            {"long_edge": 0.2, "dir_class": 0.0},
            {},
            "freqai_prediction_outlier",
            id="outlier-before-entry-gates",
        ),
        pytest.param(
            {"dir_class": 0.0, "long_edge": -0.01},
            {},
            "no_directional_signal",
            id="direction-before-edge",
        ),
        pytest.param(
            {
                "long_edge": -0.001,
                "short_edge": -0.002,
                "upside_room_pct": 0.0,
            },
            {},
            "edge_not_positive",
            id="positive-edge-before-minimum",
        ),
        pytest.param(
            {"long_edge": 0.0005, "short_edge": -0.001, "upside_room_pct": 0.0},
            {},
            "edge_below_minimum",
            id="minimum-edge-before-gap",
        ),
        pytest.param(
            {"long_edge": 0.010, "short_edge": 0.0098, "upside_room_pct": 0.0},
            {},
            "edge_gap_below_minimum",
            id="edge-gap-before-risk-adjustment",
        ),
        pytest.param(
            {
                "long_edge_lower": -0.10,
                "long_edge_upper": -0.10,
                "upside_room_pct": 0.0,
            },
            {},
            "risk_adjusted_edge_not_positive",
            id="risk-adjusted-edge-before-level-room",
        ),
        pytest.param(
            {
                "upside_room_pct": 0.0,
                "upside_room_atr": 0.0,
                "strong_upside_room_atr": 0.0,
                "level_event_class": -1.0,
            },
            {},
            "key_level_room_insufficient",
            id="percentage-room-before-atr-room",
        ),
        pytest.param(
            {
                "upside_room_atr": 0.0,
                "strong_upside_room_atr": 0.0,
                "level_event_class": -1.0,
            },
            {},
            "key_level_atr_room_insufficient",
            id="atr-room-before-strong-level",
        ),
        pytest.param(
            {"strong_upside_room_atr": 0.0, "level_event_class": -1.0},
            {},
            "strong_key_level_room_insufficient",
            id="strong-level-before-level-event",
        ),
        pytest.param(
            {"level_event_class": -1.0, "long_early_fail_risk": 1.0},
            {"avoid_trade": True},
            "key_level_event_conflict",
            id="level-event-before-early-failure-and-llm",
        ),
        pytest.param(
            {"long_early_fail_risk": 1.0},
            {"avoid_trade": True},
            "early_fail_risk_high",
            id="early-failure-before-llm",
        ),
        pytest.param(
            {},
            {
                "avoid_trade": True,
                "event_risk": 1.0,
                "direction_bias": "short",
            },
            "llm_avoid_trade",
            id="llm-veto-before-event-and-conflict",
        ),
        pytest.param(
            {},
            {"event_risk": 1.0, "direction_bias": "short"},
            "llm_event_risk_block",
            id="llm-event-before-direction-conflict",
        ),
        pytest.param(
            {},
            {"direction_bias": "short"},
            "llm_direction_conflict",
            id="llm-direction-conflict",
        ),
    ],
)
def test_entry_gate_failure_priority(
    prediction_overrides: dict[str, Any],
    llm_overrides: dict[str, Any],
    expected_reason: str,
) -> None:
    """When several gates fail, the first configured gate supplies the stable reason."""
    engine = SimpleDecisionEngine(SimpleStrategyConfig())

    decision = engine.decide_entry(
        _prediction(**prediction_overrides),
        _llm(**llm_overrides),
    )

    assert decision.action == "reject"
    assert decision.reason == expected_reason


def test_exit_priority_golden_sequence() -> None:
    """Exit evaluation short-circuits from initial path rules through prediction expiry."""
    engine = SimpleDecisionEngine(SimpleStrategyConfig())
    early_failure = _prediction(
        dir_class=-0.90,
        long_early_fail_risk=1.0,
        long_edge=-0.010,
        long_edge_lower=-0.010,
        long_edge_upper=-0.010,
    )
    reverse = _prediction(dir_class=-0.90)
    collapsed = _prediction(dir_class=0.0)
    decayed = _prediction(
        dir_class=0.0,
        long_edge=-0.010,
        long_edge_lower=-0.010,
        long_edge_upper=-0.010,
    )

    cases = [
        (
            "stop-before-all-path-rules",
            -0.020,
            {
                "age_candles": 6,
                "peak_profit": 0.020,
                "entry_pre_profit_drawdown": -0.010,
                "freqai": early_failure,
            },
            "simple_stop_loss",
        ),
        (
            "recovery-before-no-progress-and-prediction",
            -0.005,
            {
                "age_candles": 6,
                "peak_profit": 0.0,
                "entry_pre_profit_drawdown": -0.001,
                "freqai": early_failure,
            },
            "simple_path_recovery_failed",
        ),
        (
            "no-progress-before-prediction",
            -0.001,
            {"age_candles": 6, "peak_profit": 0.0, "freqai": early_failure},
            "simple_no_progress_exit",
        ),
        (
            "predicted-retention-before-live-prediction",
            0.002,
            {
                "age_candles": 4,
                "peak_profit": 0.010,
                "current_profit": 0.002,
                "entry_peak_profit": 0.005,
                "freqai": early_failure,
            },
            "simple_predicted_peak_retention",
        ),
        (
            "early-failure-before-reversal",
            -0.001,
            {"age_candles": 4, "peak_profit": 0.002, "freqai": early_failure},
            "simple_early_fail_risk",
        ),
        (
            "reversal-before-collapse",
            0.0005,
            {
                "age_candles": 4,
                "peak_profit": 0.002,
                "entry_path_score": 0.10,
                "freqai": reverse,
            },
            "simple_reverse_signal",
        ),
        (
            "collapse-before-decay-and-expiry",
            0.0,
            {
                "age_candles": 4,
                "peak_profit": 0.002,
                "entry_path_score": 0.10,
                "freqai": collapsed,
            },
            "simple_path_score_collapse",
        ),
        (
            "decay-before-expiry",
            0.0005,
            {"age_candles": 4, "peak_profit": 0.0, "freqai": decayed},
            "simple_path_decay",
        ),
        (
            "expiry-fallback",
            0.0005,
            {"age_candles": 4, "round_trip_cost": 0.001},
            "simple_prediction_expired",
        ),
    ]

    actual = [
        (
            name,
            engine.decide_exit(side="long", price_profit=price_profit, **context),
        )
        for name, price_profit, context, _ in cases
    ]

    assert actual == [(name, expected) for name, _, _, expected in cases]


def test_take_profit_precedes_predicted_path_when_trailing_is_disabled() -> None:
    """Fixed take profit is an initial rule and therefore wins over later path exits."""
    config = SimpleStrategyConfig(exit=SimpleExitConfig(trailing_enabled=False))
    engine = SimpleDecisionEngine(config)

    reason = engine.decide_exit(
        side="long",
        price_profit=0.006,
        age_candles=4,
        peak_profit=0.010,
        current_profit=0.006,
        entry_peak_profit=0.005,
        freqai=_prediction(long_early_fail_risk=1.0, dir_class=-0.90),
    )

    assert reason == "simple_take_profit"
