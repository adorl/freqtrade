"""Golden tests for the attribution CSV schema."""

import csv

from freqtrade.freqllm.attribution.writer import (
    ATTRIBUTION_FIELDS,
    MAX_ATTRIBUTION_HORIZON,
    SimpleAttributionWriter,
)


EXPECTED_BASE_FIELDS = [
    "pair",
    "reference_time",
    "side",
    "price",
    "entry_open_next",
    "decision",
    "reason",
    "freqai_available",
    "freqai_reason",
    "freqai_confidence",
    "freqai_dir_class",
    "freqai_long_probability",
    "freqai_flat_probability",
    "freqai_short_probability",
    "freqai_long_edge",
    "freqai_short_edge",
    "freqai_long_edge_lower",
    "freqai_short_edge_lower",
    "freqai_long_edge_upper",
    "freqai_short_edge_upper",
    "freqai_edge_gap",
    "freqai_level_event_class",
    "freqai_upside_room_pct",
    "freqai_downside_room_pct",
    "freqai_upside_room_atr",
    "freqai_downside_room_atr",
    "freqai_strong_upside_room_atr",
    "freqai_strong_downside_room_atr",
    "freqai_strong_resistance_strength",
    "freqai_strong_support_strength",
    "freqai_long_peak_profit",
    "freqai_short_peak_profit",
    "freqai_long_pre_profit_drawdown",
    "freqai_short_pre_profit_drawdown",
    "freqai_long_post_profit_drawdown",
    "freqai_short_post_profit_drawdown",
    "freqai_long_early_fail_risk",
    "freqai_short_early_fail_risk",
    "freqai_long_path_score",
    "freqai_short_path_score",
    "freqai_selected_path_score",
    "freqai_opposite_path_score",
    "freqai_path_score_gap",
    "freqai_selected_peak_profit",
    "freqai_selected_pre_profit_drawdown",
    "freqai_selected_post_profit_drawdown",
    "freqai_selected_reward_quality",
    "freqai_selected_risk_quality",
    "candidate_side",
    "candidate_dir_class",
    "candidate_edge",
    "candidate_opposite_edge",
    "candidate_edge_gap",
    "candidate_path_score",
    "candidate_opposite_path_score",
    "candidate_path_score_gap",
    "candidate_peak_profit",
    "candidate_pre_profit_drawdown",
    "candidate_post_profit_drawdown",
    "candidate_early_fail_risk",
    "selected_early_fail_risk",
    "early_fail_stake_multiplier",
    "early_fail_leverage_multiplier",
    "candidate_reward_quality",
    "candidate_risk_quality",
    "dir_enter_threshold",
    "path_gap_threshold",
    "dir_class_pass",
    "edge_min_pass",
    "edge_gap_pass",
    "risk_adjusted_edge_pass",
    "strong_level_room_pass",
    "quality_multiplier",
    "leverage_reward_factor",
    "leverage_drawdown_factor",
    "llm_available",
    "llm_direction_bias",
    "llm_confidence",
    "llm_event_risk",
    "llm_avoid_trade",
    "llm_alignment",
    "stake_ratio",
    "leverage",
    "stop_loss_pct",
    "take_profit_pct",
    "max_hold_candles",
]
EXPECTED_TAIL_FIELDS = [
    "future_side_ret_terminal",
    "future_terminal_step",
    "detail_timeframe",
    "detail_used",
    "future_detail_first_extreme_1",
    "future_detail_first_extreme_3",
    "future_detail_mfe_1",
    "future_detail_mae_1",
    "future_detail_mfe_3",
    "future_detail_mae_3",
]


def test_attribution_fields_golden_schema() -> None:
    """The public CSV contract keeps its stable field names and ordering."""
    expected_future = [
        field
        for step in range(1, 17)
        for field in (
            f"future_side_ret_{step}",
            f"future_mfe_{step}",
            f"future_mae_{step}",
        )
    ]
    expected = EXPECTED_BASE_FIELDS + expected_future + EXPECTED_TAIL_FIELDS

    assert MAX_ATTRIBUTION_HORIZON == 16
    assert ATTRIBUTION_FIELDS == expected
    assert len(ATTRIBUTION_FIELDS) == len(set(ATTRIBUTION_FIELDS))


def test_attribution_writer_uses_typed_schema(tmp_path) -> None:
    """Typed attribution records retain the golden schema at the CSV boundary."""
    writer = SimpleAttributionWriter(str(tmp_path))
    record_type = SimpleAttributionWriter.append.__annotations__["record"]
    record = record_type(
        pair="BTC/USDT:USDT",
        reference_time="2026-08-02T00:00:00+00:00",
        side="long",
        decision="emitted",
        future_side_ret_16=0.02,
    )
    path = writer.append(record)

    assert path == tmp_path / "entry_attribution_BTC_USDT_USDT.csv"
    assert path is not None
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    expected_values = {
        "pair": "BTC/USDT:USDT",
        "reference_time": "2026-08-02T00:00:00+00:00",
        "side": "long",
        "decision": "emitted",
        "future_side_ret_16": "0.02",
    }
    assert reader.fieldnames == ATTRIBUTION_FIELDS
    assert rows == [{field: expected_values.get(field, "") for field in ATTRIBUTION_FIELDS}]
