"""Typed attribution records shared by execution, persistence, and analysis."""

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True, kw_only=True)
class AttributionRecord:
    """Canonical signal-attribution record.

    Empty optional values intentionally serialize as empty strings because CSV is one
    of the supported persistence boundaries. No aliases for previous schemas are
    accepted.
    """

    pair: str
    reference_time: str
    side: str
    price: Any = ""
    entry_open_next: Any = ""
    decision: str = ""
    reason: str = ""
    freqai_available: Any = ""
    freqai_reason: str = ""
    freqai_confidence: Any = ""
    freqai_dir_class: Any = ""
    freqai_long_probability: Any = ""
    freqai_flat_probability: Any = ""
    freqai_short_probability: Any = ""
    freqai_long_edge: Any = ""
    freqai_short_edge: Any = ""
    freqai_long_edge_lower: Any = ""
    freqai_short_edge_lower: Any = ""
    freqai_long_edge_upper: Any = ""
    freqai_short_edge_upper: Any = ""
    freqai_edge_gap: Any = ""
    freqai_level_event_class: Any = ""
    freqai_upside_room_pct: Any = ""
    freqai_downside_room_pct: Any = ""
    freqai_upside_room_atr: Any = ""
    freqai_downside_room_atr: Any = ""
    freqai_strong_upside_room_atr: Any = ""
    freqai_strong_downside_room_atr: Any = ""
    freqai_strong_resistance_strength: Any = ""
    freqai_strong_support_strength: Any = ""
    freqai_long_peak_profit: Any = ""
    freqai_short_peak_profit: Any = ""
    freqai_long_pre_profit_drawdown: Any = ""
    freqai_short_pre_profit_drawdown: Any = ""
    freqai_long_post_profit_drawdown: Any = ""
    freqai_short_post_profit_drawdown: Any = ""
    freqai_long_early_fail_risk: Any = ""
    freqai_short_early_fail_risk: Any = ""
    freqai_long_path_score: Any = ""
    freqai_short_path_score: Any = ""
    freqai_selected_path_score: Any = ""
    freqai_opposite_path_score: Any = ""
    freqai_path_score_gap: Any = ""
    freqai_selected_peak_profit: Any = ""
    freqai_selected_pre_profit_drawdown: Any = ""
    freqai_selected_post_profit_drawdown: Any = ""
    freqai_selected_reward_quality: Any = ""
    freqai_selected_risk_quality: Any = ""
    candidate_side: str = ""
    candidate_dir_class: Any = ""
    candidate_edge: Any = ""
    candidate_opposite_edge: Any = ""
    candidate_edge_gap: Any = ""
    candidate_path_score: Any = ""
    candidate_opposite_path_score: Any = ""
    candidate_path_score_gap: Any = ""
    candidate_peak_profit: Any = ""
    candidate_pre_profit_drawdown: Any = ""
    candidate_post_profit_drawdown: Any = ""
    candidate_early_fail_risk: Any = ""
    selected_early_fail_risk: Any = ""
    early_fail_stake_multiplier: Any = ""
    early_fail_leverage_multiplier: Any = ""
    candidate_reward_quality: Any = ""
    candidate_risk_quality: Any = ""
    dir_enter_threshold: Any = ""
    path_gap_threshold: Any = ""
    dir_class_pass: Any = ""
    edge_min_pass: Any = ""
    edge_gap_pass: Any = ""
    risk_adjusted_edge_pass: Any = ""
    strong_level_room_pass: Any = ""
    quality_multiplier: Any = ""
    leverage_reward_factor: Any = ""
    leverage_drawdown_factor: Any = ""
    llm_available: Any = ""
    llm_direction_bias: str = ""
    llm_confidence: Any = ""
    llm_event_risk: Any = ""
    llm_avoid_trade: Any = ""
    llm_alignment: str = ""
    stake_ratio: Any = ""
    leverage: Any = ""
    stop_loss_pct: Any = ""
    take_profit_pct: Any = ""
    max_hold_candles: Any = ""
    future_side_ret_1: Any = ""
    future_mfe_1: Any = ""
    future_mae_1: Any = ""
    future_side_ret_2: Any = ""
    future_mfe_2: Any = ""
    future_mae_2: Any = ""
    future_side_ret_3: Any = ""
    future_mfe_3: Any = ""
    future_mae_3: Any = ""
    future_side_ret_4: Any = ""
    future_mfe_4: Any = ""
    future_mae_4: Any = ""
    future_side_ret_5: Any = ""
    future_mfe_5: Any = ""
    future_mae_5: Any = ""
    future_side_ret_6: Any = ""
    future_mfe_6: Any = ""
    future_mae_6: Any = ""
    future_side_ret_7: Any = ""
    future_mfe_7: Any = ""
    future_mae_7: Any = ""
    future_side_ret_8: Any = ""
    future_mfe_8: Any = ""
    future_mae_8: Any = ""
    future_side_ret_9: Any = ""
    future_mfe_9: Any = ""
    future_mae_9: Any = ""
    future_side_ret_10: Any = ""
    future_mfe_10: Any = ""
    future_mae_10: Any = ""
    future_side_ret_11: Any = ""
    future_mfe_11: Any = ""
    future_mae_11: Any = ""
    future_side_ret_12: Any = ""
    future_mfe_12: Any = ""
    future_mae_12: Any = ""
    future_side_ret_13: Any = ""
    future_mfe_13: Any = ""
    future_mae_13: Any = ""
    future_side_ret_14: Any = ""
    future_mfe_14: Any = ""
    future_mae_14: Any = ""
    future_side_ret_15: Any = ""
    future_mfe_15: Any = ""
    future_mae_15: Any = ""
    future_side_ret_16: Any = ""
    future_mfe_16: Any = ""
    future_mae_16: Any = ""
    future_side_ret_terminal: Any = ""
    future_terminal_step: Any = ""
    detail_timeframe: str = ""
    detail_used: Any = ""
    future_detail_first_extreme_1: str = ""
    future_detail_first_extreme_3: str = ""
    future_detail_mfe_1: Any = ""
    future_detail_mae_1: Any = ""
    future_detail_mfe_3: Any = ""
    future_detail_mae_3: Any = ""
    executed_trade: bool = False
    trade_id: str = ""
    trade_profit_ratio: Any = ""
    trade_exit_reason: str = ""
    executed_match_count: int = 0
    analysis_weight: float = 1.0
    analysis_time: Any = None
    simulated_replay: Any = None

    _ANALYSIS_KEYS: ClassVar[dict[str, str]] = {
        "_executed_trade": "executed_trade",
        "_trade_id": "trade_id",
        "_trade_profit_ratio": "trade_profit_ratio",
        "_trade_exit_reason": "trade_exit_reason",
        "_executed_match_count": "executed_match_count",
        "_weight": "analysis_weight",
        "_time": "analysis_time",
        "_sim_replay": "simulated_replay",
    }

    def __post_init__(self) -> None:
        if not self.pair.strip():
            raise ValueError("Attribution pair must not be empty")
        if not self.reference_time.strip():
            raise ValueError("Attribution reference_time must not be empty")
        if self.side not in {"long", "short"}:
            raise ValueError("Attribution side must be 'long' or 'short'")
        if self.candidate_side and self.candidate_side not in {"long", "short"}:
            raise ValueError("Attribution candidate_side must be 'long' or 'short'")

    @classmethod
    def csv_fields(cls) -> tuple[str, ...]:
        """Return canonical fields written at the CSV boundary."""
        analysis_fields = set(cls._ANALYSIS_KEYS.values())
        return tuple(field.name for field in fields(cls) if field.name not in analysis_fields)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttributionRecord":
        """Deserialize a canonical JSON object without accepting field aliases."""
        allowed = {field.name for field in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown attribution fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_csv_row(cls, row: dict[str, Any]) -> "AttributionRecord":
        """Deserialize one row from the current canonical CSV schema."""
        expected = set(cls.csv_fields())
        if set(row) != expected:
            raise ValueError("Attribution CSV schema does not match the current schema")
        return cls(**row)

    def to_dict(self, *, include_analysis: bool = False) -> dict[str, Any]:
        """Serialize at a JSON/CSV boundary."""
        result = asdict(self)
        if not include_analysis:
            for name in self._ANALYSIS_KEYS.values():
                result.pop(name)
        return result

    def get(self, name: str, default: Any = None) -> Any:
        """Read a canonical typed field for analysis helpers."""
        attribute = self._ANALYSIS_KEYS.get(name, name)
        if attribute not in self.__dataclass_fields__:
            return default
        value = getattr(self, attribute)
        return default if value is None else value

    def with_analysis(self, **changes: Any) -> "AttributionRecord":
        """Return a copy enriched with non-serialized analysis state."""
        canonical = {self._ANALYSIS_KEYS.get(name, name): value for name, value in changes.items()}
        allowed = set(self._ANALYSIS_KEYS.values())
        if set(canonical) - allowed:
            raise ValueError("Only analysis state can be changed")
        return replace(self, **canonical)

    def with_outcomes(self, updates: dict[str, Any]) -> "AttributionRecord":
        """Return a copy with canonical realized-outcome fields replaced."""
        allowed = set(self.csv_fields()) - {"pair", "reference_time", "side"}
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"Unknown attribution outcome fields: {sorted(unknown)}")
        return replace(self, **updates)
