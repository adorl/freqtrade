"""Persist adaptive attribution and propose bounded FreqLLM parameter overlays."""

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

from freqtrade.freqllm.attribution import AttributionRecord
from freqtrade.freqllm.persistence import AdaptiveRepository, FreqLLMDatabase


logger = logging.getLogger(__name__)
GLOBAL_SCOPE = "GLOBAL"
_ANALYSIS_MODULE = "freqtrade.freqllm.adaptive.analysis"


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _env(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    if value.startswith("$"):
        return os.environ.get(value[1:], "")
    return value


def _float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value in (None, ""):
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _parse_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text_value = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text_value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _timeframe_to_minutes(value: Any, default: int = 15) -> int:
    text_value = str(value or "").strip().lower()
    if not text_value:
        return default
    try:
        number = (
            int(float(text_value[:-1])) if not text_value[-1].isdigit() else int(float(text_value))
        )
    except (TypeError, ValueError):
        return default
    if text_value.endswith("m") or text_value[-1].isdigit():
        return max(1, number)
    if text_value.endswith("h"):
        return max(1, number * 60)
    if text_value.endswith("d"):
        return max(1, number * 1440)
    return default


@dataclass
class _EntryParameters:
    """Entry and direction threshold parameter group."""

    min_confidence_delta: float = 0.0
    dir_enter_threshold: float = 0.20
    dir_reverse_threshold: float = 0.25
    path_gap_threshold_multiplier: float = 1.0
    target_peak_tp_multiplier: float = 0.33
    pre_drawdown_weight: float = 1.0
    post_drawdown_weight: float = 0.5


@dataclass
class _QualityParameters(_EntryParameters):
    """Reward, risk, and edge quality parameter group."""

    pre_drawdown_free_ratio: float = 0.0
    reward_quality_floor: float = 0.50
    reward_quality_cap: float = 1.20
    risk_quality_floor: float = 0.25
    risk_quality_cap: float = 1.00
    edge_decay_profit_retention: float = 0.75
    trailing_activation_pct: float = 0.003


@dataclass
class _ExitParameters(_QualityParameters):
    """Exit timing and trailing parameter group."""

    trailing_distance_pct: float = 0.0015
    reverse_edge_gap: float = 0.001
    no_progress_age: int = 6
    no_progress_min_profit_multiplier: float = 1.0
    max_hold_candles: int = 16
    stake_quality_min_multiplier: float = 0.35
    stake_quality_max_multiplier: float = 1.20


@dataclass
class _FailureParameters(_ExitParameters):
    """Early-failure and leverage adjustment parameter group."""

    early_fail_block_threshold: float = 0.90
    early_fail_stake_weight: float = 0.70
    early_fail_stake_floor: float = 0.50
    early_fail_leverage_weight: float = 0.35
    early_fail_leverage_floor: float = 0.70
    early_fail_exit_threshold: float = 0.80
    leverage_reward_cap: float = 1.20


@dataclass
class _PathParameters(_FailureParameters):
    """Exit path-shape parameter group."""

    recovery_band_mult: float = 1.20
    recovery_sl_floor_ratio: float = 0.25
    peak_activation_ratio: float = 0.70
    post_drawdown_band_mult: float = 1.20
    path_collapse_ratio: float = 0.20
    expiry_decay_ratio: float = 0.50


@dataclass
class AdaptiveParameters(_PathParameters):
    """Bounded runtime parameters grouped by tuning responsibility."""

    llm_event_risk_block_mult: float = 1.0
    llm_conflict_confidence_mult: float = 1.0
    llm_neutral_size_mult: float = 1.0
    llm_leverage_cap_influence: float = 1.0
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0

    def bounded(self) -> "AdaptiveParameters":
        """Return a copy with every supported value clipped to its safe range."""
        values = asdict(self)
        for key, (lower, upper) in PARAM_BOUNDS.items():
            if key not in values:
                continue
            current = getattr(self, key)
            parsed = _float(values[key], current)
            if isinstance(current, int):
                values[key] = round(_clip(parsed, lower, upper))
            else:
                values[key] = _clip(parsed, lower, upper)
        return AdaptiveParameters(**values)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AdaptiveParameters":
        """Build parameters from a persisted dict, clipping values to bounds."""
        if not isinstance(data, dict):
            return cls()
        base = cls()
        values = asdict(base)
        for key in values:
            if key in data:
                if isinstance(values[key], int):
                    values[key] = _int(data[key], values[key])
                else:
                    values[key] = _float(data[key], values[key])
        return cls(**values).bounded()


PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "min_confidence_delta": (-0.10, 0.15),
    "dir_enter_threshold": (0.05, 0.60),
    "dir_reverse_threshold": (0.10, 0.60),
    "path_gap_threshold_multiplier": (0.60, 1.50),
    "target_peak_tp_multiplier": (0.20, 0.60),
    "pre_drawdown_weight": (0.25, 1.25),
    "post_drawdown_weight": (0.25, 1.00),
    "pre_drawdown_free_ratio": (0.00, 0.35),
    "reward_quality_floor": (0.35, 0.70),
    "reward_quality_cap": (1.00, 1.50),
    "risk_quality_floor": (0.15, 0.45),
    "risk_quality_cap": (0.70, 1.00),
    "edge_decay_profit_retention": (0.55, 0.90),
    "trailing_activation_pct": (0.0005, 0.02),
    "trailing_distance_pct": (0.0008, 0.008),
    "reverse_edge_gap": (0.0005, 0.003),
    "no_progress_age": (1, 12),
    "no_progress_min_profit_multiplier": (0.50, 2.00),
    "max_hold_candles": (1, 16),
    "stake_quality_min_multiplier": (0.20, 0.60),
    "stake_quality_max_multiplier": (1.00, 1.50),
    "early_fail_block_threshold": (0.05, 0.98),
    "early_fail_stake_weight": (0.00, 1.50),
    "early_fail_stake_floor": (0.20, 1.00),
    "early_fail_leverage_weight": (0.00, 1.00),
    "early_fail_leverage_floor": (0.40, 1.00),
    "early_fail_exit_threshold": (0.05, 0.98),
    "leverage_reward_cap": (1.00, 1.50),
    "recovery_band_mult": (1.00, 1.60),
    "recovery_sl_floor_ratio": (0.10, 0.50),
    "peak_activation_ratio": (0.40, 0.90),
    "post_drawdown_band_mult": (1.00, 1.60),
    "path_collapse_ratio": (0.05, 0.50),
    "expiry_decay_ratio": (0.20, 0.80),
    "llm_event_risk_block_mult": (0.70, 1.30),
    "llm_conflict_confidence_mult": (0.70, 1.30),
    "llm_neutral_size_mult": (0.70, 1.20),
    "llm_leverage_cap_influence": (0.50, 1.00),
    "stop_loss_multiplier": (1.0, 1.0),
    "take_profit_multiplier": (0.75, 1.50),
}
LOCKED_PARAMS = {
    "stop_loss_multiplier",
    "take_profit_multiplier",
    "trailing_activation_pct",
    "trailing_distance_pct",
    "edge_decay_profit_retention",
}
SLOW_PARAMS = {"no_progress_age", "max_hold_candles"}
LLM_PARAMS = {
    "llm_event_risk_block_mult",
    "llm_conflict_confidence_mult",
    "llm_neutral_size_mult",
    "llm_leverage_cap_influence",
}
RELAX_ON_INCREASE = {
    "no_progress_age",
    "max_hold_candles",
    "pre_drawdown_free_ratio",
    "reward_quality_floor",
    "reward_quality_cap",
    "risk_quality_cap",
    "stake_quality_min_multiplier",
    "stake_quality_max_multiplier",
    "early_fail_stake_floor",
    "early_fail_leverage_floor",
    "leverage_reward_cap",
    "trailing_distance_pct",
    "trailing_activation_pct",
    "take_profit_multiplier",
    "recovery_band_mult",
    "recovery_sl_floor_ratio",
    "post_drawdown_band_mult",
    "peak_activation_ratio",
    "llm_neutral_size_mult",
}
RELAX_ON_DECREASE = {
    "dir_enter_threshold",
    "dir_reverse_threshold",
    "path_gap_threshold_multiplier",
    "min_confidence_delta",
    "pre_drawdown_weight",
    "post_drawdown_weight",
    "risk_quality_floor",
    "reverse_edge_gap",
    "early_fail_block_threshold",
    "early_fail_stake_weight",
    "early_fail_leverage_weight",
    "early_fail_exit_threshold",
    "no_progress_min_profit_multiplier",
    "target_peak_tp_multiplier",
    "path_collapse_ratio",
    "expiry_decay_ratio",
    "llm_event_risk_block_mult",
    "llm_conflict_confidence_mult",
    "llm_leverage_cap_influence",
}


def _is_relaxation(param: str, old: float, new: float) -> bool:
    if new == old:
        return False
    increasing = new > old
    if param in RELAX_ON_INCREASE:
        return increasing
    if param in RELAX_ON_DECREASE:
        return not increasing
    return False


@dataclass
class _AdaptiveRuntimeConfig:
    """Runtime scheduling and sample-window configuration group."""

    enabled: bool = False
    mode: str = "suggest"  # observe | suggest | apply
    update_interval_signals: int = 100
    update_interval_trades: int = 20
    lookback_days: int = 14
    min_mature_signals: int = 100
    min_closed_trades: int = 10


@dataclass
class _AdaptiveStorageConfig(_AdaptiveRuntimeConfig):
    """Storage, reporting, and base risk configuration group."""

    opportunity_threshold: float = 0.004
    output_path: str = "user_data/freqllm/adaptive/adaptive_report.json"
    round_trip_cost: float = 0.0012
    stop_loss_pct: float = 0.010
    take_profit_pct: float = 0.005
    trailing_enabled: bool = True


@dataclass
class _AdaptiveSignalConfig(_AdaptiveStorageConfig):
    """Core signal threshold configuration group."""

    edge_decay_min_profit_pct: float = 0.001
    signal_min_confidence: float = 0.45
    signal_min_edge: float = 0.0010
    signal_min_edge_gap: float = 0.0005
    signal_min_level_room_atr: float = 0.35
    signal_level_room_buffer_pct: float = 0.0
    signal_level_event_conflict_threshold: float = 0.35


@dataclass
class _AdaptiveContextConfig(_AdaptiveSignalConfig):
    """Market-context and LLM filter configuration group."""

    signal_strong_level_room_atr: float = 0.35
    signal_min_level_strength: float = 0.0
    signal_edge_q50_weight: float = 0.70
    signal_edge_q80_weight: float = 0.20
    signal_edge_q20_weight: float = 0.10
    require_edge_positive: bool = True
    require_llm: bool = False


@dataclass
class _AdaptiveValidationConfig(_AdaptiveContextConfig):
    """LLM conflict and out-of-sample validation configuration group."""

    llm_event_risk_block: float = 0.80
    llm_conflict_confidence: float = 0.60
    llm_conflict_blocks: bool = True
    oos_fraction: float = 0.30
    min_oos_signals: int = 30
    min_samples_rule: int = 30
    relax_min_samples: int = 60


@dataclass
class _AdaptiveTuningConfig(_AdaptiveValidationConfig):
    """Statistical tuning and circuit-breaker configuration group."""

    z_tighten: float = 1.64
    z_relax: float = 2.33
    decay_half_life_days: float = 0.0
    breaker_consecutive_losses: int = 4
    breaker_drawdown: float = 0.06
    ema_alpha: float = 0.30
    slow_cooldown_cycles: int = 5


@dataclass
class AdaptiveConfig(_AdaptiveTuningConfig):
    """Validated adaptive feedback configuration assembled from strategy settings."""

    per_pair: bool = False
    min_pair_signals: int = 60
    timeframe_minutes: int = 15
    max_open_trades: int = 1
    execution_aware: bool = True
    min_executed_signals: int = 10
    defaults: AdaptiveParameters = field(default_factory=AdaptiveParameters)

    def half_life_days(self) -> float:
        """Return the configured decay half-life or a lookback-derived default."""
        if self.decay_half_life_days and self.decay_half_life_days > 0:
            return self.decay_half_life_days
        return max(self.lookback_days / 2.0, 0.5)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AdaptiveConfig":
        """Build the adaptive config from the strategy config dict."""
        root = config.get("llm_strategy", {}) if isinstance(config, dict) else {}
        root = root if isinstance(root, dict) else {}
        if "exit" in root:
            raise ValueError("Removed configuration field: llm_strategy.exit")
        raw = root.get("adaptive", {})
        raw = raw if isinstance(raw, dict) else {}
        if "params" in raw:
            raise ValueError("Removed configuration field: llm_strategy.adaptive.params")
        risk = root.get("risk", {}) if isinstance(root.get("risk", {}), dict) else {}
        exit_raw = root.get("simple_exit", {})
        exit_raw = exit_raw if isinstance(exit_raw, dict) else {}
        signal = root.get("signal", {}) if isinstance(root.get("signal", {}), dict) else {}
        feature_params = {}
        if isinstance(config, dict):
            feature_params = config.get("freqai", {}).get("feature_parameters", {}) or {}
        fee = _float(feature_params.get("label_fee_rate"), 0.0005) or 0.0005
        slippage = _float(feature_params.get("label_slippage_rate"), 0.0001) or 0.0001
        round_trip = max(0.0, (fee + slippage) * 2.0)
        defaults = AdaptiveParameters.from_dict(raw.get("defaults", {}))
        baseline = AdaptiveParameters()
        for param in LOCKED_PARAMS:
            setattr(defaults, param, getattr(baseline, param))
        mode = str(raw.get("mode", "suggest") or "suggest").lower()
        if mode not in {"observe", "suggest", "apply"}:
            mode = "suggest"
        return cls(
            enabled=_bool(raw.get("enabled"), False),
            mode=mode,
            update_interval_signals=max(10, _int(raw.get("update_interval_signals"), 100)),
            update_interval_trades=max(1, _int(raw.get("update_interval_trades"), 20)),
            lookback_days=max(1, _int(raw.get("lookback_days"), 14)),
            min_mature_signals=max(10, _int(raw.get("min_mature_signals"), 100)),
            min_closed_trades=max(0, _int(raw.get("min_closed_trades"), 10)),
            opportunity_threshold=max(0.0, _float(raw.get("opportunity_threshold"), 0.004) or 0.0),
            output_path=str(
                _env(raw.get("output_path", "user_data/freqllm/adaptive/adaptive_report.json"))
            ),
            round_trip_cost=round_trip,
            stop_loss_pct=max(_float(risk.get("stop_loss_pct"), 0.010) or 0.010, 1e-6),
            take_profit_pct=max(_float(risk.get("take_profit_pct"), 0.005) or 0.005, 1e-6),
            trailing_enabled=_bool(exit_raw.get("trailing_enabled"), True),
            edge_decay_min_profit_pct=max(
                _float(exit_raw.get("edge_decay_min_profit_pct"), 0.001) or 0.0, 0.0
            ),
            signal_min_confidence=_clip(
                _float(signal.get("min_confidence"), 0.45) or 0.45, 0.0, 1.0
            ),
            signal_min_edge=max(_float(signal.get("min_edge"), 0.0010) or 0.0010, 1e-9),
            signal_min_edge_gap=max(_float(signal.get("min_edge_gap"), 0.0005) or 0.0005, 1e-9),
            signal_min_level_room_atr=max(
                _float(signal.get("min_level_room_atr"), 0.35) or 0.0, 0.0
            ),
            signal_level_room_buffer_pct=max(
                _float(signal.get("level_room_buffer_pct"), 0.0) or 0.0, 0.0
            ),
            signal_level_event_conflict_threshold=_clip(
                _float(signal.get("level_event_conflict_threshold"), 0.35) or 0.35,
                0.0,
                3.0,
            ),
            signal_strong_level_room_atr=max(
                _float(signal.get("strong_level_room_atr"), 0.35) or 0.0, 0.0
            ),
            signal_min_level_strength=max(
                _float(signal.get("min_level_strength"), 0.0) or 0.0, 0.0
            ),
            signal_edge_q50_weight=_clip(
                _float(signal.get("edge_q50_weight"), 0.70) or 0.0, 0.0, 1.0
            ),
            signal_edge_q80_weight=_clip(
                _float(signal.get("edge_q80_weight"), 0.20) or 0.0, 0.0, 1.0
            ),
            signal_edge_q20_weight=_clip(
                _float(signal.get("edge_q20_weight"), 0.10) or 0.0, 0.0, 1.0
            ),
            require_edge_positive=_bool(signal.get("require_edge_positive"), True),
            require_llm=_bool(signal.get("require_llm"), False),
            llm_event_risk_block=_clip(
                _float(signal.get("llm_event_risk_block"), 0.80) or 0.80, 0.0, 1.0
            ),
            llm_conflict_confidence=_clip(
                _float(signal.get("llm_conflict_confidence"), 0.60) or 0.60,
                0.0,
                1.0,
            ),
            llm_conflict_blocks=_bool(signal.get("llm_conflict_blocks"), True),
            oos_fraction=_clip(_float(raw.get("oos_fraction"), 0.30) or 0.30, 0.10, 0.50),
            min_oos_signals=max(5, _int(raw.get("min_oos_signals"), 30)),
            min_samples_rule=max(5, _int(raw.get("min_samples_rule"), 30)),
            relax_min_samples=max(10, _int(raw.get("relax_min_samples"), 60)),
            z_tighten=max(0.0, _float(raw.get("z_tighten"), 1.64) or 1.64),
            z_relax=max(0.0, _float(raw.get("z_relax"), 2.33) or 2.33),
            decay_half_life_days=max(0.0, _float(raw.get("decay_half_life_days"), 0.0) or 0.0),
            breaker_consecutive_losses=max(0, _int(raw.get("breaker_consecutive_losses"), 4)),
            breaker_drawdown=max(0.0, _float(raw.get("breaker_drawdown"), 0.06) or 0.06),
            ema_alpha=_clip(_float(raw.get("ema_alpha"), 0.30) or 0.30, 0.05, 1.0),
            slow_cooldown_cycles=max(1, _int(raw.get("slow_cooldown_cycles"), 5)),
            per_pair=_bool(raw.get("per_pair"), False),
            min_pair_signals=max(20, _int(raw.get("min_pair_signals"), 60)),
            timeframe_minutes=_timeframe_to_minutes(
                config.get("timeframe") if isinstance(config, dict) else "", 15
            ),
            max_open_trades=max(
                1, _int(config.get("max_open_trades") if isinstance(config, dict) else 1, 1)
            ),
            execution_aware=_bool(raw.get("execution_aware"), True),
            min_executed_signals=max(1, _int(raw.get("min_executed_signals"), 10)),
            defaults=defaults,
        )


def _analysis_module() -> ModuleType:
    """Load analysis lazily after this module has initialized to avoid a cycle."""
    return import_module(_ANALYSIS_MODULE)


def _encode_json(value: Any, *, sort_keys: bool = False) -> str:
    """Encode canonical JSON without non-finite numbers or implicit string coercion."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=sort_keys)


def _decode_json_object(raw_value: Any, description: str) -> dict[str, Any] | None:
    """Decode a persisted JSON object and log malformed database content."""
    try:
        value = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Skipping malformed %s JSON", description, exc_info=True)
        return None
    if not isinstance(value, dict):
        logger.warning("Skipping non-object %s JSON", description)
        return None
    return value


def mark_executed_signals(signals, trades, cfg):
    """Match attribution signals to executed trades."""
    return _analysis_module().mark_executed_signals(signals, trades, cfg)


@dataclass
class _ManagerState:
    """Mutable counters, parameters, and cooldown state for a manager."""

    global_params: AdaptiveParameters
    pair_params: dict[str, AdaptiveParameters] = field(default_factory=dict)
    signals_since_update: int = 0
    trades_since_update: int = 0
    cycle: int = 0
    last_change_cycle: dict[tuple[str, str], int] = field(default_factory=dict)
    online_updates_enabled: bool = True


@dataclass
class _UpdateProposal:
    """A complete parameter update request for one adaptive scope."""

    scope: str
    old: AdaptiveParameters
    proposed: AdaptiveParameters
    metrics: dict[str, Any]
    reasons: list[str]


class AdaptiveFeedbackManager:
    """Persist feedback and apply validated adaptive parameter overlays."""

    def __init__(self, database: FreqLLMDatabase, config: AdaptiveConfig):
        self.config = config
        self._store: AdaptiveRepository | None = database.adaptive if config.enabled else None
        self._state = _ManagerState(config.defaults.bounded())
        if self._store is not None:
            self._load_state()

    @classmethod
    def from_strategy_config(
        cls, database: FreqLLMDatabase, config: dict[str, Any]
    ) -> "AdaptiveFeedbackManager":
        """Create a manager using a composition-root-owned database."""
        return cls(database, AdaptiveConfig.from_config(config))

    @property
    def enabled(self) -> bool:
        """Return whether feedback persistence is configured and available."""
        return self.config.enabled and self._store is not None

    @property
    def mode(self) -> str:
        """Return the configured observe, suggest, or apply mode."""
        return self.config.mode

    @property
    def online_updates_enabled(self) -> bool:
        """Return whether live update-only tuning is enabled."""
        return self._state.online_updates_enabled

    @online_updates_enabled.setter
    def online_updates_enabled(self, enabled: bool) -> None:
        self._state.online_updates_enabled = bool(enabled)

    def set_round_trip_cost(self, cost: float) -> None:
        """Replace the configured round-trip cost when the value is positive."""
        cost = _float(cost, self.config.round_trip_cost) or self.config.round_trip_cost
        if cost > 0:
            self.config.round_trip_cost = cost

    def current_parameters(self, pair: str | None = None) -> AdaptiveParameters:
        """Return the active bounded parameters for a pair or the global scope."""
        pair_params = self._state.pair_params
        if pair and self.config.per_pair and pair in pair_params:
            return pair_params[pair].bounded()
        return self._state.global_params.bounded()

    def current_parameters_dict(self, pair: str | None = None) -> dict[str, Any]:
        """Return the active parameters as a plain dict."""
        return asdict(self.current_parameters(pair))

    def _load_state(self) -> None:
        if self._store is None:
            return
        rows = self._store.query(
            "SELECT scope, params_json FROM adaptive_state "
            "WHERE id IN (SELECT MAX(id) FROM adaptive_state GROUP BY scope)"
        )
        for row in rows:
            values = _decode_json_object(row.get("params_json"), "adaptive state")
            if values is None:
                continue
            params = AdaptiveParameters.from_dict(values)
            scope = str(row.get("scope") or "")
            if scope == GLOBAL_SCOPE:
                self._state.global_params = params
            elif scope:
                self._state.pair_params[scope] = params

    def record_signal(self, record: AttributionRecord) -> None:
        """Insert or update a typed signal attribution record."""
        if not self.enabled:
            return
        pair = record.pair
        reference_time = record.reference_time
        side = record.candidate_side or record.side
        matured = 1 if record.future_side_ret_terminal not in (None, "") else 0
        now = _utcnow()
        params = {
            "pair": pair,
            "reference_time": reference_time,
            "side": side,
            "decision": record.decision,
            "reason": record.reason[:96],
            "payload_json": _encode_json(record.to_dict()),
            "matured": matured,
            "now": now,
        }
        existing = self._store.query(
            "SELECT id FROM adaptive_signals WHERE pair=:pair "
            "AND reference_time=:reference_time AND side=:side",
            params,
        )
        if existing:
            params["id"] = existing[0]["id"]
            self._store.execute(
                "UPDATE adaptive_signals SET decision=:decision, reason=:reason, "
                "payload_json=:payload_json, matured=:matured, updated_at=:now WHERE id=:id",
                params,
            )
        else:
            self._store.execute(
                "INSERT INTO adaptive_signals(pair, reference_time, side, decision, reason, "
                "payload_json, matured, created_at, updated_at) VALUES(:pair, :reference_time, "
                ":side, :decision, :reason, :payload_json, :matured, :now, :now)",
                params,
            )
        self._state.signals_since_update += 1
        if self.online_updates_enabled:
            self.maybe_update()

    def record_trade(self, payload: dict[str, Any]) -> None:
        """Record a closed trade payload for adaptive analysis."""
        if not self.enabled:
            return
        pair = str(payload.get("pair") or "")
        if not pair:
            return
        now = _utcnow()
        params = {
            "trade_id": str(payload.get("trade_id") or ""),
            "pair": pair,
            "side": str(payload.get("side") or ""),
            "open_time": str(payload.get("open_time") or ""),
            "close_time": str(payload.get("close_time") or ""),
            "exit_reason": str(payload.get("exit_reason") or "")[:64],
            "profit_ratio": _float(payload.get("profit_ratio"), 0.0),
            "profit_abs": _float(payload.get("profit_abs"), 0.0),
            "payload_json": _encode_json(payload),
            "now": now,
        }
        existing = self._store.query(
            "SELECT id FROM adaptive_trades WHERE trade_id=:trade_id AND close_time=:close_time "
            "AND exit_reason=:exit_reason",
            params,
        )
        if not existing:
            self._store.execute(
                "INSERT INTO adaptive_trades(trade_id, pair, side, open_time, "
                "close_time, exit_reason, profit_ratio, profit_abs, payload_json, "
                "created_at) VALUES(:trade_id, :pair, :side, :open_time, "
                ":close_time, :exit_reason, :profit_ratio, :profit_abs, "
                ":payload_json, :now)",
                params,
            )
        self._state.trades_since_update += 1
        if self.online_updates_enabled:
            self.maybe_update()

    def pending_signals(self, limit: int = 500) -> list[AttributionRecord]:
        """Return typed unmatured signals awaiting outcome updates."""
        if self._store is None:
            return []
        rows = self._store.query(
            "SELECT payload_json FROM adaptive_signals "
            "WHERE matured=0 ORDER BY reference_time ASC LIMIT :limit",
            {"limit": limit},
        )
        result: list[AttributionRecord] = []
        for row in rows:
            payload = _decode_json_object(row.get("payload_json"), "pending signal")
            if payload is not None:
                result.append(AttributionRecord.from_dict(payload))
        return result

    def update_signal_outcome(
        self, pair: str, reference_time: str, side: str, updates: dict[str, Any]
    ) -> None:
        """Attach realised outcomes to a stored signal payload."""
        if self._store is None or not updates:
            return
        rows = self._store.query(
            "SELECT id, payload_json FROM adaptive_signals WHERE pair=:pair AND "
            "reference_time=:reference_time AND side=:side",
            {"pair": pair, "reference_time": reference_time, "side": side},
        )
        if not rows:
            return
        payload = _decode_json_object(rows[0].get("payload_json"), "signal outcome")
        if payload is None:
            return
        record = AttributionRecord.from_dict(payload).with_outcomes(updates)
        self._store.execute(
            "UPDATE adaptive_signals SET payload_json=:payload_json, matured=1, "
            "updated_at=:now WHERE id=:id",
            {
                "payload_json": _encode_json(record.to_dict()),
                "now": _utcnow(),
                "id": rows[0]["id"],
            },
        )
        self._state.signals_since_update += 1
        if self.online_updates_enabled:
            self.maybe_update()

    def _cutoff_iso(self) -> str:
        cutoff = datetime.now(UTC).timestamp() - self.config.lookback_days * 86400
        return datetime.fromtimestamp(cutoff, tz=UTC).isoformat()

    def _load_signals(self, pair: str | None, limit: int = 8000) -> list[AttributionRecord]:
        if self._store is None:
            return []
        sql = (
            "SELECT payload_json FROM adaptive_signals "
            "WHERE matured=1 AND reference_time >= :cutoff"
        )
        params: dict[str, Any] = {"cutoff": self._cutoff_iso(), "limit": limit}
        if pair:
            sql += " AND pair = :pair"
            params["pair"] = pair
        sql += " ORDER BY reference_time DESC LIMIT :limit"
        result: list[AttributionRecord] = []
        for row in self._store.query(sql, params):
            payload = _decode_json_object(row.get("payload_json"), "mature signal")
            if payload is not None:
                result.append(AttributionRecord.from_dict(payload))
        return list(reversed(result))

    def _load_trades(self, pair: str | None, limit: int = 2000) -> list[dict[str, Any]]:
        if self._store is None:
            return []
        sql = "SELECT payload_json FROM adaptive_trades WHERE close_time >= :cutoff"
        params: dict[str, Any] = {"cutoff": self._cutoff_iso(), "limit": limit}
        if pair:
            sql += " AND pair = :pair"
            params["pair"] = pair
        sql += " ORDER BY close_time DESC LIMIT :limit"
        result = []
        for row in self._store.query(sql, params):
            payload = _decode_json_object(row.get("payload_json"), "closed trade")
            if payload is not None:
                result.append(payload)
        return list(reversed(result))

    def _active_pairs(self) -> list[str]:
        if self._store is None:
            return []
        rows = self._store.query(
            "SELECT pair, COUNT(*) AS n FROM adaptive_signals "
            "WHERE matured=1 AND reference_time >= :cutoff GROUP BY pair "
            "HAVING COUNT(*) >= :min_n",
            {"cutoff": self._cutoff_iso(), "min_n": self.config.min_pair_signals},
        )
        return [str(row["pair"]) for row in rows if row.get("pair")]

    def maybe_update(self) -> None:
        """Run an update after either configured attribution threshold is met."""
        if not self.enabled or self.config.mode == "observe":
            return
        state = self._state
        if (
            state.signals_since_update < self.config.update_interval_signals
            and state.trades_since_update < self.config.update_interval_trades
        ):
            return
        state.signals_since_update = 0
        state.trades_since_update = 0
        self.run_update()

    def run_update(self) -> None:
        """Analyze eligible global and pair scopes and persist the latest proposal."""
        if not self.enabled:
            return
        self._state.cycle += 1
        scopes: list[tuple[str | None, str]] = [(None, GLOBAL_SCOPE)]
        if self.config.per_pair:
            scopes.extend((pair, pair) for pair in self._active_pairs())
        last_report = None
        for pair, scope in scopes:
            signals = self._load_signals(pair)
            trades = self._load_trades(pair)
            signals = mark_executed_signals(signals, trades, self.config)
            if (
                len(signals) < self.config.min_mature_signals
                and len(trades) < self.config.min_closed_trades
            ):
                continue
            base = self.current_parameters(pair)
            proposed, metrics, reasons = analyze(
                signals, trades, base, self.config, online=self.online_updates_enabled
            )
            metrics["scope"] = scope
            last_report = (proposed, metrics, reasons)
            if self.config.mode == "apply":
                self._apply(_UpdateProposal(scope, base, proposed, metrics, reasons))
        if last_report is not None:
            self.save_report(*last_report)

    def _next_parameter_value(
        self, update: _UpdateProposal, param: str, target: float, base_value: float
    ) -> tuple[float | int | None, str | None]:
        if param in LOCKED_PARAMS or target == base_value:
            return None, None
        if param in LLM_PARAMS and not self.online_updates_enabled:
            return None, None
        breaker_active = bool(update.metrics.get("circuit_breaker_active"))
        if breaker_active and _is_relaxation(param, base_value, target):
            return None, f"breaker_block_relax_{param}"
        if param not in SLOW_PARAMS:
            alpha = self.config.ema_alpha
            return alpha * target + (1.0 - alpha) * base_value, None
        key = (update.scope, param)
        last_cycle = self._state.last_change_cycle.get(key, -(10**9))
        if self._state.cycle - last_cycle < self.config.slow_cooldown_cycles:
            return None, None
        self._state.last_change_cycle[key] = self._state.cycle
        step = 1 if target > base_value else -1
        return int(base_value) + step, None

    def _build_applied_parameters(
        self, update: _UpdateProposal
    ) -> tuple[AdaptiveParameters, list[str]]:
        old_values = asdict(update.old)
        new_values = dict(old_values)
        applied_reasons: list[str] = []
        for param, target in asdict(update.proposed.bounded()).items():
            value, reason = self._next_parameter_value(
                update, param, float(target), float(old_values[param])
            )
            if reason:
                applied_reasons.append(reason)
            if value is not None:
                new_values[param] = value
        return AdaptiveParameters(**new_values).bounded(), applied_reasons

    def _save_active_state(
        self,
        update: _UpdateProposal,
        new_params: AdaptiveParameters,
        applied_reasons: list[str],
        now: str,
    ) -> None:
        self._store.execute(
            "INSERT INTO adaptive_state(scope, active_from, mode, params_json, "
            "metrics_json, reasons_json, created_at) VALUES(:scope, :now, :mode, "
            ":params, :metrics, :reasons, :now)",
            {
                "scope": update.scope,
                "now": now,
                "mode": self.config.mode,
                "params": _encode_json(asdict(new_params), sort_keys=True),
                "metrics": _encode_json(update.metrics, sort_keys=True),
                "reasons": _encode_json(update.reasons + applied_reasons),
            },
        )

    def _record_parameter_changes(
        self,
        update: _UpdateProposal,
        new_params: AdaptiveParameters,
        applied_reasons: list[str],
        now: str,
    ) -> None:
        old_values = asdict(update.old)
        reason = ";".join(update.reasons + applied_reasons)[:255]
        metrics = _encode_json(update.metrics, sort_keys=True)
        for param, new_value in asdict(new_params).items():
            old_value = old_values.get(param)
            if old_value == new_value:
                continue
            self._store.execute(
                "INSERT INTO adaptive_updates(scope, param_name, old_value, "
                "new_value, reason, metrics_json, created_at) VALUES(:scope, "
                ":param, :old, :new, :reason, :metrics, :now)",
                {
                    "scope": update.scope,
                    "param": param,
                    "old": _float(old_value, 0.0),
                    "new": _float(new_value, 0.0),
                    "reason": reason,
                    "metrics": metrics,
                    "now": now,
                },
            )

    def _apply(self, update: _UpdateProposal) -> None:
        if self._store is None:
            return
        new_params, applied_reasons = self._build_applied_parameters(update)
        if asdict(new_params) == asdict(update.old.bounded()):
            return
        now = _utcnow()
        self._save_active_state(update, new_params, applied_reasons, now)
        self._record_parameter_changes(update, new_params, applied_reasons, now)
        if update.scope == GLOBAL_SCOPE:
            self._state.global_params = new_params
        else:
            self._state.pair_params[update.scope] = new_params

    def apply_parameters(
        self, params: AdaptiveParameters, metrics: dict[str, Any], reasons: list[str]
    ) -> None:
        """Apply a proposed parameter overlay to the global scope."""
        if self._store is None:
            self._state.global_params = params.bounded()
            return
        self._state.cycle += 1
        self._apply(
            _UpdateProposal(GLOBAL_SCOPE, self.current_parameters(), params, metrics, reasons)
        )

    def save_report(
        self, params: AdaptiveParameters, metrics: dict[str, Any], reasons: list[str]
    ) -> None:
        """Persist the latest adaptive proposal report to disk."""
        output = Path(self.config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "updated_at": _utcnow(),
                    "mode": self.config.mode,
                    "params": asdict(params.bounded()),
                    "metrics": metrics,
                    "reasons": reasons,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )


def analyze(signals, trades, current, config=None, online=False):
    """Delegate adaptive replay analysis to the isolated analysis module."""
    return _analysis_module().analyze(signals, trades, current, config, online)


def propose_parameters(signals, trades, current, config=None):
    """Return an adaptive parameter proposal."""
    return _analysis_module().propose_parameters(signals, trades, current, config)


def run_offline_analysis(*args, **kwargs):
    """Run offline attribution analysis through the analysis module."""
    return _analysis_module().run_offline_analysis(*args, **kwargs)


def main():
    """Run the adaptive analysis command-line interface."""
    return _analysis_module().main()


if __name__ == "__main__":
    main()
