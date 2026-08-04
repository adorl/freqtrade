"""Typed contracts shared by FreqAI target generation and runtime integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from pandas import DataFrame


if TYPE_CHECKING:
    from freqtrade.freqllm.adaptive import AdaptiveFeedbackManager
    from freqtrade.freqllm.advisor import LLMAdvisor
    from freqtrade.freqllm.attribution import SimpleAttributionWriter
    from freqtrade.freqllm.configuration import LLMStrategyConfig
    from freqtrade.freqllm.decision import SimpleDecisionEngine, SimpleStrategyConfig
    from freqtrade.freqllm.observability import PerformanceTracker, TokenTracker
    from freqtrade.freqllm.persistence import FreqLLMDatabase


@dataclass
class DetailBuffers:
    """Mutable arrays populated from a finer-grained candle stream."""

    entry: np.ndarray
    high: np.ndarray
    low: np.ndarray
    open: np.ndarray
    close: np.ndarray
    terminal: np.ndarray

    @classmethod
    def allocate(cls, rows: int, steps: int) -> DetailBuffers:
        """Allocate empty vector and matrix buffers."""
        matrices = [np.full((rows, steps), np.nan, dtype=float) for _ in range(4)]
        return cls(
            np.full(rows, np.nan, dtype=float),
            *matrices,
            np.full(rows, np.nan, dtype=float),
        )


@dataclass(frozen=True)
class EarlySettings:
    """Early path-failure thresholds."""

    steps: int
    mae_mfe_ratio: float
    return_mfe_ratio: float


@dataclass(frozen=True)
class BarrierSettings:
    """Fixed and trailing barrier settings."""

    stop_loss: float
    take_profit: float
    trailing_enabled: bool
    trailing_activation: float
    trailing_distance: float


@dataclass(frozen=True)
class DecaySettings:
    """Recovery, drawdown, and no-progress exit settings."""

    retention: float
    min_profit: float
    recovery_band_mult: float
    recovery_sl_floor_ratio: float
    post_drawdown_band_mult: float
    max_hold_candles: int


@dataclass(frozen=True)
class PolicySettings:
    """Normalized exit-policy settings used by every simulated row."""

    direction: str
    round_trip_cost: float
    steps_per_candle: int
    early: EarlySettings
    barrier: BarrierSettings
    decay: DecaySettings


@dataclass(frozen=True)
class PolicyArrays:
    """Numpy views of aligned future-price windows."""

    entry: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    terminal: np.ndarray
    opens: np.ndarray | None
    closes: np.ndarray | None


@dataclass
class PolicyState:
    """Path-dependent state for one simulated trade."""

    peak_profit: float = 0.0
    min_profit_before_positive: float = 0.0
    max_post_profit_drawdown: float = 0.0
    max_adverse_excursion: float = 0.0
    has_positive_profit: bool = False
    exit_profit: float | None = None


@dataclass
class PolicyResults:
    """Simulation result arrays for all input rows."""

    returns: np.ndarray
    peaks: np.ndarray
    pre_drawdowns: np.ndarray
    post_drawdowns: np.ndarray
    adverse_excursions: np.ndarray

    @classmethod
    def allocate(cls, rows: int) -> PolicyResults:
        """Allocate one NaN-filled vector per result field."""
        return cls(*(np.full(rows, np.nan, dtype=float) for _ in range(5)))


@dataclass(frozen=True)
class WindowTiming:
    """Mapping between policy steps and strategy candles."""

    early_steps: int
    steps_per_candle: int


@dataclass
class TargetWindows:
    """Aligned entry and future OHLC windows for target generation."""

    entry: pd.Series
    high: DataFrame
    low: DataFrame
    close: pd.Series
    open: DataFrame
    step_close: DataFrame
    timing: WindowTiming


@dataclass(frozen=True)
class PolicyTargets:
    """Long/short policy simulations and early-failure labels."""

    long: DataFrame
    short: DataFrame
    long_early_fail: pd.Series
    short_early_fail: pd.Series


@dataclass(frozen=True)
class StrategyExecutionSettings:
    """Portfolio limits consumed by execution callbacks."""

    portfolio_enabled: bool = True
    max_same_direction_positions: int = 0
    max_gross_exposure: float = 0.0


@dataclass
class StrategyRuntimeState:
    """Mutable state shared by the composed strategy mixins."""

    advisor: LLMAdvisor | None = None
    advisor_config: LLMStrategyConfig | None = None
    token_tracker: TokenTracker | None = None
    performance_tracker: PerformanceTracker | None = None
    database: FreqLLMDatabase | None = None
    llm_advice: dict[str, dict[str, Any]] = field(default_factory=dict)
    llm_advice_time: dict[str, datetime] = field(default_factory=dict)
    external_feature_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    fee_rate_cache: dict[str, float] = field(default_factory=dict)
    detail_dataframe_cache: dict[tuple[str, str], DataFrame] = field(default_factory=dict)
    feature_params: dict[str, Any] = field(default_factory=dict)
    is_backtest_mode: bool = False
    disable_external_in_backtest: bool = True


@dataclass(frozen=True)
class StrategyCollaborators:
    """Explicit services and mutable state used by strategy mixins."""

    simple_config: SimpleStrategyConfig
    decision_engine: SimpleDecisionEngine
    adaptive_manager: AdaptiveFeedbackManager
    attribution_writer: SimpleAttributionWriter
    execution: StrategyExecutionSettings = field(default_factory=StrategyExecutionSettings)
    runtime: StrategyRuntimeState = field(default_factory=StrategyRuntimeState)


class StrategyComposition(Protocol):
    """Host contract implemented once by the composed Freqtrade strategy."""

    strategy_collaborators: StrategyCollaborators


@dataclass(frozen=True)
class EarlyWindow:
    """Price extrema and terminal close in an early-failure window."""

    entry: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series


@dataclass(frozen=True)
class Excursions:
    """Adverse, favorable, and terminal returns for one direction."""

    adverse: pd.Series
    favorable: pd.Series
    close: pd.Series


@dataclass(frozen=True)
class FuturePrices:
    """Base-timeframe future price windows used for level-event labels."""

    close: DataFrame
    high: DataFrame
    low: DataFrame


@dataclass(frozen=True)
class LevelContext:
    """Current causal support/resistance levels and their volatility buffer."""

    nearest_resistance: pd.Series
    nearest_support: pd.Series
    strong_resistance: pd.Series
    strong_support: pd.Series
    buffer: pd.Series
    near_resistance: pd.Series
    near_support: pd.Series
