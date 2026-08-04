"""Simplified FreqAI + LLM short-horizon futures strategy.

The strategy intentionally keeps the trading loop small:
1. FreqAI predicts the configured policy horizon in strategy candles.
2. A simple edge model chooses long / short / no-trade.
3. LLM advice is used only as a market-risk and direction filter.
4. Exits are fixed TP/SL, reverse signal, edge decay, or prediction expiry.
5. Backtests export compact attribution CSV files.
"""

import logging

from sqlalchemy.exc import SQLAlchemyError

from freqtrade.enums import RunMode
from freqtrade.freqllm.adaptive import AdaptiveFeedbackManager
from freqtrade.freqllm.advisor import AdvisorDependencies, LLMAdvisor, LLMClientFactory
from freqtrade.freqllm.attribution import SimpleAttributionWriter
from freqtrade.freqllm.configuration import LLMStrategyConfig
from freqtrade.freqllm.decision import SimpleDecisionEngine, SimpleStrategyConfig
from freqtrade.freqllm.observability import PerformanceTracker, TokenTracker
from freqtrade.freqllm.persistence import FreqLLMDatabase
from freqtrade.freqllm.strategy import (
    StrategyCollaborators,
    StrategyExecutionMixin,
    StrategyExecutionSettings,
    StrategyFreqaiMixin,
    StrategyMarketDataMixin,
    StrategyRuntimeMixin,
    StrategyRuntimeState,
    StrategyTargetsMixin,
)
from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)

_STARTUP_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    OSError,
    RuntimeError,
    SQLAlchemyError,
    TypeError,
    ValueError,
)


class LLMStrategy(
    StrategyExecutionMixin,
    StrategyFreqaiMixin,
    StrategyTargetsMixin,
    StrategyRuntimeMixin,
    StrategyMarketDataMixin,
    IStrategy,
):
    """FreqLLM strategy: FreqAI policy model, deterministic gates, LLM context filter."""

    INTERFACE_VERSION = 3
    can_short = True
    position_adjustment_enable = False
    use_custom_stoploss = True
    process_only_new_candles = True

    timeframe = "15m"
    startup_candle_count = 240
    minimal_roi = {"0": 100}
    stoploss = -0.99

    freqai_target_columns = {
        "dir_class": "&-dir_class",
        "long_probability": "&-long_probability",
        "flat_probability": "&-flat_probability",
        "short_probability": "&-short_probability",
        "long_edge_q20": "&-long_edge_q20",
        "long_edge_q50": "&-long_edge_q50",
        "long_edge_q80": "&-long_edge_q80",
        "short_edge_q20": "&-short_edge_q20",
        "short_edge_q50": "&-short_edge_q50",
        "short_edge_q80": "&-short_edge_q80",
        "long_peak_profit": "&-long_peak_profit",
        "short_peak_profit": "&-short_peak_profit",
        "long_pre_profit_drawdown": "&-long_pre_profit_drawdown",
        "short_pre_profit_drawdown": "&-short_pre_profit_drawdown",
        "long_post_profit_drawdown": "&-long_post_profit_drawdown",
        "short_post_profit_drawdown": "&-short_post_profit_drawdown",
        "long_early_fail_risk": "&-long_early_fail_risk",
        "short_early_fail_risk": "&-short_early_fail_risk",
        "level_event_class": "&-level_event_class",
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        simple_config = SimpleStrategyConfig.from_config(config)
        advisor_config = LLMStrategyConfig.from_config(self._build_advisor_config())
        feature_params = config.get("freqai", {}).get("feature_parameters", {})
        ext_cfg = (config.get("llm_strategy", {}) or {}).get("external_features", {})
        disable_external = (
            bool(ext_cfg.get("disable_in_backtest", True)) if isinstance(ext_cfg, dict) else True
        )
        pf_cfg = (config.get("llm_strategy", {}) or {}).get("portfolio", {})
        pf_cfg = pf_cfg if isinstance(pf_cfg, dict) else {}
        execution_settings = StrategyExecutionSettings(
            portfolio_enabled=bool(pf_cfg.get("enabled", True)),
            max_same_direction_positions=max(
                0, int(pf_cfg.get("max_same_direction_positions", 0) or 0)
            ),
            max_gross_exposure=max(0.0, float(pf_cfg.get("max_gross_exposure", 0.0) or 0.0)),
        )
        label_horizon = self._label_horizon(
            feature_params,
            simple_config.exit.max_hold_candles,
        )
        if simple_config.exit.max_hold_candles != label_horizon:
            logger.info(
                "Aligning max_hold_candles %s -> label horizon %s",
                simple_config.exit.max_hold_candles,
                label_horizon,
            )
            simple_config.exit.max_hold_candles = label_horizon
        self._apply_policy_retrain_identifier(config, simple_config)
        database = FreqLLMDatabase(advisor_config.db_url)
        try:
            adaptive_manager = AdaptiveFeedbackManager.from_strategy_config(database, config)
        except _STARTUP_ERRORS:
            database.close()
            raise
        decision_engine = SimpleDecisionEngine(
            simple_config,
            adaptive_params=(
                adaptive_manager.current_parameters_dict() if adaptive_manager.enabled else {}
            ),
        )
        runtime = StrategyRuntimeState(
            database=database,
            feature_params=feature_params,
            disable_external_in_backtest=disable_external,
        )
        self.strategy_collaborators = StrategyCollaborators(
            simple_config=simple_config,
            decision_engine=decision_engine,
            adaptive_manager=adaptive_manager,
            attribution_writer=SimpleAttributionWriter(
                simple_config.attribution.directory,
                enabled=simple_config.attribution.enabled,
            ),
            execution=execution_settings,
            runtime=runtime,
        )

    def bot_start(self, **kwargs) -> None:
        del kwargs
        collaborators = self.strategy_collaborators
        runtime = collaborators.runtime
        runmode = getattr(self.dp, "runmode", None) if self.dp is not None else None
        llm_raw = self.config.get("llm_strategy", {})
        raw_bypass = llm_raw.get("llm_bypass_enabled", False)
        cfg_bypass = raw_bypass is True or (
            isinstance(raw_bypass, str) and raw_bypass.strip().lower() in {"1", "true", "yes", "on"}
        )
        runtime.is_backtest_mode = runmode in (RunMode.BACKTEST, RunMode.HYPEROPT) or cfg_bypass
        collaborators.adaptive_manager.online_updates_enabled = not runtime.is_backtest_mode
        collaborators.adaptive_manager.set_round_trip_cost(
            self._configured_round_trip_cost("", runtime.feature_params)
        )
        if runtime.is_backtest_mode:
            logger.info("Backtest/bypass mode uses the neutral LLM context filter.")
            return

        try:
            advisor_config = LLMStrategyConfig.from_config(self._build_advisor_config())
            advisor_config.execution.leverage_max = int(
                max(1.0, collaborators.simple_config.risk.max_leverage)
            )
            advisor_config.sizing.max_stake_ratio = float(
                collaborators.simple_config.risk.stake_ratio
            )
            database = runtime.database
            if database is None:
                raise RuntimeError("FreqLLM database unavailable")
            token_tracker = TokenTracker(
                cost_per_1k=advisor_config.llm.token_cost_per_1k,
                log_interval=advisor_config.llm.token_stats_log_interval,
                database=database,
            )
            performance_tracker = PerformanceTracker(
                win_rate_threshold=advisor_config.performance.win_rate_threshold,
                feedback_trades=advisor_config.performance.feedback_trades,
                database=database,
            )
            dependencies = AdvisorDependencies(
                data_provider=self.dp,
                wallets=self.wallets,
                exchange=getattr(self.dp, "_exchange", None) if self.dp is not None else None,
                token_tracker=token_tracker,
                performance_tracker=performance_tracker,
                database=database,
            )
            client = LLMClientFactory.create(advisor_config)
            try:
                advisor = LLMAdvisor(advisor_config, client, dependencies)
            except _STARTUP_ERRORS:
                client.close()
                raise
            runtime.advisor_config = advisor_config
            runtime.token_tracker = token_tracker
            runtime.performance_tracker = performance_tracker
            runtime.advisor = advisor
            collaborators.performance_tracker = performance_tracker
            logger.info("Live LLM advisor enabled.")
        except _STARTUP_ERRORS as exc:
            runtime.advisor = None
            logger.warning("LLM advisor disabled; strategy will use neutral LLM filter: %s", exc)

    def bot_stop(self, **kwargs) -> None:
        """Release the composition-root-owned provider and database resources."""
        logger.debug("Stopping FreqLLM resources with callback context keys: %s", sorted(kwargs))
        runtime = self.strategy_collaborators.runtime
        advisor = runtime.advisor
        database = runtime.database
        runtime.advisor = None
        runtime.database = None
        if advisor is not None:
            try:
                advisor.close()
            except _STARTUP_ERRORS as exc:
                logger.warning("Failed to close LLM advisor resources: %s", exc)
        if database is not None:
            try:
                database.close()
            except _STARTUP_ERRORS as exc:
                logger.warning("Failed to close FreqLLM database resources: %s", exc)
