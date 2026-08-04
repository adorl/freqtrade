"""
LLM Advisor - orchestrates data collection, LLM calls, and advice parsing.
Integrates all sub-modules to produce structured trading recommendations.
"""

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from freqtrade.freqllm.advisor.clients import LLMClient, LLMResponse
from freqtrade.freqllm.advisor.clients.context_manager import ConversationContextManager
from freqtrade.freqllm.advisor.collectors.account_info import AccountInfoCollector
from freqtrade.freqllm.advisor.collectors.long_short_ratio import LongShortRatioCollector
from freqtrade.freqllm.advisor.collectors.market_data import MarketDataCollector
from freqtrade.freqllm.configuration import LLMStrategyConfig
from freqtrade.freqllm.observability import PerformanceTracker, TokenTracker


logger = logging.getLogger(__name__)

_COLLECTION_ERRORS = (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError)
_LLM_ERRORS = (OSError, RuntimeError, ValueError)


@dataclass(frozen=True)
class AdvisorDependencies:
    """External dependencies used to assemble an advisor's collaborators."""

    data_provider: Any
    wallets: Any
    exchange: Any
    token_tracker: TokenTracker
    performance_tracker: PerformanceTracker
    database: Any = None


@dataclass(frozen=True)
class AdvisorCollaborators:
    """Runtime services owned by an advisor instance."""

    market_collector: MarketDataCollector
    ls_collector: LongShortRatioCollector
    account_collector: AccountInfoCollector
    context_manager: ConversationContextManager
    token_tracker: TokenTracker
    performance_tracker: PerformanceTracker


@dataclass(frozen=True)
class AnalysisRiskContext:
    """Normalized portfolio-risk inputs for one analysis cycle."""

    is_high_vol: bool = False
    portfolio_exposure: Any = None
    max_portfolio_exposure: Any = None
    same_dir_counts: Any = None
    max_same_direction_positions: Any = None
    is_black_swan_paused: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AnalysisRiskContext":
        """Create normalized risk inputs from keyword analysis options."""
        return cls(
            is_high_vol=bool(values.get("is_high_vol", False)),
            portfolio_exposure=values.get("portfolio_exposure"),
            max_portfolio_exposure=values.get("max_portfolio_exposure"),
            same_dir_counts=values.get("same_dir_counts"),
            max_same_direction_positions=values.get("max_same_direction_positions"),
            is_black_swan_paused=bool(values.get("is_black_swan_paused", False)),
        )


@dataclass(frozen=True)
class AnalysisContext:
    """Prompt-ready text collected for one pair."""

    market: str = ""
    long_short: str = ""
    account: str = ""
    position: str = ""
    feedback: str = ""
    is_high_vol: bool = False


@dataclass(frozen=True)
class PredictionWindow:
    """Validated strategy timeframe and FreqAI prediction horizon."""

    timeframe: str
    candles: int

    @classmethod
    def from_values(cls, timeframe: str, candles: int) -> "PredictionWindow":
        """Validate runtime prediction-window values supplied by the strategy."""
        if not isinstance(timeframe, str) or not re.fullmatch(r"[1-9]\d*[smhdwM]", timeframe):
            raise ValueError(f"Invalid strategy timeframe: {timeframe!r}")
        if isinstance(candles, bool) or not isinstance(candles, int) or candles <= 0:
            raise ValueError(f"Prediction horizon must be a positive integer, got {candles!r}")
        return cls(timeframe=timeframe, candles=candles)


# System prompt template
SYSTEM_PROMPT = """You are an expert cryptocurrency futures market-context advisor.
Your task is to analyse the provided market, positioning, account, and position context,
then return a precise trading filter in **strict JSON format**.

Required JSON fields:
- "action": one of "open_long", "open_short", "close", "hold"
- "leverage": integer 1-{max_leverage} (advisory only; execution applies its own cap)
- "stop_loss_ratio": float 0.0-1.0 (advisory only; execution uses fixed configured stops)
- "reason": string (concise explanation, max 200 chars)
- "confidence": float 0.0-1.0
- "event_risk": float 0.0-1.0 (higher means event / narrative / liquidation risk is elevated)
- "leverage_cap_multiplier": float 0.1-1.0 (risk haircut for execution leverage)
- "direction_bias": one of "long", "short", "neutral"

Optional reference fields:
- "stake_ratio": float 0.0-{max_stake_ratio} (advisory only)
- "avoid_trade": boolean
- "thesis_invalidators": array of short strings

Important framework context:
- FreqAI is the primary fast signal and predicts the next {prediction_horizon}
  {timeframe} strategy candle(s).
- The LLM must act as a contextual filter: market regime, positioning/crowding,
  funding/OI risk, account/position risk, and whether new exposure should be avoided.
- Do not perform scenario-by-scenario technical analysis or override FreqAI with
  indicator-style reasoning. Prefer neutral/avoid when context is unclear.
- Execution is deliberately simple: path-quality adjusted stake/leverage, fixed TP/SL,
  reverse-signal exit, path-decay exit, trailing protection, and prediction-window expiry.

Rules:
1. Only output valid JSON. Do not include any text outside the JSON object.
2. Always include every required field, even when action is "hold" or "close".
3. Use action="hold" when context does not clearly support directional exposure.
4. Use avoid_trade=true when event risk, crowding, account risk, or data quality
   makes fresh exposure unattractive.
5. direction_bias should describe the slow contextual prior, not a short-term indicator trigger.
6. Never exceed the configured leverage or stake-ratio advisory limits.
7. This is a futures/perpetual market; both long and short are allowed.
"""


class LLMAdvisor:
    """
    Central advisor class that:
    1. Collects market data, long/short ratios, account info, and performance history.
    2. Builds a structured prompt and calls the LLM.
    3. Parses the JSON response into a validated advice dict.
    4. Records token usage and updates conversation context.
    """

    def __init__(
        self,
        config: LLMStrategyConfig,
        llm_client: LLMClient,
        dependencies: AdvisorDependencies,
    ):
        """Initialize the advisor from one explicit dependency bundle."""
        self.config = config
        self.llm_client = llm_client
        self._collaborators = AdvisorCollaborators(
            market_collector=MarketDataCollector(
                dependencies.data_provider,
                config,
                exchange=dependencies.exchange,
                database=dependencies.database,
            ),
            ls_collector=LongShortRatioCollector(
                dependencies.exchange,
                database=dependencies.database,
                config=config,
            ),
            account_collector=AccountInfoCollector(dependencies.wallets, config),
            context_manager=ConversationContextManager(
                context_enabled=config.context.enabled,
                context_max_turns=config.context.max_turns,
            ),
            token_tracker=dependencies.token_tracker,
            performance_tracker=dependencies.performance_tracker,
        )

    def close(self) -> None:
        """Release resources owned by the provider client."""
        self.llm_client.close()

    @property
    def market_collector(self) -> MarketDataCollector:
        """Return the market-data collector collaborator."""
        return self._collaborators.market_collector

    @property
    def ls_collector(self) -> LongShortRatioCollector:
        """Return the long/short-ratio collector collaborator."""
        return self._collaborators.ls_collector

    @property
    def account_collector(self) -> AccountInfoCollector:
        """Return the account collector collaborator."""
        return self._collaborators.account_collector

    @property
    def context_manager(self) -> ConversationContextManager:
        """Return the conversation-context collaborator."""
        return self._collaborators.context_manager

    @property
    def token_tracker(self) -> TokenTracker:
        """Return the token-usage tracker collaborator."""
        return self._collaborators.token_tracker

    @property
    def performance_tracker(self) -> PerformanceTracker:
        """Return the performance-feedback tracker collaborator."""
        return self._collaborators.performance_tracker

    def analyze(
        self,
        pair: str,
        pair_list: list[str] | None = None,
        sr_context: str | None = None,
        **risk_context: Any,
    ) -> dict[str, Any]:
        """
        Run a full analysis cycle for the given pair.

        Steps:
        1. Collect market data, L/S ratios, account info, and performance feedback.
        2. Build the user prompt with support/resistance and risk context.
        3. Call the LLM (with conversation history if context is enabled).
        4. Parse and validate the JSON response.
        5. Record token usage and update conversation context.

        :param pair: Trading pair to analyse.
        :param pair_list: All managed pairs (for account context).
        :param sr_context: Optional support/resistance level summary text.
        :param timeframe: Active strategy candle timeframe, supplied as runtime context.
        :param prediction_horizon: Active FreqAI horizon, supplied as runtime context.
        :param is_high_vol: Whether high-volatility mode is active.
        :param portfolio_exposure: Current total leveraged portfolio exposure ratio.
        :param max_portfolio_exposure: Configured maximum leveraged portfolio exposure.
        :param same_dir_counts: Current open position counts by direction.
        :param max_same_direction_positions: Configured max open positions per direction.
        :param is_black_swan_paused: Whether portfolio-level black swan pause is active.
        :return: Validated advice dict.
        """
        managed_pairs = pair_list if pair_list is not None else [pair]
        prediction_window = PredictionWindow.from_values(
            risk_context.pop("timeframe", None),
            risk_context.pop("prediction_horizon", None),
        )
        risk = AnalysisRiskContext.from_mapping(risk_context)
        logger.info("[%s] Starting LLM analysis...", pair)

        context = self._collect_analysis_context(pair, managed_pairs, risk)
        user_prompt = self._build_user_prompt(pair, context, prediction_window, sr_context)
        response = self._request_advice(pair, user_prompt, prediction_window)
        if response is None:
            advice = self._hold_advice(reason="LLM call failed")
            advice["available"] = False
            return advice

        advice = self._parse_response(pair, response.content)
        self._record_analysis(pair, user_prompt, response, advice)
        return advice

    def _collect_analysis_context(
        self,
        pair: str,
        pair_list: list[str],
        risk: AnalysisRiskContext,
    ) -> AnalysisContext:
        """Collect all prompt sections while isolating recoverable source failures."""
        market_text, market_data = self._collect_market_context(pair)
        return AnalysisContext(
            market=market_text,
            long_short=self._collect_long_short_context(pair),
            account=self._collect_account_context(pair, pair_list, risk),
            position=self._collect_position_context(pair, market_data),
            feedback=self._collect_feedback_context(pair),
            is_high_vol=risk.is_high_vol,
        )

    def _collect_market_context(self, pair: str) -> tuple[str, dict[str, Any]]:
        """Collect market data through its public API and render it once."""
        market_data: dict[str, Any] = {}
        try:
            market_data = self.market_collector.collect(pair)
            return self.market_collector.build_context_text(pair, market_data), market_data
        except _COLLECTION_ERRORS as error:
            logger.warning("[%s] Market data collection failed: %s", pair, error)
            fallback = f"## Market Data - {pair}\n(Collection failed: {error})\n"
            return fallback, market_data

    def _collect_long_short_context(self, pair: str) -> str:
        """Collect the long/short positioning prompt section."""
        try:
            settings = self.config.market_data
            return self.ls_collector.build_context_text(
                pair,
                period=settings.long_short_ratio_period,
                limit=settings.long_short_ratio_limit,
                display_limit=settings.ls_ratio_display_limit,
            )
        except _COLLECTION_ERRORS as error:
            logger.warning("[%s] L/S ratio collection failed: %s", pair, error)
            return ""

    def _collect_account_context(
        self,
        pair: str,
        pair_list: list[str],
        risk: AnalysisRiskContext,
    ) -> str:
        """Collect the portfolio-risk prompt section."""
        try:
            return self.account_collector.build_account_context_text(
                pair_list,
                portfolio_exposure=risk.portfolio_exposure,
                max_portfolio_exposure=risk.max_portfolio_exposure,
                same_dir_counts=risk.same_dir_counts,
                max_same_direction_positions=risk.max_same_direction_positions,
                is_black_swan_paused=risk.is_black_swan_paused,
            )
        except _COLLECTION_ERRORS as error:
            logger.warning("[%s] Account info collection failed: %s", pair, error)
            return ""

    def _collect_position_context(self, pair: str, market_data: Mapping[str, Any]) -> str:
        """Collect the current-position prompt section."""
        try:
            current_price = self._get_current_price(pair, market_data)
            return self.account_collector.build_position_context_text(pair, current_price)
        except _COLLECTION_ERRORS as error:
            logger.warning("[%s] Position info collection failed: %s", pair, error)
            return ""

    def _collect_feedback_context(self, pair: str) -> str:
        """Collect the realized-performance prompt section."""
        try:
            return self.performance_tracker.get_feedback_text(pair)
        except _COLLECTION_ERRORS as error:
            logger.warning("[%s] Performance feedback failed: %s", pair, error)
            return ""

    def _request_advice(
        self,
        pair: str,
        user_prompt: str,
        prediction_window: PredictionWindow,
    ) -> LLMResponse | None:
        """Call the LLM and convert expected provider failures to unavailable advice."""
        try:
            messages = self._build_messages(pair, user_prompt, prediction_window)
            return self.llm_client.chat(messages, pair=pair)
        except _LLM_ERRORS as error:
            logger.error("[%s] LLM call failed: %s", pair, error)
            return None

    def _record_analysis(
        self,
        pair: str,
        user_prompt: str,
        response: LLMResponse,
        advice: dict[str, Any],
    ) -> None:
        """Record usage, sanitized conversation context, and the completion summary."""
        self.token_tracker.record(
            pair=pair,
            call_type="analysis",
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )
        assistant_message = (
            json.dumps(advice)
            if advice.get("reason", "").startswith("JSON parse failed")
            else response.content
        )
        self.context_manager.add_turn(
            pair=pair,
            user_msg=user_prompt,
            assistant_msg=assistant_message,
        )
        logger.info(
            "[%s] Analysis complete: action=%s, confidence=%s, tokens=%s",
            pair,
            advice.get("action"),
            advice.get("confidence"),
            response.total_tokens,
        )

    # Private helpers

    def _get_current_price(
        self,
        pair: str,
        market_data: Mapping[str, Any],
    ) -> float | None:
        """Extract the latest close price from data returned by the collector's public API."""
        klines = market_data.get("klines", {})
        if not isinstance(klines, Mapping):
            return None
        for timeframe in self.config.market_data.kline_timeframes:
            timeframe_data = klines.get(timeframe)
            if not isinstance(timeframe_data, list) or not timeframe_data:
                continue
            latest = timeframe_data[-1]
            if not isinstance(latest, Mapping):
                continue
            try:
                return float(latest.get("close", 0))
            except (TypeError, ValueError) as error:
                logger.debug(
                    "[%s] Invalid cached close for timeframe %s: %s",
                    pair,
                    timeframe,
                    error,
                )
        return None

    @staticmethod
    def _compact_optional_section(section: str | None) -> str:
        """Drop placeholder-only sections so the prompt stays focused."""
        if not section:
            return ""
        normalized = section.strip()
        if not normalized:
            return ""
        skip_markers = (
            "**No open position.**",
            "No completed trades recorded yet.",
        )
        if any(marker in normalized for marker in skip_markers):
            return ""
        return normalized

    def _build_user_prompt(
        self,
        pair: str,
        context: AnalysisContext,
        prediction_window: PredictionWindow,
        sr_context: str | None = None,
    ) -> str:
        """Assemble a compact prompt using the active strategy prediction window."""
        sections = [f"# Trading Analysis Request: {pair}"]

        for section in (
            context.market,
            context.long_short,
            context.account,
            context.position,
            context.feedback,
        ):
            compacted = self._compact_optional_section(section)
            if compacted:
                sections.append(compacted)

        if sr_context and sr_context.strip():
            sections.append(sr_context.strip())

        volatility_note = (
            "- High-volatility mode is active; apply extra caution to fresh exposure.\n"
            if context.is_high_vol
            else ""
        )
        sections.append(
            "## Simplified Execution Context\n"
            "- FreqAI is the primary fast signal and predicts the next "
            f"{prediction_window.candles} {prediction_window.timeframe} strategy candle(s).\n"
            "- Use this LLM response only for contextual direction bias, avoid-trade "
            "judgement, event risk, and leverage caps.\n"
            f"{volatility_note}"
            "- Avoid detailed indicator/scenario analysis; prefer neutral when context is mixed."
        )

        sections.extend(
            [
                "---",
                (
                    "Based on all the above information, provide your trading recommendation "
                    "as a single JSON object."
                ),
            ]
        )
        return "\n\n".join(sections)

    def _build_messages(
        self,
        pair: str,
        user_prompt: str,
        prediction_window: PredictionWindow,
    ) -> list[dict[str, str]]:
        """Build messages with a system prompt matching the active prediction window."""
        system_prompt = SYSTEM_PROMPT.format(
            max_leverage=self.config.execution.leverage_max,
            max_stake_ratio=self.config.sizing.max_stake_ratio,
            timeframe=prediction_window.timeframe,
            prediction_horizon=prediction_window.candles,
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        history = self.context_manager.get_history(pair)
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _parse_response(self, pair: str, content: str) -> dict[str, Any]:
        """
        Parse the LLM response content into a validated advice dict.

        Parsing strategy:
        1. Try direct JSON parse.
        2. Extract JSON block via regex (```json ... ``` or first { ... }).
        3. Fall back to hold advice on failure.
        """
        content = content.strip()

        # Attempt 1: direct parse
        try:
            advice = json.loads(content)
            return self._validate_advice(pair, advice)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract JSON block from markdown code fence
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            try:
                advice = json.loads(match.group(1))
                return self._validate_advice(pair, advice)
            except json.JSONDecodeError:
                pass

        # Attempt 3: find first { ... } block (non-greedy to avoid spanning
        # multiple JSON objects when the response contains extra text)
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            try:
                advice = json.loads(match.group(0))
                return self._validate_advice(pair, advice)
            except json.JSONDecodeError:
                pass

        # Attempt 3b: greedy fallback in case the non-greedy match cut a
        # valid nested JSON object short (e.g. {"a": {"b": 1}})
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                advice = json.loads(match.group(0))
                return self._validate_advice(pair, advice)
            except json.JSONDecodeError:
                pass

        logger.error(
            "[%s] Failed to parse LLM response as JSON. Raw content (first 200 chars): %s",
            pair,
            content[:200],
        )
        return self._hold_advice(reason="JSON parse failed")

    def _validate_advice(self, pair: str, advice: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize parsed advice, applying bounded safe defaults."""
        action = self._normalize_action(pair, advice.get("action", "hold"))
        advice["action"] = action

        leverage, leverage_valid = self._bounded_int(
            advice.get("leverage", 1),
            default=1,
            minimum=1,
            maximum=self.config.execution.leverage_max,
        )
        if not leverage_valid:
            logger.warning(
                "[%s] Invalid leverage value: %s; defaulting to 1",
                pair,
                advice.get("leverage"),
            )
        advice["leverage"] = leverage

        stake_ratio, stake_valid = self._bounded_float(
            advice.get("stake_ratio", 0.1),
            default=0.1,
            minimum=0.0,
            maximum=self.config.sizing.max_stake_ratio,
        )
        if not stake_valid:
            logger.warning(
                "[%s] Invalid stake_ratio value: %s; defaulting to 0.1",
                pair,
                advice.get("stake_ratio"),
            )
        advice["stake_ratio"] = stake_ratio
        advice["stop_loss_ratio"] = self._bounded_float(
            advice.get("stop_loss_ratio", 0.02), 0.02, 0.0, float("inf")
        )[0]
        advice["confidence"] = self._bounded_float(advice.get("confidence", 0.5), 0.5, 0.0, 1.0)[0]

        direction, direction_issue = self._normalize_direction(action, advice.get("direction_bias"))
        advice["direction_bias"] = direction
        event_risk, event_issue = self._required_ratio(advice, "event_risk", 0.0, 0.0)
        advice["event_risk"] = event_risk
        advice["avoid_trade"] = self._normalize_boolean(advice.get("avoid_trade", False))
        advice["thesis_invalidators"] = self._normalize_invalidators(
            advice.get("thesis_invalidators", [])
        )
        leverage_cap, leverage_cap_issue = self._required_ratio(
            advice, "leverage_cap_multiplier", 1.0, 0.1
        )
        advice["leverage_cap_multiplier"] = leverage_cap
        advice.setdefault("reason", "")

        issues = [
            issue
            for issue in (direction_issue, event_issue, leverage_cap_issue)
            if issue is not None
        ]
        if issues:
            logger.warning(
                "[%s] Missing or invalid required LLM fields: %s; safe defaults were applied.",
                pair,
                ", ".join(issues),
            )
        return advice

    @staticmethod
    def _normalize_action(pair: str, raw_action: Any) -> str:
        """Return a supported action or the safe hold fallback."""
        if raw_action in {"open_long", "open_short", "close", "hold"}:
            return str(raw_action)
        logger.warning(
            "[%s] Invalid action '%s' from LLM; defaulting to 'hold'.",
            pair,
            raw_action,
        )
        return "hold"

    @staticmethod
    def _bounded_float(
        value: Any,
        default: float,
        minimum: float,
        maximum: float,
    ) -> tuple[float, bool]:
        """Convert and clamp a float, returning whether conversion succeeded."""
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return default, False
        return max(minimum, min(converted, maximum)), True

    @staticmethod
    def _bounded_int(
        value: Any,
        default: int,
        minimum: int,
        maximum: int,
    ) -> tuple[int, bool]:
        """Convert and clamp an integer, returning whether conversion succeeded."""
        try:
            converted = int(value)
        except (TypeError, ValueError):
            return default, False
        return max(minimum, min(converted, maximum)), True

    @staticmethod
    def _normalize_direction(action: str, raw_direction: Any) -> tuple[str, str | None]:
        """Validate the canonical direction bias and fail closed to neutral."""
        del action
        if raw_direction in {"long", "short", "neutral"}:
            return str(raw_direction), None
        issue = (
            "direction_bias missing"
            if raw_direction is None
            else f"direction_bias invalid={raw_direction!r}"
        )
        return "neutral", issue

    @classmethod
    def _required_ratio(
        cls,
        advice: Mapping[str, Any],
        field: str,
        default: float,
        minimum: float,
    ) -> tuple[float, str | None]:
        """Validate a required bounded ratio and describe missing or invalid input."""
        raw_value = advice.get(field)
        if raw_value is None:
            return default, f"{field} missing"
        value, valid = cls._bounded_float(raw_value, default, minimum, 1.0)
        if not valid:
            return default, f"{field} invalid={raw_value!r}"
        return value, None

    @staticmethod
    def _normalize_boolean(value: Any) -> bool:
        """Normalize common textual boolean values without truthy-string surprises."""
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _normalize_invalidators(value: Any) -> list[str]:
        """Normalize thesis invalidators to at most five non-empty strings."""
        if isinstance(value, str):
            normalized = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
        else:
            normalized = []
        return normalized[:5]

    @staticmethod
    def _hold_advice(reason: str = "") -> dict[str, Any]:
        """Return a safe default 'hold' advice dict."""
        return {
            "action": "hold",
            "leverage": 1,
            "stake_ratio": 0.0,
            "stop_loss_ratio": 0.02,
            "reason": reason,
            "confidence": 0.0,
            "direction_bias": "neutral",
            "event_risk": 0.0,
            "avoid_trade": False,
            "thesis_invalidators": [],
            "leverage_cap_multiplier": 1.0,
        }
