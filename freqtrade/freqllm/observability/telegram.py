"""Data aggregation and formatting for custom FreqLLM Telegram commands."""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from freqtrade.persistence import Trade


logger = logging.getLogger(__name__)

_MD_SPECIAL = ("_", "*", "`", "[")
_DISPLAY_ERRORS = (
    ArithmeticError,
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    SQLAlchemyError,
)


def _esc(text: Any) -> str:
    """Escape Telegram MarkdownV1 special characters in dynamic values."""
    escaped = str(text)
    for character in _MD_SPECIAL:
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _strategy_name(strategy: Any) -> str:
    """Return the active strategy name through the Freqtrade strategy API."""
    return str(strategy.get_strategy_name())


def _collaborators(strategy: Any) -> Any:
    """Return the required explicit FreqLLM strategy contract."""
    return strategy.strategy_collaborators


def _strategy_pairs(strategy: Any, open_trades: list[Any]) -> list[str]:
    """Collect configured, advised, and open pairs without legacy state managers."""
    pairs = {str(trade.pair) for trade in open_trades if getattr(trade, "pair", None)}
    advice = _collaborators(strategy).runtime.llm_advice
    if isinstance(advice, Mapping):
        pairs.update(str(pair) for pair in advice if pair)
    config = getattr(strategy, "config", None)
    if isinstance(config, Mapping):
        exchange = config.get("exchange", {})
        if isinstance(exchange, Mapping):
            whitelist = exchange.get("pair_whitelist", [])
            if isinstance(whitelist, list):
                pairs.update(str(pair) for pair in whitelist if isinstance(pair, str) and pair)
    return sorted(pairs)


def _open_strategy_trades(strategy: Any) -> list[Any]:
    """Return open trades belonging to the currently loaded strategy."""
    strategy_name = _strategy_name(strategy)
    return [
        trade
        for trade in Trade.get_open_trades()
        if not getattr(trade, "strategy", None) or trade.strategy == strategy_name
    ]


def _current_rate(strategy: Any, trade: Any) -> float:
    """Read the cached exit rate and fall back to a persisted trade rate."""
    try:
        data_provider = getattr(strategy, "dp", None)
        exchange = getattr(data_provider, "_exchange", None)
        if exchange is not None:
            rate = exchange.get_rate(
                trade.pair,
                side="exit",
                is_short=bool(trade.is_short),
                refresh=False,
            )
            if rate is not None and float(rate) > 0:
                return float(rate)
    except _DISPLAY_ERRORS as exc:
        logger.debug("Failed to read current rate for %s: %s", trade.pair, exc)
    return float(trade.close_rate or trade.open_rate)


def _append_trade_duration(lines: list[str], trade: Any) -> None:
    """Append the elapsed duration for an open trade."""
    if not trade.open_date:
        return
    open_date = trade.open_date
    if open_date.tzinfo is None:
        open_date = open_date.replace(tzinfo=UTC)
    hours = (datetime.now(UTC) - open_date).total_seconds() / 3600
    lines.append(f"  Duration: `{hours:.1f}h`")


def _append_open_trade(lines: list[str], strategy: Any, trade: Any) -> None:
    """Append live entry and profit fields for one open trade."""
    direction = "LONG" if not trade.is_short else "SHORT"
    current_rate = _current_rate(strategy, trade)
    profit_pct = trade.calc_profit_ratio(current_rate) * 100
    profit_marker = "+" if profit_pct >= 0 else ""
    lines.append(
        f"*{_esc(trade.pair)}* `{direction}`\n"
        f"  Entry: `{trade.open_rate:.8g}` | Current: `{current_rate:.8g}`\n"
        f"  Leverage: `{trade.leverage or 1}x` | Stake: `{trade.stake_amount:.4f}`\n"
        f"  P&L: `{profit_marker}{profit_pct:.2f}%`"
    )
    enter_tag = getattr(trade, "enter_tag", None)
    if enter_tag:
        lines.append(f"  Entry reason: `{_esc(enter_tag)}`")
    _append_trade_duration(lines, trade)
    lines.append("")


def _append_open_positions(lines: list[str], strategy: Any, open_trades: list[Any]) -> None:
    """Append all positions opened by the active strategy."""
    if not open_trades:
        lines.append("_No open positions_\n")
        return
    lines.append(f"*Open Positions:* `{len(open_trades)}`\n")
    for trade in open_trades:
        _append_open_trade(lines, strategy, trade)


def _advice_age_text(timestamp: Any) -> str:
    """Return compact elapsed time for a recorded advice timestamp."""
    if not isinstance(timestamp, datetime):
        return ""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    elapsed_minutes = max(0, int((datetime.now(UTC) - timestamp).total_seconds() / 60))
    return f" | {elapsed_minutes}m ago"


def _append_advice_status(lines: list[str], strategy: Any) -> None:
    """Append the strategy's current advice dictionary and timestamps."""
    advice_by_pair = _collaborators(strategy).runtime.llm_advice
    advice_times = _collaborators(strategy).runtime.llm_advice_time
    lines.append("*LLM Advice Status:*")
    if not isinstance(advice_by_pair, Mapping) or not advice_by_pair:
        lines.append("  _No active advice_")
        return
    times = advice_times if isinstance(advice_times, Mapping) else {}
    for pair in sorted(advice_by_pair):
        advice = advice_by_pair[pair]
        if not isinstance(advice, Mapping):
            continue
        action = _esc(advice.get("action", "hold"))
        direction = _esc(advice.get("direction_bias", "neutral"))
        confidence = float(advice.get("confidence", 0.0))
        event_risk = float(advice.get("event_risk", 0.0))
        avoid = " | AVOID" if bool(advice.get("avoid_trade", False)) else ""
        age = _advice_age_text(times.get(pair))
        lines.append(
            f"  `{_esc(pair)}`: {action} / {direction} | confidence "
            f"`{confidence:.0%}` | event risk `{event_risk:.0%}`{avoid}{age}"
        )
        reason = str(advice.get("reason", "")).strip()
        if reason:
            lines.append(f"    {_esc(reason[:160])}")


def get_llm_orders_text(strategy: Any) -> str:
    """Build active positions and current LLM advice from real strategy state."""
    try:
        lines = ["*FreqLLM Orders Overview*\n"]
        open_trades = _open_strategy_trades(strategy)
        _append_open_positions(lines, strategy, open_trades)
        _append_advice_status(lines, strategy)
        return "\n".join(lines)
    except _DISPLAY_ERRORS as exc:
        logger.warning("get_llm_orders_text error: %s", exc)
        return "Failed to retrieve FreqLLM order info."


def _kelly_value(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Calculate the raw Kelly value from aggregate performance."""
    if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0:
        return 0.0
    return win_rate - (1 - win_rate) / (avg_win / avg_loss)


def _append_global_stats(lines: list[str], stats: Mapping[str, Any]) -> None:
    """Append aggregate performance statistics without synthetic empty defaults."""
    total_trades = int(stats.get("total_trades", 0))
    lines.append("*Global Stats:*")
    if total_trades <= 0:
        lines.extend(["  _No closed trade records yet_", ""])
        return
    win_rate = float(stats.get("win_rate", 0.0))
    avg_win = float(stats.get("avg_win_ratio", 0.0))
    avg_loss = float(stats.get("avg_loss_ratio", 0.0))
    lines.extend(
        [
            f"  Total Trades: `{total_trades}`",
            f"  Win Rate: `{win_rate:.1%}`",
            f"  Avg Win: `{avg_win:.2%}` | Avg Loss: `{avg_loss:.2%}`",
            f"  Kelly Value: `{_kelly_value(win_rate, avg_win, avg_loss):.3f}`",
            "",
        ]
    )


def _append_pair_stats(lines: list[str], tracker: Any, pairs: list[str]) -> None:
    """Append tracked performance for all current strategy pairs."""
    lines.append("*Per-Pair Stats:*")
    populated = []
    for pair in pairs:
        stats = tracker.get_stats(pair)
        if int(stats.get("total_trades", 0)) > 0:
            populated.append((pair, stats))
    if not populated:
        lines.append("  _No closed trade records yet_")
        return
    for pair, stats in populated:
        lines.append(
            f"  `{_esc(pair)}`: {stats['total_trades']} trades | "
            f"win `{float(stats['win_rate']):.0%}` | "
            f"avg `{float(stats['avg_profit']):+.2%}` | "
            f"max loss streak `{stats['max_consecutive_losses']}`"
        )


def get_llm_profit_text(strategy: Any) -> str:
    """Build FreqLLM performance statistics from the current tracker."""
    try:
        lines = ["*FreqLLM Performance*\n"]
        tracker = _collaborators(strategy).performance_tracker
        if tracker is None:
            lines.append("_PerformanceTracker not initialized_")
            return "\n".join(lines)
        open_trades = _open_strategy_trades(strategy)
        _append_global_stats(lines, tracker.get_global_numeric_stats())
        _append_pair_stats(lines, tracker, _strategy_pairs(strategy, open_trades))
        return "\n".join(lines)
    except _DISPLAY_ERRORS as exc:
        logger.warning("get_llm_profit_text error: %s", exc)
        return "Failed to retrieve FreqLLM performance info."


def _append_token_totals(lines: list[str], stats: Mapping[str, Any]) -> None:
    """Append cumulative token totals."""
    lines.extend(
        [
            "*Cumulative Usage:*",
            f"  Total Calls: `{int(stats.get('total_calls', 0))}`",
            f"  Prompt Tokens: `{int(stats.get('total_prompt_tokens', 0)):,}`",
            f"  Completion Tokens: `{int(stats.get('total_completion_tokens', 0)):,}`",
            f"  Total Tokens: `{int(stats.get('total_tokens', 0)):,}`",
        ]
    )
    total_cost = float(stats.get("total_cost", 0.0))
    if total_cost > 0:
        lines.append(f"  Est. Cost: `${total_cost:.4f}`")
    if stats.get("last_updated"):
        lines.append(f"  Last Updated: `{_esc(stats['last_updated'])}`")
    lines.append("")


def _format_pair_token_usage(pair: str, stats: Mapping[str, Any], total: int) -> str:
    """Format one pair's token usage."""
    pair_tokens = int(stats.get("total_tokens", 0))
    percentage = pair_tokens / total * 100
    cost = float(stats.get("cost", 0.0))
    cost_suffix = f" | ${cost:.4f}" if cost > 0 else ""
    return (
        f"  `{_esc(pair)}`: `{percentage:.1f}%` | "
        f"calls `{int(stats.get('calls', 0))}` | tokens `{pair_tokens:,}`{cost_suffix}"
    )


def _append_pair_token_distribution(lines: list[str], stats: Mapping[str, Any]) -> None:
    """Append per-pair token distribution."""
    pair_stats = stats.get("pair_stats", {})
    if not isinstance(pair_stats, Mapping) or not pair_stats:
        lines.append("_No per-pair token usage yet_")
        return
    lines.append("*Per-Pair Distribution:*")
    sorted_pairs = sorted(
        pair_stats.items(),
        key=lambda item: int(item[1].get("total_tokens", 0)),
        reverse=True,
    )
    total_tokens = int(stats.get("total_tokens", 0)) or 1
    for pair, pair_usage in sorted_pairs:
        if isinstance(pair_usage, Mapping):
            lines.append(_format_pair_token_usage(str(pair), pair_usage, total_tokens))


def get_llm_tokens_text(strategy: Any) -> str:
    """Build LLM API token usage from the current strategy tracker."""
    try:
        lines = ["*LLM Token Usage*\n"]
        tracker = _collaborators(strategy).runtime.token_tracker
        if tracker is None:
            lines.append("_TokenTracker not initialized_")
            return "\n".join(lines)
        stats = tracker.get_stats_dict()
        if not isinstance(stats, Mapping):
            raise TypeError("TokenTracker returned invalid statistics")
        _append_token_totals(lines, stats)
        _append_pair_token_distribution(lines, stats)
        return "\n".join(lines)
    except _DISPLAY_ERRORS as exc:
        logger.warning("get_llm_tokens_text error: %s", exc)
        return "Failed to retrieve token usage info."
