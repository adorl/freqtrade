"""Focused tests for runtime prompt horizons and current Telegram observability state."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from freqtrade.freqllm.advisor.service import (
    AnalysisContext,
    LLMAdvisor,
    PredictionWindow,
)
from freqtrade.freqllm.observability import telegram as telegram_observability
from freqtrade.rpc.telegram import Telegram


def _context_history(pair: str) -> list[dict[str, str]]:
    """Return one prior conversation turn for prompt tests."""
    return [{"role": "assistant", "content": f"prior:{pair}"}]


def _exchange_rate(pair: str, *, side: str, is_short: bool, refresh: bool) -> float:
    """Return a deterministic current exit rate."""
    assert (pair, side, is_short, refresh) == ("BTC/USDT", "exit", False, False)
    return 110.0


def _trade() -> SimpleNamespace:
    """Build one representative open FreqLLM trade."""
    return SimpleNamespace(
        pair="BTC/USDT",
        strategy="LLMStrategy",
        is_short=False,
        open_rate=100.0,
        close_rate=None,
        leverage=2,
        stake_amount=50.0,
        enter_tag="freqai_long",
        open_date=datetime.now(UTC) - timedelta(hours=2),
        calc_profit_ratio=lambda rate: rate / 100.0 - 1.0,
    )


def _global_stats() -> dict[str, float]:
    """Return deterministic aggregate performance statistics."""
    return {
        "total_trades": 2,
        "win_rate": 0.5,
        "avg_win_ratio": 0.1,
        "avg_loss_ratio": 0.05,
    }


def _pair_stats(pair: str) -> dict[str, float | int]:
    """Return deterministic per-pair performance statistics."""
    if pair != "BTC/USDT":
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "max_consecutive_losses": 0,
        }
    return {
        "total_trades": 2,
        "win_rate": 0.5,
        "avg_profit": 0.025,
        "max_consecutive_losses": 1,
    }


def _token_stats() -> dict:
    """Return deterministic cumulative token statistics."""
    return {
        "total_calls": 1,
        "total_prompt_tokens": 80,
        "total_completion_tokens": 20,
        "total_tokens": 100,
        "total_cost": 0.01,
        "pair_stats": {"BTC/USDT": {"calls": 1, "total_tokens": 100, "cost": 0.01}},
    }


def _advisor_for_prompt_tests() -> LLMAdvisor:
    """Create a minimal advisor instance for pure prompt rendering tests."""
    advisor = object.__new__(LLMAdvisor)
    advisor.config = SimpleNamespace(
        execution=SimpleNamespace(leverage_max=4),
        sizing=SimpleNamespace(max_stake_ratio=0.25),
    )
    advisor.__dict__["_collaborators"] = SimpleNamespace(
        context_manager=SimpleNamespace(get_history=_context_history)
    )
    return advisor


def _current_strategy() -> SimpleNamespace:
    """Build current LLMStrategy runtime state without legacy managers."""
    advice_time = datetime.now(UTC) - timedelta(minutes=3)
    return SimpleNamespace(
        get_strategy_name=lambda: "LLMStrategy",
        dp=SimpleNamespace(_exchange=SimpleNamespace(get_rate=_exchange_rate)),
        config={"exchange": {"pair_whitelist": ["BTC/USDT"]}},
        strategy_collaborators=SimpleNamespace(
            runtime=SimpleNamespace(
                llm_advice={
                    "BTC/USDT": {
                        "action": "open_long",
                        "direction_bias": "long",
                        "confidence": 0.8,
                        "event_risk": 0.1,
                        "avoid_trade": False,
                        "reason": "trend and positioning align",
                    }
                },
                llm_advice_time={"BTC/USDT": advice_time},
                token_tracker=SimpleNamespace(get_stats_dict=_token_stats),
            ),
            performance_tracker=SimpleNamespace(
                get_global_numeric_stats=_global_stats,
                get_stats=_pair_stats,
            ),
        ),
    )


def test_prompts_use_injected_timeframe_and_horizon() -> None:
    """Both prompt layers should describe the injected prediction window."""
    advisor = _advisor_for_prompt_tests()
    window = PredictionWindow.from_values("30m", 7)
    build_user_prompt = vars(LLMAdvisor)["_build_user_prompt"]
    build_messages = vars(LLMAdvisor)["_build_messages"]

    user_prompt = build_user_prompt(advisor, "BTC/USDT", AnalysisContext(), window)
    messages = build_messages(advisor, "BTC/USDT", user_prompt, window)

    assert "next 7 30m strategy candle(s)" in user_prompt
    assert "next 7\n  30m strategy candle(s)" in messages[0]["content"]
    assert "1-2" not in user_prompt
    assert "1-2" not in messages[0]["content"]
    assert messages[1] == {"role": "assistant", "content": "prior:BTC/USDT"}


def test_prediction_window_rejects_invalid_runtime_values() -> None:
    """Malformed timeframe or horizon values should fail closed."""
    invalid_values = (("", 2), ("15 minutes", 2), ("15m", 0), ("15m", True), ("15m", 1.5))
    for timeframe, horizon in invalid_values:
        try:
            PredictionWindow.from_values(timeframe, horizon)
        except ValueError:
            continue
        raise AssertionError(f"Invalid prediction window was accepted: {timeframe}, {horizon}")


def test_orders_render_current_rate_and_current_advice() -> None:
    """Orders should show live profit and advice from current strategy state."""
    with patch.object(telegram_observability.Trade, "get_open_trades", return_value=[_trade()]):
        text = telegram_observability.get_llm_orders_text(_current_strategy())

    assert "Current: `110`" in text
    assert "P&L: `+10.00%`" in text
    assert r"open\_long / long" in text
    assert "confidence `80%`" in text


def test_profit_and_tokens_use_current_tracker_attributes() -> None:
    """Performance and token views should use current tracker attributes."""
    with patch.object(telegram_observability.Trade, "get_open_trades", return_value=[_trade()]):
        strategy = _current_strategy()
        profit_text = telegram_observability.get_llm_profit_text(strategy)
        token_text = telegram_observability.get_llm_tokens_text(strategy)

    assert "Total Trades: `2`" in profit_text
    assert "avg `+2.50%`" in profit_text
    assert "Total Tokens: `100`" in token_text
    assert "BTC/USDT" in token_text


def test_rpc_strategy_identification_uses_strategy_name() -> None:
    """RPC identification should accept only the actual LLMStrategy name."""
    llm_strategy = _current_strategy()
    bot = SimpleNamespace(strategy=llm_strategy)
    telegram = SimpleNamespace()
    telegram.__dict__["_rpc"] = SimpleNamespace(_freqtrade=bot)
    get_llm_strategy = vars(Telegram)["_get_llm_strategy"]
    assert get_llm_strategy(telegram) is llm_strategy

    bot.strategy = SimpleNamespace(get_strategy_name=lambda: "OtherStrategy")
    assert get_llm_strategy(telegram) is None
