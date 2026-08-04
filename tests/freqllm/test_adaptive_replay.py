"""Golden tests for deterministic adaptive exit replay."""

from math import isclose

from freqtrade.freqllm.adaptive import AdaptiveConfig, AdaptiveParameters
from freqtrade.freqllm.adaptive.analysis import _exit_replay_with_step


def _assert_replay(
    result: tuple[float, int, str] | None,
    expected: tuple[float, int, str],
) -> None:
    assert result is not None
    assert isclose(result[0], expected[0])
    assert result[1:] == expected[1:]


def test_replay_stop_loss_wins_when_both_barriers_are_touched() -> None:
    """A same-candle stop and take-profit collision resolves conservatively to the stop."""
    config = AdaptiveConfig(
        round_trip_cost=0.001,
        stop_loss_pct=0.010,
        take_profit_pct=0.005,
        trailing_enabled=False,
    )
    row = {
        "future_side_ret_1": 0.003,
        "future_mfe_1": 0.020,
        "future_mae_1": -0.020,
    }

    result = _exit_replay_with_step(row, AdaptiveParameters(max_hold_candles=4), config)

    _assert_replay(result, (-0.011, 1, "stop_loss"))


def test_replay_path_failure_precedes_take_profit() -> None:
    """A failed expected recovery is evaluated before a favorable intrabar extreme."""
    config = AdaptiveConfig(
        round_trip_cost=0.001,
        stop_loss_pct=0.010,
        take_profit_pct=0.005,
        trailing_enabled=False,
    )
    row = {
        "candidate_pre_profit_drawdown": -0.005,
        "future_side_ret_1": -0.007,
        "future_mfe_1": 0.020,
        "future_mae_1": -0.009,
    }

    result = _exit_replay_with_step(row, AdaptiveParameters(max_hold_candles=4), config)

    _assert_replay(result, (-0.008, 1, "recovery_failed"))


def test_replay_trailing_exit_returns_the_stop_level_net_of_cost() -> None:
    """Trailing replay uses the retained peak rather than the candle close."""
    config = AdaptiveConfig(
        round_trip_cost=0.001,
        stop_loss_pct=0.010,
        take_profit_pct=0.005,
        trailing_enabled=True,
    )
    row = {
        "future_side_ret_1": 0.004,
        "future_mfe_1": 0.006,
        "future_mae_1": -0.001,
    }

    result = _exit_replay_with_step(row, AdaptiveParameters(max_hold_candles=4), config)

    _assert_replay(result, (0.0035, 1, "trailing"))


def test_replay_max_hold_ignores_later_points_and_charges_cost() -> None:
    """Only configured replay steps are considered and terminal return is net of cost."""
    config = AdaptiveConfig(
        round_trip_cost=0.001,
        stop_loss_pct=0.010,
        take_profit_pct=0.020,
        trailing_enabled=False,
    )
    row = {
        "future_side_ret_1": 0.001,
        "future_mfe_1": 0.001,
        "future_mae_1": 0.0,
        "future_side_ret_2": 0.002,
        "future_mfe_2": 0.002,
        "future_mae_2": 0.0,
        "future_side_ret_3": 0.030,
        "future_mfe_3": 0.030,
        "future_mae_3": 0.0,
    }

    result = _exit_replay_with_step(row, AdaptiveParameters(max_hold_candles=2), config)

    _assert_replay(result, (0.001, 2, "max_hold"))


def test_replay_without_future_path_returns_none() -> None:
    """A row without any future return or excursion is not replayable."""
    result = _exit_replay_with_step({}, AdaptiveParameters(), AdaptiveConfig())

    assert result is None
