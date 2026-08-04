"""
Account and position information collector.
Gathers wallet balances, open trade details, and per-pair position data.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from freqtrade.persistence import Trade


logger = logging.getLogger(__name__)
_COLLECTION_ERRORS = (AttributeError, RuntimeError, SQLAlchemyError, TypeError, ValueError)


class AccountInfoCollector:
    """Collect account balance and position information."""

    def __init__(self, wallets, config):
        """Initialise the collector with Wallets and strategy configuration."""
        self.wallets = wallets
        self.config = config

    def collect_account(self, pair_list: list[str]) -> dict[str, Any]:
        """Collect balances, open-trade capacity, and per-pair allocations."""
        result: dict[str, Any] = {"timestamp": datetime.now(UTC).isoformat()}
        try:
            result["available_balance"] = self.wallets.get_available_stake_amount()
            result["total_balance"] = self.wallets.get_total_stake_amount()
            result["used_balance"] = result["total_balance"] - result["available_balance"]
        except _COLLECTION_ERRORS as error:
            logger.warning("Failed to collect wallet balance: %s", error)

        try:
            open_trades = Trade.get_open_trades()
            result["open_trade_count"] = len(open_trades)
            result["max_open_trades"] = getattr(self.config, "max_open_trades", "N/A")
        except _COLLECTION_ERRORS as error:
            logger.warning("Failed to collect open trade count: %s", error)

        try:
            result["pair_allocations"] = self._collect_pair_allocations(pair_list)
        except _COLLECTION_ERRORS as error:
            logger.warning("Failed to collect pair allocations: %s", error)
        return result

    @staticmethod
    def _collect_pair_allocations(pair_list: list[str]) -> dict[str, Any]:
        """Build a per-pair allocation summary from open trades."""
        allocations = {}
        for pair in pair_list:
            open_trades = Trade.get_trades_proxy(pair=pair, is_open=True)
            if open_trades:
                allocations[pair] = {
                    "has_position": True,
                    "stake_amount": open_trades[0].stake_amount,
                }
            else:
                allocations[pair] = {"has_position": False, "stake_amount": 0.0}
        return allocations

    def collect_position(self, pair: str, current_price: float | None = None) -> dict[str, Any]:
        """Collect current position details for a trading pair."""
        result: dict[str, Any] = {
            "pair": pair,
            "has_position": False,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        try:
            open_trades = Trade.get_trades_proxy(pair=pair, is_open=True)
            if not open_trades:
                return result
            trade = open_trades[0]
            result.update(
                {
                    "has_position": True,
                    "trade_id": trade.id,
                    "direction": "short" if trade.is_short else "long",
                    "is_short": trade.is_short,
                    "open_price": trade.open_rate,
                    "amount": trade.amount,
                    "stake_amount": trade.stake_amount,
                    "leverage": trade.leverage,
                }
            )
            if current_price:
                price_difference = (
                    trade.open_rate - current_price
                    if trade.is_short
                    else current_price - trade.open_rate
                )
                result.update(
                    {
                        "current_price": current_price,
                        "unrealized_pnl": price_difference * trade.amount,
                        "unrealized_pnl_pct": price_difference / trade.open_rate * 100,
                        "position_value": current_price * trade.amount,
                    }
                )
            if trade.stop_loss:
                result["stop_loss_price"] = trade.stop_loss
                result["initial_stop_loss"] = trade.initial_stop_loss
            if trade.has_open_sl_orders:
                result["has_stoploss_order"] = True
            if trade.open_date:
                open_date = trade.open_date
                if open_date.tzinfo is None:
                    open_date = open_date.replace(tzinfo=UTC)
                duration = datetime.now(UTC) - open_date
                result["duration_hours"] = round(duration.total_seconds() / 3600, 2)
                result["open_date"] = trade.open_date.isoformat()
            liquidation_price = getattr(trade, "liquidation_price", None)
            if liquidation_price is not None:
                result["liquidation_price"] = liquidation_price
            realized_profit = getattr(trade, "realized_profit", None)
            if realized_profit is not None:
                result["realized_profit"] = realized_profit
        except _COLLECTION_ERRORS as error:
            logger.warning("[%s] Failed to collect position info: %s", pair, error)
        return result

    def build_account_context_text(
        self,
        pair_list: list[str],
        **risk_context: Any,
    ) -> str:
        """Format compact portfolio-risk information as Markdown text."""
        portfolio_exposure = risk_context.get("portfolio_exposure")
        max_portfolio_exposure = risk_context.get("max_portfolio_exposure")
        same_dir_counts = risk_context.get("same_dir_counts")
        max_same_direction_positions = risk_context.get("max_same_direction_positions")
        is_black_swan_paused = bool(risk_context.get("is_black_swan_paused", False))
        data = self.collect_account(pair_list)
        lines = ["## Portfolio Risk Context", f"**Timestamp**: {data.get('timestamp', 'N/A')}", ""]
        risk_parts = []
        open_trade_count = data.get("open_trade_count")
        max_open_trades = data.get("max_open_trades")
        if open_trade_count is not None:
            risk_parts.append(f"open trades {open_trade_count} / {max_open_trades}")
        if portfolio_exposure is not None:
            limit_str = f"{max_portfolio_exposure:.1f}x" if max_portfolio_exposure else "N/A"
            risk_parts.append(f"portfolio exposure {portfolio_exposure:.2f}x / {limit_str}")
        if same_dir_counts is not None:
            max_dir = max_same_direction_positions or "N/A"
            risk_parts.append(f"long positions {same_dir_counts.get('long', 0)} / {max_dir}")
            risk_parts.append(f"short positions {same_dir_counts.get('short', 0)} / {max_dir}")
        risk_parts.append(
            "black swan pause ACTIVE - new entries blocked"
            if is_black_swan_paused
            else "black swan pause inactive"
        )
        lines.append(f"- {' | '.join(risk_parts)}")
        lines.append("")
        return "\n".join(lines)

    def build_position_context_text(self, pair: str, current_price: float | None = None) -> str:
        """Format compact position-risk information as Markdown text."""
        data = self.collect_position(pair, current_price)
        lines = [f"## Current Position - {pair}", ""]
        if not data.get("has_position"):
            return "\n".join([*lines, "**No open position.**", ""])

        direction = data.get("direction", "N/A")
        details = [
            f"direction {'Long' if direction == 'long' else 'Short'}",
            f"entry {data.get('open_price', 'N/A')}",
        ]
        if data.get("current_price"):
            details.append(f"current {data.get('current_price', 'N/A')}")
        if data.get("unrealized_pnl") is not None:
            details.append(
                f"unrealised PnL {data['unrealized_pnl']:.4f} "
                f"({data.get('unrealized_pnl_pct', 0):.2f}%)"
            )
        details.append(f"leverage {data.get('leverage', 'N/A')}x")

        current = data.get("current_price")
        for label, value in (
            ("stop", data.get("stop_loss_price")),
            ("liquidation", data.get("liquidation_price")),
        ):
            if current and value:
                try:
                    distance = abs(float(current) - float(value)) / max(float(current), 1e-12)
                    details.append(f"{label} distance {distance:.2%}")
                except (TypeError, ValueError):
                    details.append(f"{label} {value}")
        if data.get("duration_hours") is not None:
            details.append(f"duration {data.get('duration_hours', 0):.1f}h")
        return "\n".join([*lines, f"- {' | '.join(details)}", ""])
