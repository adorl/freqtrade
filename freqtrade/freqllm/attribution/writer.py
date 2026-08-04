"""CSV attribution writer for the simplified FreqAI + LLM strategy."""

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from freqtrade.freqllm.attribution.records import AttributionRecord


MAX_ATTRIBUTION_HORIZON = 16
ATTRIBUTION_FIELDS = AttributionRecord.csv_fields()


def sanitize_pair(pair: str) -> str:
    """Return a filesystem-safe identifier for a trading pair."""
    return pair.replace("/", "_").replace(":", "_").replace("-", "_")


def _side_return(close_value: float, entry_open: float, side: str) -> float | str:
    """Calculate the return in the position direction."""
    if close_value <= 0:
        return ""
    if side == "short":
        return entry_open / close_value - 1.0
    return close_value / entry_open - 1.0


def _window_path(window, entry_open: float, side: str) -> tuple[float | str, float | str]:
    """Calculate favorable and adverse excursion for one candle window."""
    if window.empty:
        return "", ""
    high_value = float(window["high"].max()) if "high" in window else 0.0
    low_value = float(window["low"].min()) if "low" in window else 0.0
    if side == "short":
        return (
            entry_open / low_value - 1.0 if low_value > 0 else "",
            entry_open / high_value - 1.0 if high_value > 0 else "",
        )
    return (
        high_value / entry_open - 1.0 if high_value > 0 else "",
        low_value / entry_open - 1.0 if low_value > 0 else "",
    )


def _first_extreme(window, favorable_index, adverse_index) -> str:
    """Describe which price extreme occurred first."""
    if favorable_index is None or adverse_index is None:
        return ""
    favorable_time = window.loc[favorable_index, "date"]
    adverse_time = window.loc[adverse_index, "date"]
    if favorable_time < adverse_time:
        return "favorable_first"
    if adverse_time < favorable_time:
        return "adverse_first"
    return "same_detail_candle"


def _detail_path(
    detail_future,
    entry_time,
    step: int,
    context: dict[str, Any],
) -> tuple[float | str, float | str, str]:
    """Calculate intraperiod path metrics for a requested strategy candle."""
    step_end = entry_time + pd.Timedelta(minutes=context["timeframe_minutes"] * step)
    window = detail_future.loc[detail_future["date"] < step_end]
    if window.empty:
        return "", "", ""
    high_index = window["high"].idxmax() if "high" in window else None
    low_index = window["low"].idxmin() if "low" in window else None
    if context["side"] == "short":
        favorable_index, adverse_index = low_index, high_index
    else:
        favorable_index, adverse_index = high_index, low_index
    mfe, mae = _window_path(window, context["entry_open"], context["side"])
    return mfe, mae, _first_extreme(window, favorable_index, adverse_index)


def _usable_detail(options: dict[str, Any]) -> bool:
    """Return whether a detail dataframe can refine the attribution result."""
    detail = options.get("detail_dataframe")
    valid_frame = detail is not None and not getattr(detail, "empty", True)
    valid_columns = valid_frame and "date" in detail
    timeframe = int(options.get("timeframe_minutes", 0) or 0)
    detail_timeframe = int(options.get("detail_timeframe_minutes", 0) or 0)
    return bool(valid_columns and 0 < detail_timeframe < timeframe)


def _add_detail_metrics(
    result: dict[str, Any],
    entry_row,
    horizon: int,
    context: dict[str, Any],
) -> None:
    """Add detail-timeframe metrics to an existing attribution result."""
    entry_time = pd.to_datetime(entry_row.get("date"), utc=True, errors="coerce")
    if pd.isna(entry_time):
        return
    detail = context["detail_dataframe"].copy()
    detail["date"] = pd.to_datetime(detail["date"], utc=True, errors="coerce")
    detail = detail.dropna(subset=["date"]).sort_values("date")
    end_time = entry_time + pd.Timedelta(minutes=context["timeframe_minutes"] * horizon)
    detail_future = detail.loc[(detail["date"] >= entry_time) & (detail["date"] < end_time)]
    if detail_future.empty:
        return
    first = _detail_path(detail_future, entry_time, 1, context)
    third = _detail_path(detail_future, entry_time, 3, context)
    result.update(
        {
            "detail_used": True,
            "future_detail_first_extreme_1": first[2],
            "future_detail_first_extreme_3": third[2],
            "future_detail_mfe_1": first[0],
            "future_detail_mae_1": first[1],
            "future_detail_mfe_3": third[0],
            "future_detail_mae_3": third[1],
        }
    )


def _add_future_metrics(
    result: dict[str, Any],
    future,
    entry_open: float,
    side: str,
    horizon: int,
) -> tuple[float | str, int]:
    """Populate per-candle metrics and return the terminal return and step."""
    terminal: tuple[float | str, int] = ("", 0)
    for step in range(1, horizon + 1):
        window = future.iloc[: min(step, len(future))]
        close_value = float(window.iloc[-1].get("close", 0.0) or 0.0)
        side_return = _side_return(close_value, entry_open, side)
        mfe, mae = _window_path(window, entry_open, side)
        result.update(
            {
                f"future_side_ret_{step}": side_return,
                f"future_mfe_{step}": mfe,
                f"future_mae_{step}": mae,
            }
        )
        if side_return != "":
            terminal = side_return, step
    return terminal


class SimpleAttributionWriter:
    """Persist entry attribution and calculate its realized path metrics."""

    def __init__(self, directory: str, enabled: bool = True):
        """Initialize a writer rooted at ``directory``."""
        self.directory = Path(directory)
        self.enabled = enabled
        self._reset_pairs: set[str] = set()
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def path_for_pair(self, pair: str) -> Path:
        """Return the attribution CSV path for ``pair``."""
        return self.directory / f"entry_attribution_{sanitize_pair(pair)}.csv"

    def reset_pair_once(self, pair: str) -> None:
        """Remove a pair's previous CSV at most once for this writer."""
        if not self.enabled or pair in self._reset_pairs:
            return
        path = self.path_for_pair(pair)
        if path.exists():
            path.unlink()
        self._reset_pairs.add(pair)

    def append(self, record: AttributionRecord) -> Path | None:
        """Append one typed attribution record at the CSV boundary."""
        if not self.enabled:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for_pair(record.pair)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ATTRIBUTION_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(record.to_dict())
        return path

    @staticmethod
    def realized_metrics(
        dataframe,
        row_position: int,
        entry_price: float,
        side: str,
        **options: Any,
    ) -> dict[str, Any]:
        """Calculate realized candle and optional detail-timeframe path metrics."""
        if row_position < 0 or row_position + 1 >= len(dataframe):
            return {}
        entry_row = dataframe.iloc[row_position + 1]
        entry_open = float(entry_row.get("open", entry_price) or entry_price or 0.0)
        if entry_open <= 0:
            return {}
        horizon = max(1, int(options.get("max_hold_candles", 4) or 4))
        future = dataframe.iloc[row_position + 1 : row_position + 1 + horizon]
        if future.empty:
            return {"entry_open_next": entry_open}
        result: dict[str, Any] = {
            "entry_open_next": entry_open,
            "detail_timeframe": str(options.get("detail_timeframe", "") or ""),
            "detail_used": False,
        }
        terminal = _add_future_metrics(result, future, entry_open, side, horizon)
        result["future_side_ret_terminal"], result["future_terminal_step"] = terminal
        if _usable_detail(options):
            context = {**options, "entry_open": entry_open, "side": side}
            context["timeframe_minutes"] = int(context.get("timeframe_minutes", 0) or 0)
            _add_detail_metrics(result, entry_row, horizon, context)
        return result
