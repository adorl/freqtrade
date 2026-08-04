"""Minimal configuration schema for the simplified FreqAI + LLM strategy."""

import ipaddress
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

_DENIED_INTERNAL_PREFIXES = {9, 10, 11, 21, 30}
_LOCAL_HOSTNAMES = {"localhost", "localhost.localdomain"}


def _resolve_env(value: str) -> str:
    if not value:
        return value
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    if value.startswith("$"):
        return os.environ.get(value[1:], "")
    return value


def _is_environment_reference(value: str) -> bool:
    return value.startswith("$") and bool(value.removeprefix("$").strip("{}"))


def _validate_api_base_url(provider: str, value: str) -> None:
    if not value:
        return
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname or parsed.username or parsed.password:
        raise ValueError(
            "llm_strategy.llm.api_base_url must be an absolute URL without credentials"
        )
    is_local_ollama = provider == "ollama" and hostname in _LOCAL_HOSTNAMES
    if parsed.scheme != "https" and not (is_local_ollama and parsed.scheme == "http"):
        raise ValueError("llm_strategy.llm.api_base_url must use HTTPS")
    if hostname in _LOCAL_HOSTNAMES or hostname.endswith((".local", ".internal")):
        if not is_local_ollama:
            raise ValueError("llm_strategy.llm.api_base_url must not target an internal host")
        return
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    first_octet = int(str(address).split(".", maxsplit=1)[0]) if address.version == 4 else None
    if not address.is_global or first_octet in _DENIED_INTERNAL_PREFIXES:
        raise ValueError("llm_strategy.llm.api_base_url must not target an internal IP address")


@dataclass
class LLMConfig:
    """Connection and accounting settings for the configured LLM provider."""

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    api_base_url: str = ""
    token_cost_per_1k: float = 0.0
    token_stats_log_interval: int = 10


@dataclass
class ContextConfig:
    """Conversation history settings."""

    enabled: bool = True
    max_turns: int = 1


@dataclass
class ScheduleConfig:
    """Periodic analysis scheduling settings."""

    analysis_interval_minutes: int = 15


@dataclass
class _MarketHistoryConfig:
    """Historical market-data collection settings."""

    kline_limit: int = 80
    kline_timeframes: list[str] = field(default_factory=lambda: ["15m", "30m", "1h"])
    long_short_ratio_period: str = "15m"
    long_short_ratio_limit: int = 48
    ls_ratio_display_limit: int = 3
    ls_ratio_history_days: int = 30
    ls_ratio_percentile_window_days: int = 30


@dataclass
class _MarketSnapshotConfig:
    """Current market snapshot collection settings."""

    ls_ratio_change_rate_periods: int = 24
    orderbook_depth: int = 25
    open_interest_enabled: bool = True
    open_interest_limit: int = 48
    reference_price_klines_enabled: bool = True
    reference_price_kline_limit: int = 16


@dataclass
class MarketDataConfig(_MarketHistoryConfig, _MarketSnapshotConfig):
    """Combined historical and current market-data settings."""

    spot_trade_flow_enabled: bool = True
    spot_trade_limit: int = 500
    leverage_stats_enabled: bool = True


@dataclass
class ExecutionConfig:
    """Internal advisor limits copied from the deterministic risk policy."""

    leverage_max: int = 5


@dataclass
class SizingConfig:
    """Internal advisor limits copied from the deterministic risk policy."""

    max_stake_ratio: float = 0.25


@dataclass
class PerformanceConfig:
    """Trade feedback and performance threshold settings."""

    feedback_trades: int = 5
    win_rate_threshold: float = 0.4


GROUP_CONFIG_TYPES: dict[str, type[Any]] = {
    "llm": LLMConfig,
    "context": ContextConfig,
    "schedule": ScheduleConfig,
    "market_data": MarketDataConfig,
    "performance": PerformanceConfig,
}


@dataclass
class _StrategyCoreConfig:
    """Core fields shared by the complete runtime strategy configuration."""

    db_url: str = "sqlite:///user_data/freqllm/freqllm.db"
    llm_bypass_enabled: bool = False
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


@dataclass
class LLMStrategyConfig(_StrategyCoreConfig):
    """Runtime config needed only by live LLM advisor and data collectors."""

    sizing: SizingConfig = field(default_factory=SizingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLMStrategyConfig":
        """Build and validate strategy settings from a Freqtrade configuration object."""
        if not isinstance(config, dict):
            raise TypeError("Configuration root must be an object")
        raw = config.get("llm_strategy", {})
        if not isinstance(raw, dict):
            raise TypeError("llm_strategy must be an object")
        allowed_options = {"db_url", "llm_bypass_enabled", *GROUP_CONFIG_TYPES}
        unknown_options = sorted(set(raw) - allowed_options)
        if unknown_options:
            path = f"llm_strategy.{unknown_options[0]}"
            raise ValueError(f"Unknown configuration option: {path}")
        llm_section = raw.get("llm", {})
        if not isinstance(llm_section, dict):
            raise TypeError("llm_strategy.llm must be an object")
        api_key_source = str(llm_section.get("api_key", "") or "")
        if api_key_source and not _is_environment_reference(api_key_source):
            raise ValueError("llm_strategy.llm.api_key must reference an environment variable")
        db_url_source = str(raw.get("db_url", "") or "")
        if db_url_source and not _is_environment_reference(db_url_source):
            parsed_db_url = urlparse(db_url_source)
            if parsed_db_url.password is not None:
                raise ValueError(
                    "llm_strategy.db_url credentials must come from an environment variable"
                )
        inst = cls()
        if db_url_source:
            inst.db_url = db_url_source
        inst.llm_bypass_enabled = inst._coerce(
            raw.get("llm_bypass_enabled", inst.llm_bypass_enabled),
            inst.llm_bypass_enabled,
            "llm_strategy.llm_bypass_enabled",
        )
        for group_name, group_type in GROUP_CONFIG_TYPES.items():
            section = raw.get(group_name, {})
            if not isinstance(section, dict):
                raise TypeError(f"llm_strategy.{group_name} must be an object")
            inst._apply_group(group_name, group_type, section)
        inst.db_url = _resolve_env(inst.db_url)
        inst.llm.api_key = _resolve_env(inst.llm.api_key)
        inst._validate()
        return inst

    def to_dict(self) -> dict[str, Any]:
        """Return a logging-safe representation with credentials redacted."""
        result = asdict(self)
        result["db_url"] = "***" if self.db_url else ""
        result["llm"]["api_key"] = "***" if self.llm.api_key else ""
        return result

    def _apply_group(self, group_name: str, group_type: type[Any], section: dict[str, Any]) -> None:
        target = getattr(self, group_name)
        valid_fields = {f.name: f.type for f in fields(group_type)}
        for key, value in section.items():
            path = f"llm_strategy.{group_name}.{key}"
            if key not in valid_fields:
                raise ValueError(f"Unknown configuration option: {path}")
            current = getattr(target, key)
            setattr(target, key, self._coerce(value, current, path))

    @staticmethod
    def _coerce(value: Any, current: Any, path: str) -> Any:
        try:
            if isinstance(current, bool):
                return LLMStrategyConfig._coerce_bool(value)
            if isinstance(current, list):
                if not isinstance(value, list):
                    raise TypeError("expected a list")
                return list(value)
            if isinstance(current, dict):
                if not isinstance(value, dict):
                    raise TypeError("expected an object")
                return dict(value)
            if isinstance(current, (int, float, str)):
                return type(current)(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {path}: {value!r}") from exc
        return value

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
            return True
        if isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off"}:
            return False
        raise TypeError("expected a boolean")

    def _validate(self) -> None:
        self.llm.provider = self.llm.provider.strip().lower()
        _validate_api_base_url(self.llm.provider, self.llm.api_base_url.strip())
        self.context.max_turns = max(0, int(self.context.max_turns))
        self.schedule.analysis_interval_minutes = max(
            1, int(self.schedule.analysis_interval_minutes)
        )
        self.market_data.kline_limit = max(10, int(self.market_data.kline_limit))
        self.market_data.long_short_ratio_limit = max(
            1, int(self.market_data.long_short_ratio_limit)
        )
        self.market_data.ls_ratio_display_limit = max(
            1, int(self.market_data.ls_ratio_display_limit)
        )
        self.market_data.orderbook_depth = max(5, int(self.market_data.orderbook_depth))
        self.market_data.open_interest_limit = max(1, int(self.market_data.open_interest_limit))
        self.market_data.reference_price_kline_limit = max(
            1, int(self.market_data.reference_price_kline_limit)
        )
        self.market_data.spot_trade_limit = max(1, int(self.market_data.spot_trade_limit))
        self.execution.leverage_max = max(1, int(self.execution.leverage_max))
        self.sizing.max_stake_ratio = min(max(float(self.sizing.max_stake_ratio), 0.0), 1.0)
        self.llm.token_stats_log_interval = max(1, int(self.llm.token_stats_log_interval))
        if not self.llm_bypass_enabled and not self.llm.api_key and self.llm.provider != "ollama":
            logger.warning("API key for provider '%s' is not configured.", self.llm.provider)
