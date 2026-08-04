"""Adaptive replay, tuning, and offline analysis helpers."""

import argparse
import csv
import json
import math
import zipfile
from collections import namedtuple
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from freqtrade.freqllm.adaptive.feedback import (
    PARAM_BOUNDS,
    AdaptiveConfig,
    AdaptiveFeedbackManager,
    AdaptiveParameters,
    _bool,
    _clip,
    _float,
    _is_relaxation,
    _parse_time,
    _utcnow,
)
from freqtrade.freqllm.attribution import AttributionRecord
from freqtrade.freqllm.persistence import FreqLLMDatabase


def _match_key(row: AttributionRecord, reference: datetime) -> tuple[str, str, str]:
    """Return the stable pair, side, and candle key used for trade matching."""
    side = str(row.get("candidate_side") or row.get("side") or "")
    return str(row.get("pair") or ""), side, reference.isoformat()


def _trade_match_key(trade: dict[str, Any], timeframe: timedelta) -> tuple[str, str, str] | None:
    """Return a trade's matching attribution key, if its timestamp is valid."""
    opened = _parse_time(trade.get("open_time") or trade.get("open_date"))
    if opened is None:
        return None
    side = str(trade.get("side") or "")
    if not side:
        side = "short" if _bool(trade.get("is_short"), False) else "long"
    return str(trade.get("pair") or ""), side, (opened - timeframe).isoformat()


def mark_executed_signals(
    signals: list[AttributionRecord],
    trades: list[dict[str, Any]],
    cfg: AdaptiveConfig,
) -> list[AttributionRecord]:
    """Mark attribution rows that correspond to actual executed trades.
    Attribution rows are candidate signals generated in populate_entry_trend;
    the backtest engine may execute only a subset because of open-position and
    max-open-trades constraints. Matching the executed subset keeps replay-based
    tuning aligned with what the strategy actually traded.
    """
    if not signals:
        return []
    enriched = [row.with_analysis(_executed_trade=False) for row in signals]
    if not trades:
        return enriched
    by_key: dict[tuple[str, str, str], list[int]] = {}
    tf_delta = timedelta(minutes=max(1, int(cfg.timeframe_minutes)))
    for idx, row in enumerate(enriched):
        ref = _parse_time(row.reference_time)
        if ref is None:
            continue
        by_key.setdefault(_match_key(row, ref), []).append(idx)
    used: set[int] = set()
    matched = 0
    for trade in trades:
        key = _trade_match_key(trade, tf_delta)
        if key is None:
            continue
        candidates = by_key.get(key, [])
        if not candidates:
            continue
        idx = next((candidate for candidate in candidates if candidate not in used), None)
        if idx is None:
            continue
        used.add(idx)
        matched += 1
        enriched[idx] = enriched[idx].with_analysis(
            _executed_trade=True,
            _trade_id=str(trade.get("trade_id") or ""),
            _trade_profit_ratio=_float(trade.get("profit_ratio"), 0.0),
            _trade_exit_reason=str(trade.get("exit_reason") or ""),
        )
    return [row.with_analysis(_executed_match_count=matched) for row in enriched]


def _decay_weight(reference_time: Any, now: datetime | None, half_life_days: float) -> float:
    if now is None or half_life_days <= 0:
        return 1.0
    ts = _parse_time(reference_time)
    if ts is None:
        return 1.0
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life_days)


def _weighted_stats(samples: list[tuple[float, float]]) -> tuple[float, float, float]:
    sw = sum(w for _, w in samples)
    if sw <= 0 or not samples:
        return 0.0, 0.0, 0.0
    mean = sum(v * w for v, w in samples) / sw
    sw2 = sum(w * w for _, w in samples)
    n_eff = (sw * sw) / sw2 if sw2 > 0 else 0.0
    if n_eff <= 1:
        return mean, 0.0, n_eff
    var = sum(w * (v - mean) ** 2 for v, w in samples) / sw
    var *= n_eff / (n_eff - 1)
    se = math.sqrt(var / n_eff) if var > 0 else 0.0
    return mean, se, n_eff


_ReplayPolicy = namedtuple(
    "_ReplayPolicy",
    "stop_loss take_profit cost max_hold no_progress_age activation distance retention "
    "min_profit min_progress recovery_band recovery_floor peak_activation post_drawdown_band "
    "trailing_enabled",
)
_EntryPath = namedtuple("_EntryPath", "peak pre_drawdown post_drawdown")
_ReplayPoint = namedtuple("_ReplayPoint", "step close favorable adverse peak")
_ReplayResult = namedtuple("_ReplayResult", "net_return step reason")


def _build_replay_policy(params: AdaptiveParameters, cfg: AdaptiveConfig) -> _ReplayPolicy:
    cost = max(cfg.round_trip_cost, 0.0)
    profit = max(cfg.edge_decay_min_profit_pct, 0.0)
    return _ReplayPolicy(
        max(cfg.stop_loss_pct, 1e-6),
        max(cfg.take_profit_pct, 1e-6),
        cost,
        min(int(params.max_hold_candles), 16),
        int(params.no_progress_age),
        max(params.trailing_activation_pct, 1e-6),
        max(params.trailing_distance_pct, 1e-6),
        _clip(params.edge_decay_profit_retention, 0.1, 1.0),
        profit,
        cost + profit * params.no_progress_min_profit_multiplier,
        params.recovery_band_mult,
        params.recovery_sl_floor_ratio,
        params.peak_activation_ratio,
        params.post_drawdown_band_mult,
        cfg.trailing_enabled,
    )


def _entry_path(row: AttributionRecord) -> _EntryPath:
    return _EntryPath(
        max(_float(row.get("candidate_peak_profit"), 0.0) or 0.0, 0.0),
        abs(_float(row.get("candidate_pre_profit_drawdown"), 0.0) or 0.0),
        max(_float(row.get("candidate_post_profit_drawdown"), 0.0) or 0.0, 0.0),
    )


def _read_replay_point(
    row: AttributionRecord, step: int, previous_peak: float
) -> _ReplayPoint | None:
    close = _float(row.get(f"future_side_ret_{step}"), None)
    favorable = _float(row.get(f"future_mfe_{step}"), None)
    adverse = _float(row.get(f"future_mae_{step}"), None)
    if close is None and favorable is None and adverse is None:
        return None
    favorable = favorable if favorable is not None else (close or 0.0)
    return _ReplayPoint(
        step=step,
        close=close if close is not None else favorable,
        favorable=favorable,
        adverse=adverse if adverse is not None else 0.0,
        peak=max(previous_peak, favorable),
    )


def _path_shape_exit(
    point: _ReplayPoint, entry: _EntryPath, policy: _ReplayPolicy
) -> tuple[float, str] | None:
    allowed = max(
        entry.pre_drawdown * policy.recovery_band,
        policy.stop_loss * policy.recovery_floor,
    )
    if entry.pre_drawdown > 0 and point.close <= -allowed:
        return point.close, "recovery_failed"
    if (
        point.step >= policy.no_progress_age
        and point.close <= 0
        and point.peak < policy.min_progress
    ):
        return point.close, "no_progress"
    peak_activated = point.peak >= max(policy.min_profit, entry.peak * policy.peak_activation)
    if entry.peak > 0 and peak_activated and 0 < point.close <= point.peak * policy.retention:
        return point.close, "peak_retention"
    actual_post = max(0.0, point.peak - point.close)
    expected_post = max(
        entry.post_drawdown * policy.post_drawdown_band,
        policy.distance * policy.post_drawdown_band,
    )
    if (
        entry.post_drawdown > 0
        and point.peak > 0
        and point.close > 0
        and actual_post >= expected_post
    ):
        return point.close, "post_drawdown_broken"
    return None


def _trailing_exit(point: _ReplayPoint, policy: _ReplayPolicy) -> tuple[float, str] | None:
    if policy.trailing_enabled and point.peak >= policy.activation:
        retained = point.peak * policy.retention if point.peak >= policy.min_profit else 0.0
        stop_level = max(point.peak - policy.distance, retained, 0.0)
        return (stop_level, "trailing") if point.close <= stop_level else None
    if not policy.trailing_enabled and point.favorable >= policy.take_profit:
        return policy.take_profit, "take_profit"
    return None


def _point_exit(
    point: _ReplayPoint, entry: _EntryPath, policy: _ReplayPolicy
) -> tuple[float, str] | None:
    if point.adverse <= -policy.stop_loss:
        return -policy.stop_loss, "stop_loss"
    return _path_shape_exit(point, entry, policy) or _trailing_exit(point, policy)


def _replay(
    row: AttributionRecord, params: AdaptiveParameters, cfg: AdaptiveConfig
) -> _ReplayResult | None:
    policy = _build_replay_policy(params, cfg)
    entry = _entry_path(row)
    last_point = None
    for step in range(1, policy.max_hold + 1):
        point = _read_replay_point(row, step, last_point.peak if last_point else 0.0)
        if point is None:
            break
        last_point = point
        exit_value = _point_exit(point, entry, policy)
        if exit_value is not None:
            value, reason = exit_value
            return _ReplayResult(value - policy.cost, step, reason)
    if last_point is None:
        return None
    return _ReplayResult(last_point.close - policy.cost, last_point.step, "max_hold")


def _exit_replay(
    row: AttributionRecord, params: AdaptiveParameters, cfg: AdaptiveConfig
) -> float | None:
    """Replay an exit and return only its net return."""
    result = _replay(row, params, cfg)
    return result.net_return if result else None


def _exit_replay_with_step(
    row: AttributionRecord,
    params: AdaptiveParameters,
    cfg: AdaptiveConfig,
) -> tuple[float, int, str] | None:
    """Return ``(net_return, exit_step, reason)`` for execution simulation."""
    result = _replay(row, params, cfg)
    return (result.net_return, result.step, result.reason) if result else None


def _paired_replay_delta(rows, base, cand, cfg) -> tuple[float, float, float]:
    """Weighted (mean, se, n_eff) of the per-trade replay difference cand - base."""
    deltas: list[tuple[float, float]] = []
    for row in rows:
        b = _exit_replay(row, base, cfg)
        c = _exit_replay(row, cand, cfg)
        if b is None or c is None:
            continue
        weight = _float(row.get("_weight"), 1.0) or 1.0
        deltas.append((c - b, weight))
    return _weighted_stats(deltas)


def _risk_adjusted_edge_is_positive(
    row: AttributionRecord, direction: float, edge: float, confidence: float, cfg: AdaptiveConfig
) -> bool:
    """Evaluate the quantile-weighted edge gate when quantiles are available."""
    prefix = "long" if direction > 0 else "short"
    lower = _float(row.get(f"freqai_{prefix}_edge_lower"), None)
    probability = _float(row.get(f"freqai_{prefix}_probability"), confidence)
    if lower is None or probability is None:
        return True
    upper = _float(row.get(f"freqai_{prefix}_edge_upper"), None)
    if upper is None:
        adjusted = probability * edge - max(0.0, -lower)
    else:
        quantile = (
            cfg.signal_edge_q50_weight * edge
            + cfg.signal_edge_q80_weight * upper
            + cfg.signal_edge_q20_weight * lower
        )
        adjusted = probability * quantile
    return adjusted > 0.0


def _base_entry_gates(
    row: AttributionRecord,
    direction: float,
    threshold: float,
    cfg: AdaptiveConfig,
    params: AdaptiveParameters,
) -> bool:
    """Evaluate direction, confidence, edge, and early-failure gates."""
    edge = _float(row.get("candidate_edge"), None)
    gap = _float(row.get("candidate_edge_gap"), None)
    confidence = _float(row.get("freqai_confidence"), None)
    if edge is None or gap is None or confidence is None:
        return False
    minimum_confidence = _clip(cfg.signal_min_confidence + params.min_confidence_delta, 0.0, 1.0)
    early_fail = _float(row.get("candidate_early_fail_risk"), None)
    return (
        abs(direction) >= threshold
        and confidence >= minimum_confidence
        and (not cfg.require_edge_positive or edge > 0.0)
        and edge >= cfg.signal_min_edge
        and gap >= cfg.signal_min_edge_gap
        and _risk_adjusted_edge_is_positive(row, direction, edge, confidence, cfg)
        and (early_fail is None or early_fail < params.early_fail_block_threshold)
    )


def _level_room_gates(row: AttributionRecord, side: str, cfg: AdaptiveConfig) -> bool:
    """Evaluate nearby-level room and level-event conflict gates."""
    direction = "upside" if side == "long" else "downside"
    room_pct = _float(row.get(f"freqai_{direction}_room_pct"), None)
    room_atr = _float(row.get(f"freqai_{direction}_room_atr"), None)
    strong_room = _float(row.get(f"freqai_strong_{direction}_room_atr"), None)
    strength_name = "resistance" if side == "long" else "support"
    strong_strength = _float(row.get(f"freqai_strong_{strength_name}_strength"), None)
    required_room = cfg.round_trip_cost + cfg.signal_min_edge + cfg.signal_level_room_buffer_pct
    if room_pct is not None and room_pct < required_room:
        return False
    if room_atr is not None and room_atr < cfg.signal_min_level_room_atr:
        return False
    if (
        strong_room is not None
        and strong_strength is not None
        and strong_strength >= cfg.signal_min_level_strength
        and strong_room < cfg.signal_strong_level_room_atr
    ):
        return False
    event = _float(row.get("freqai_level_event_class"), None)
    if event is None:
        return True
    if side == "long":
        return event > -cfg.signal_level_event_conflict_threshold
    return side != "short" or event < cfg.signal_level_event_conflict_threshold


def _llm_entry_gates(
    row: AttributionRecord, side: str, cfg: AdaptiveConfig, params: AdaptiveParameters
) -> bool:
    """Evaluate optional LLM veto and event-risk gates."""
    if not _bool(row.get("llm_available"), False):
        return not cfg.require_llm
    event_risk = _float(row.get("llm_event_risk"), 0.0) or 0.0
    event_limit = _clip(cfg.llm_event_risk_block * params.llm_event_risk_block_mult, 0.0, 1.0)
    bias = str(row.get("llm_direction_bias") or "neutral")
    conflicts = cfg.llm_conflict_blocks and bias in {"long", "short"} and bias != side
    conflict_limit = _clip(
        cfg.llm_conflict_confidence * params.llm_conflict_confidence_mult, 0.0, 1.0
    )
    confidence = _float(row.get("llm_confidence"), 0.0) or 0.0
    return (
        not _bool(row.get("llm_avoid_trade"), False)
        and event_risk < event_limit
        and (not conflicts or confidence < conflict_limit)
    )


def _is_emitted(
    row: AttributionRecord,
    dir_threshold: float,
    cfg: AdaptiveConfig,
    params: AdaptiveParameters,
) -> bool:
    """Replay the deterministic FreqAI entry gates for adaptive evaluation."""
    direction = _float(row.get("candidate_dir_class"), None)
    if direction is None:
        return False
    side = str(row.get("candidate_side") or row.get("side") or "")
    return (
        _base_entry_gates(row, direction, dir_threshold, cfg, params)
        and _level_room_gates(row, side, cfg)
        and _llm_entry_gates(row, side, cfg, params)
    )


def _policy_objective(signals, params, cfg, dir_threshold) -> tuple[float, float, float]:
    samples: list[tuple[float, float]] = []
    for row in signals:
        if not _is_emitted(row, dir_threshold, cfg, params):
            continue
        replay = _exit_replay(row, params, cfg)
        if replay is None:
            continue
        weight = _float(row.get("_weight"), 1.0) or 1.0
        samples.append((replay, weight))
    return _weighted_stats(samples)


def _simulate_candidate(
    row: AttributionRecord, params: AdaptiveParameters, cfg: AdaptiveConfig, threshold: float
) -> tuple[AttributionRecord, int] | None:
    """Apply entry and exit replay to one otherwise-eligible candidate."""
    if not _is_emitted(row, threshold, cfg, params):
        return None
    replay = _replay(row, params, cfg)
    if replay is None:
        return None
    return row.with_analysis(_sim_replay=replay.net_return), replay.step


def simulate_executed_signals(
    signals: list[AttributionRecord],
    params: AdaptiveParameters,
    cfg: AdaptiveConfig,
    dir_threshold: float | None = None,
) -> list[AttributionRecord]:
    """Counterfactual execution simulator for entry-gate tuning.
    It scans candidate rows in chronological order, applies the candidate entry
    rule, and enforces pair-level occupancy plus max_open_trades. The replayed
    exit step determines when the slot becomes available again.
    """
    threshold = params.dir_enter_threshold if dir_threshold is None else dir_threshold
    ordered = sorted(
        signals,
        key=lambda r: _parse_time(r.get("reference_time")) or datetime.min.replace(tzinfo=UTC),
    )
    active: list[tuple[str, datetime]] = []
    executed: list[AttributionRecord] = []
    tf_delta = timedelta(minutes=max(1, int(cfg.timeframe_minutes)))
    for row in ordered:
        ref = _parse_time(row.get("reference_time"))
        if ref is None:
            continue
        active = [(pair, until) for pair, until in active if ref < until]
        pair = str(row.get("pair") or "")
        if not pair or any(active_pair == pair for active_pair, _ in active):
            continue
        if len(active) >= max(1, int(cfg.max_open_trades)):
            continue
        simulation = _simulate_candidate(row, params, cfg, threshold)
        if simulation is None:
            continue
        enriched, exit_step = simulation
        executed.append(enriched)
        active.append((pair, ref + tf_delta * max(1, exit_step)))
    return executed


def _execution_policy_objective(signals, params, cfg, dir_threshold) -> tuple[float, float, float]:
    samples: list[tuple[float, float]] = []
    for row in simulate_executed_signals(signals, params, cfg, dir_threshold):
        value = _float(row.get("_sim_replay"), None)
        if value is None:
            continue
        weight = _float(row.get("_weight"), 1.0) or 1.0
        samples.append((value, weight))
    return _weighted_stats(samples)


def _objective(signals, params, cfg, dir_threshold) -> tuple[float, float, float]:
    if cfg.execution_aware:
        return _execution_policy_objective(signals, params, cfg, dir_threshold)
    return _policy_objective(signals, params, cfg, dir_threshold)


def _circuit_breaker(
    trades: list[dict[str, Any]], cfg: AdaptiveConfig
) -> tuple[bool, dict[str, Any]]:
    ordered = sorted(trades, key=lambda t: str(t.get("close_time") or ""))
    consecutive = max_consecutive = 0
    equity = peak_equity = max_drawdown = 0.0
    for trade in ordered:
        pr = _float(trade.get("profit_ratio"), 0.0) or 0.0
        if pr < 0:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
        equity += pr
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, peak_equity - equity)
    active = False
    if cfg.breaker_consecutive_losses > 0 and consecutive >= cfg.breaker_consecutive_losses:
        active = True
    if cfg.breaker_drawdown > 0 and max_drawdown >= cfg.breaker_drawdown:
        active = True
    return active, {
        "breaker_current_consecutive_losses": consecutive,
        "breaker_max_consecutive_losses": max_consecutive,
        "breaker_max_drawdown": max_drawdown,
    }


def _best_entry_threshold(in_sample, current, cfg) -> float:
    """Return the best sufficiently sampled in-sample direction threshold."""
    lower, upper = PARAM_BOUNDS["dir_enter_threshold"]
    grid = [round(lower + 0.05 * index, 4) for index in range(int((upper - lower) / 0.05) + 1)]
    best = current.dir_enter_threshold
    best_mean, _, _ = _objective(in_sample, current, cfg, best)
    for threshold in grid:
        mean, _, sample_size = _objective(in_sample, current, cfg, threshold)
        if sample_size >= cfg.min_samples_rule and mean > best_mean + 1e-9:
            best_mean, best = mean, threshold
    return best


def _entry_oos_result(out_sample, current, cfg, best: float, relaxing: bool) -> str | None:
    """Return an OOS rejection reason, or ``None`` when the candidate passes."""
    current_mean, _, _ = _objective(out_sample, current, cfg, current.dir_enter_threshold)
    new_mean, _, new_size = _objective(out_sample, current, cfg, best)
    if relaxing and (new_size < cfg.relax_min_samples or new_mean <= current_mean + 1e-9):
        return "entry_relax_failed_oos"
    if not relaxing and (new_size < cfg.min_oos_signals or new_mean + 1e-9 < current_mean):
        return "entry_tighten_failed_oos"
    return None


def _tune_entry_threshold(in_sample, out_sample, proposed, *analysis_context) -> str | None:
    """Counterfactually scan and OOS-validate the direction entry threshold."""
    current, cfg, breaker_active, oos_ready = analysis_context
    if not in_sample:
        return None
    best = _best_entry_threshold(in_sample, current, cfg)
    if best == current.dir_enter_threshold:
        return None
    relaxing = best < current.dir_enter_threshold
    if relaxing and breaker_active:
        return "entry_relax_blocked_by_breaker"
    if not oos_ready:
        return f"entry_{'relax' if relaxing else 'tighten'}_blocked_no_oos"
    rejection = _entry_oos_result(out_sample, current, cfg, best, relaxing)
    if rejection:
        return rejection
    proposed.dir_enter_threshold = best
    action = "relaxed" if relaxing else "tightened"
    return f"entry_{action}_dir_threshold_scan"


def _tune_llm_influence(signals, proposed, cfg, breaker_active, online) -> list[str]:
    """Live-only: nudge LLM influence based on alignment-bucket realised edge."""
    reasons: list[str] = []
    if not online:
        return reasons
    emitted = [r for r in signals if str(r.get("decision")) == "emitted"]
    conflict = [
        (_exit_replay(r, proposed, cfg), _float(r.get("_weight"), 1.0) or 1.0)
        for r in emitted
        if str(r.get("llm_alignment")) == "conflict"
    ]
    aligned = [
        (_exit_replay(r, proposed, cfg), _float(r.get("_weight"), 1.0) or 1.0)
        for r in emitted
        if str(r.get("llm_alignment")) == "aligned"
    ]
    conflict = [(v, w) for v, w in conflict if v is not None]
    aligned = [(v, w) for v, w in aligned if v is not None]
    if len(conflict) >= cfg.min_samples_rule and len(aligned) >= cfg.min_samples_rule:
        c_mean, c_se, _ = _weighted_stats(conflict)
        a_mean, a_se, _ = _weighted_stats(aligned)
        diff = a_mean - c_mean
        pooled = math.sqrt(c_se**2 + a_se**2)
        if pooled > 0 and diff / pooled >= cfg.z_tighten:
            # Aligned clearly outperforms conflict -> trust LLM more (tighten conflict gate).
            proposed.llm_conflict_confidence_mult = _clip(
                proposed.llm_conflict_confidence_mult - 0.05,
                *PARAM_BOUNDS["llm_conflict_confidence_mult"],
            )
            reasons.append("llm_aligned_outperforms_trust_more")
        elif pooled > 0 and -diff / pooled >= cfg.z_relax and not breaker_active:
            # Conflict trades outperform aligned -> reduce LLM veto power (relaxation).
            proposed.llm_conflict_confidence_mult = _clip(
                proposed.llm_conflict_confidence_mult + 0.05,
                *PARAM_BOUNDS["llm_conflict_confidence_mult"],
            )
            proposed.llm_neutral_size_mult = _clip(
                proposed.llm_neutral_size_mult + 0.05, *PARAM_BOUNDS["llm_neutral_size_mult"]
            )
            reasons.append("llm_conflict_outperforms_reduce_veto")
    return reasons


def _extreme_bucket_stats(signals, label, proposed, cfg) -> tuple[float, float, float]:
    """Return weighted replay statistics for one first-extreme label."""
    values = [
        (_exit_replay(row, proposed, cfg), _float(row.get("_weight"), 1.0) or 1.0)
        for row in signals
        if row.get("future_detail_first_extreme_3") == label
    ]
    return _weighted_stats([(value, weight) for value, weight in values if value is not None])


def _adjust_drawdown_weights(proposed, decrease_penalty: bool) -> str:
    """Apply one bounded drawdown-weight adjustment and return its reason."""
    direction = -1 if decrease_penalty else 1
    proposed.pre_drawdown_weight = _clip(
        proposed.pre_drawdown_weight + direction * 0.10, *PARAM_BOUNDS["pre_drawdown_weight"]
    )
    proposed.pre_drawdown_free_ratio = _clip(
        proposed.pre_drawdown_free_ratio - direction * 0.05,
        *PARAM_BOUNDS["pre_drawdown_free_ratio"],
    )
    if decrease_penalty:
        return "adverse_first_outperformed_reduce_pre_dd_penalty"
    return "favorable_first_outperformed_increase_pre_dd_penalty"


def _tune_drawdown_weights(signals, proposed, cfg, breaker_active) -> list[str]:
    adverse = _extreme_bucket_stats(signals, "adverse_first", proposed, cfg)
    favorable = _extreme_bucket_stats(signals, "favorable_first", proposed, cfg)
    if adverse[2] < cfg.min_samples_rule or favorable[2] < cfg.min_samples_rule:
        return []
    difference = adverse[0] - favorable[0]
    pooled_se = math.sqrt(adverse[1] ** 2 + favorable[1] ** 2)
    if pooled_se <= 0:
        return []
    significance = abs(difference) / pooled_se
    can_relax = (
        not breaker_active and significance >= cfg.z_relax and adverse[2] >= cfg.relax_min_samples
    )
    if difference > 0 and can_relax:
        return [_adjust_drawdown_weights(proposed, True)]
    if difference < 0 and significance >= cfg.z_tighten:
        return [_adjust_drawdown_weights(proposed, False)]
    return []


def _exit_param_candidate(emitted, proposed, attr, delta, *search_context):
    cfg = search_context[0]
    value = int(_clip(getattr(proposed, attr) + delta, *PARAM_BOUNDS[attr]))
    base_mean = search_context[1]
    if value == getattr(proposed, attr) or (delta > 0 and search_context[2]):
        return None
    candidate = AdaptiveParameters.from_dict(asdict(proposed))
    setattr(candidate, attr, value)
    mean, error, sample_size = _policy_objective(
        emitted, candidate, cfg, candidate.dir_enter_threshold
    )
    improvement = mean - base_mean
    minimum = cfg.relax_min_samples if delta > 0 else cfg.min_samples_rule
    significance = cfg.z_relax if delta > 0 else cfg.z_tighten
    if (
        sample_size >= minimum
        and improvement > 0
        and (error <= 0 or improvement >= significance * error)
    ):
        return value, mean
    return None


def _tune_exit_params(in_sample, proposed, cfg, breaker_active) -> list[str]:
    emitted = [row for row in in_sample if str(row.get("decision")) == "emitted"]
    base_mean, _, base_size = _policy_objective(
        emitted, proposed, cfg, proposed.dir_enter_threshold
    )
    if len(emitted) < cfg.min_samples_rule or base_size < cfg.min_samples_rule:
        return []
    reasons: list[str] = []
    for attr in ("max_hold_candles", "no_progress_age"):
        for delta in (-1, 1):
            accepted = _exit_param_candidate(
                emitted, proposed, attr, delta, cfg, base_mean, breaker_active
            )
            if accepted is not None:
                value, base_mean = accepted
                setattr(proposed, attr, value)
                reasons.append(f"exit_replay_{attr}_{'up' if delta > 0 else 'down'}")
                break
    return reasons


@dataclass(frozen=True)
class _PathCandidate:
    """Best grid candidate for one adaptive path constant."""

    value: float
    improvement: float
    error: float


def _best_path_candidate(emitted, proposed, cfg, attr: str, step: float) -> _PathCandidate:
    """Scan a bounded grid and return its best sufficiently sampled candidate."""
    current = getattr(proposed, attr)
    lower, upper = PARAM_BOUNDS[attr]
    best = _PathCandidate(current, 0.0, 0.0)
    for index in range(round((upper - lower) / step) + 1):
        value = round(lower + step * index, 6)
        if abs(value - current) < 1e-12:
            continue
        mean, error, sample_size = _paired_replay_delta(
            emitted, proposed, _set(proposed, attr, value), cfg
        )
        if sample_size >= cfg.min_samples_rule and mean > best.improvement + 1e-12:
            best = _PathCandidate(value, mean, error)
    return best


def _path_candidate_passes(emitted, oos_emitted, proposed, attr, candidate, *validation) -> bool:
    """Apply breaker, significance, and out-of-sample guards to a candidate."""
    cfg = validation[0]
    current = getattr(proposed, attr)
    if candidate.value == current or not validation[2] or not oos_emitted:
        return False
    relaxing = _is_relaxation(attr, current, candidate.value)
    if relaxing and validation[1]:
        return False
    minimum = cfg.relax_min_samples if relaxing else cfg.min_samples_rule
    significance = cfg.z_relax if relaxing else cfg.z_tighten
    _, _, sample_size = _paired_replay_delta(
        emitted, proposed, _set(proposed, attr, candidate.value), cfg
    )
    if sample_size < minimum or (
        candidate.error > 0 and candidate.improvement < significance * candidate.error
    ):
        return False
    oos_mean, _, oos_size = _paired_replay_delta(
        oos_emitted, proposed, _set(proposed, attr, candidate.value), cfg
    )
    required_oos = cfg.relax_min_samples if relaxing else cfg.min_oos_signals
    return oos_size >= required_oos and oos_mean >= -1e-12 and (not relaxing or oos_mean > 0)


def _tune_exit_path_constants(in_sample, out_sample, proposed, *analysis_context) -> list[str]:
    """Tune path-shape constants with paired replay and OOS validation."""
    cfg, breaker_active, oos_ready = analysis_context
    emitted = [row for row in in_sample if str(row.get("decision")) == "emitted"]
    oos_emitted = [row for row in out_sample if str(row.get("decision")) == "emitted"]
    if len(emitted) < cfg.min_samples_rule:
        return []
    reasons: list[str] = []
    for attr, step in (
        ("recovery_band_mult", 0.10),
        ("recovery_sl_floor_ratio", 0.05),
        ("peak_activation_ratio", 0.05),
        ("post_drawdown_band_mult", 0.10),
    ):
        candidate = _best_path_candidate(emitted, proposed, cfg, attr, step)
        if _path_candidate_passes(
            emitted, oos_emitted, proposed, attr, candidate, cfg, breaker_active, oos_ready
        ):
            setattr(proposed, attr, candidate.value)
            reasons.append(f"exit_path_replay_{attr}")
    return reasons


def _set(params: AdaptiveParameters, attr: str, value: float) -> AdaptiveParameters:
    candidate = AdaptiveParameters.from_dict(asdict(params))
    setattr(candidate, attr, value)
    return candidate


def _tune_reward_quality(signals, proposed, cfg, breaker_active) -> list[str]:
    reasons: list[str] = []
    reward_values = [
        (_float(r.get("candidate_reward_quality"), None), _float(r.get("_weight"), 1.0) or 1.0)
        for r in signals
    ]
    reward_values = [(v, w) for v, w in reward_values if v is not None]
    if len(reward_values) < cfg.min_samples_rule:
        return reasons
    total_w = sum(w for _, w in reward_values)
    floor_share = (
        sum(w for v, w in reward_values if v <= proposed.reward_quality_floor + 1e-9) / total_w
    )
    rejected = [r for r in signals if str(r.get("decision")) == "rejected"]
    missed = [
        r for r in rejected if (_exit_replay(r, proposed, cfg) or -1.0) > cfg.opportunity_threshold
    ]
    missed_rate = (len(missed) / len(rejected)) if rejected else 0.0
    if floor_share > 0.70 and missed_rate > 0.05 and not breaker_active:
        proposed.target_peak_tp_multiplier = _clip(
            proposed.target_peak_tp_multiplier - 0.05, *PARAM_BOUNDS["target_peak_tp_multiplier"]
        )
        reasons.append("reward_quality_floor_saturated_lower_target_peak")
    return reasons


_SampleWindow = namedtuple("_SampleWindow", "all in_sample out_sample oos_ready half_life")
_ReplayScope = namedtuple("_ReplayScope", "all in_sample out_sample uses_executed oos_ready")


def _sample_window(signals, cfg) -> _SampleWindow:
    times = [_parse_time(row.get("reference_time")) for row in signals]
    times = [value for value in times if value is not None]
    now = max(times) if times else None
    half_life = cfg.half_life_days()
    weighted: list[AttributionRecord] = []
    for row in signals:
        timestamp = _parse_time(row.get("reference_time"))
        too_old = (
            now is not None
            and timestamp is not None
            and (now - timestamp).total_seconds() / 86400.0 > cfg.lookback_days
        )
        if too_old:
            continue
        weighted.append(
            row.with_analysis(
                _weight=_decay_weight(row.reference_time, now, half_life),
                _time=timestamp,
            )
        )
    weighted.sort(key=lambda row: row.get("_time") or datetime.min.replace(tzinfo=UTC))
    split = int(len(weighted) * (1.0 - cfg.oos_fraction))
    out_sample = weighted[split:]
    return _SampleWindow(
        weighted, weighted[:split], out_sample, len(out_sample) >= cfg.min_oos_signals, half_life
    )


def _replay_scope(window: _SampleWindow, cfg: AdaptiveConfig) -> _ReplayScope:
    executed = [row for row in window.all if bool(row.get("_executed_trade"))]
    use_executed = bool(cfg.execution_aware and len(executed) >= cfg.min_executed_signals)
    if not use_executed:
        return _ReplayScope(
            window.all, window.in_sample, window.out_sample, False, window.oos_ready
        )
    in_sample = [row for row in window.in_sample if bool(row.get("_executed_trade"))]
    out_sample = [row for row in window.out_sample if bool(row.get("_executed_trade"))]
    return _ReplayScope(
        executed, in_sample, out_sample, True, len(out_sample) >= cfg.min_executed_signals
    )


def _tune_parameters(window, scope, current, *analysis_context):
    cfg, breaker_active, online = analysis_context
    proposed = AdaptiveParameters.from_dict(asdict(current))
    entry_reason = _tune_entry_threshold(
        window.in_sample,
        window.out_sample,
        proposed,
        current,
        cfg,
        breaker_active,
        window.oos_ready,
    )
    reasons = [entry_reason] if entry_reason else []
    reasons.extend(_tune_reward_quality(window.all, proposed, cfg, breaker_active))
    reasons.extend(_tune_drawdown_weights(scope.all, proposed, cfg, breaker_active))
    reasons.extend(_tune_exit_params(scope.in_sample, proposed, cfg, breaker_active))
    path_reasons = _tune_exit_path_constants(
        scope.in_sample, scope.out_sample, proposed, cfg, breaker_active, scope.oos_ready
    )
    reasons.extend(path_reasons)
    reasons.extend(_tune_llm_influence(scope.all, proposed, cfg, breaker_active, online))
    return proposed, reasons


def _oos_validate(window, scope, current, proposed, cfg):
    if not scope.oos_ready:
        return proposed, False
    samples = window.out_sample if cfg.execution_aware else scope.out_sample
    objective = _execution_policy_objective if cfg.execution_aware else _policy_objective
    current_mean, _, _ = objective(samples, current, cfg, current.dir_enter_threshold)
    new_mean, _, new_size = objective(samples, proposed, cfg, proposed.dir_enter_threshold)
    minimum = cfg.min_executed_signals if scope.uses_executed else cfg.min_oos_signals
    if new_size < minimum or new_mean + 1e-9 < current_mean:
        return AdaptiveParameters.from_dict(asdict(current)), True
    return proposed, False


def _analysis_metrics(window, scope, trades, current, proposed, *analysis_context):
    cfg, breaker_active, rolled_back = analysis_context
    emitted = sum(str(row.get("decision")) == "emitted" for row in window.all)
    rejected = sum(str(row.get("decision")) == "rejected" for row in window.all)
    executed = sum(bool(row.get("_executed_trade")) for row in window.all)
    simulated = ([], [])
    if cfg.execution_aware:
        simulated = (
            simulate_executed_signals(window.all, current, cfg, current.dir_enter_threshold),
            simulate_executed_signals(window.all, proposed, cfg, proposed.dir_enter_threshold),
        )
    return {
        "signals": len(window.all),
        "in_sample": len(window.in_sample),
        "out_sample": len(window.out_sample),
        "oos_ready": scope.oos_ready,
        "candidate_oos_ready": window.oos_ready,
        "oos_rolled_back": rolled_back,
        "emitted": emitted,
        "rejected": rejected,
        "trades": len(trades),
        "executed_signals": executed,
        "replay_scope": "executed" if scope.uses_executed else "candidate",
        "simulated_current_signals": len(simulated[0]),
        "simulated_proposed_signals": len(simulated[1]),
        "replay_signals": len(scope.all),
        "replay_in_sample": len(scope.in_sample),
        "replay_out_sample": len(scope.out_sample),
        "round_trip_cost": cfg.round_trip_cost,
        "half_life_days": window.half_life,
        "circuit_breaker_active": breaker_active,
    }


def analyze(
    signals, trades, current, config=None, online=False
) -> tuple[AdaptiveParameters, dict[str, Any], list[str]]:
    """Analyze attributed signals and return a bounded adaptive proposal."""
    cfg = config or AdaptiveConfig()
    window = _sample_window(signals, cfg)
    scope = _replay_scope(window, cfg)
    breaker_active, breaker_metrics = _circuit_breaker(trades, cfg)
    proposed, reasons = _tune_parameters(window, scope, current, cfg, breaker_active, online)
    proposed, rolled_back = _oos_validate(window, scope, current, proposed, cfg)
    if rolled_back:
        reasons = ["oos_validation_rollback"]
    metrics = _analysis_metrics(
        window, scope, trades, current, proposed, cfg, breaker_active, rolled_back
    )
    metrics.update(breaker_metrics)
    return proposed.bounded(), metrics, reasons


def propose_parameters(signals, trades, current, config=None):
    """Return an adaptive parameter proposal and its supporting report data."""
    return analyze(signals, trades, current, config)


def load_attribution_csv(path: str) -> list[AttributionRecord]:
    """Load only the current canonical attribution CSV schema."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [AttributionRecord.from_csv_row(row) for row in csv.DictReader(handle)]


def load_trades_from_backtest_zip(path: str) -> list[dict[str, Any]]:
    """Extract normalized trades from a Freqtrade backtest result archive."""
    if not path:
        return []
    with zipfile.ZipFile(path) as archive:
        result_names = [
            name
            for name in archive.namelist()
            if name.endswith(".json") and not name.endswith("_config.json")
        ]
        if not result_names:
            return []
        data = json.loads(archive.read(result_names[0]))
    strategy = next(iter(data.get("strategy", {}).values()), {})
    return [
        {
            "trade_id": trade.get("trade_id") or trade.get("id") or "",
            "pair": trade.get("pair"),
            "side": "short" if trade.get("is_short") else "long",
            "open_time": trade.get("open_date"),
            "close_time": trade.get("close_date"),
            "exit_reason": trade.get("exit_reason"),
            "profit_ratio": trade.get("profit_ratio"),
            "profit_abs": trade.get("profit_abs"),
            "entry_tag": trade.get("enter_tag"),
            "duration": trade.get("trade_duration"),
            "leverage": trade.get("leverage"),
            "stake_amount": trade.get("stake_amount"),
        }
        for trade in strategy.get("trades", [])
    ]


@dataclass(frozen=True)
class OfflineAnalysisOptions:
    """Optional state, persistence, and configuration inputs for offline analysis."""

    current_params: str | None = None
    database: FreqLLMDatabase | None = None
    apply_state: bool = False
    config_path: str = ""


def _load_offline_config(path: str) -> AdaptiveConfig:
    """Load an adaptive config, falling back only for expected file/data errors."""
    if not path:
        return AdaptiveConfig()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return AdaptiveConfig.from_config(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return AdaptiveConfig()


def _load_offline_params(path: str | None, cfg: AdaptiveConfig) -> AdaptiveParameters:
    """Load current parameters from a UTF-8 report, or use config defaults."""
    if not path:
        return cfg.defaults
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AdaptiveParameters.from_dict(data.get("params"))


def _write_offline_report(path: str, report: dict[str, Any]) -> None:
    """Create the report directory and write deterministic UTF-8 JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def run_offline_analysis(
    attribution: str,
    trades_zip: str,
    output: str,
    options: OfflineAnalysisOptions | None = None,
) -> None:
    """Analyze attribution and optional backtest trades, then write a report."""
    options = options or OfflineAnalysisOptions()
    trades = load_trades_from_backtest_zip(trades_zip) if trades_zip else []
    cfg = _load_offline_config(options.config_path)
    params = _load_offline_params(options.current_params, cfg)
    cfg.output_path = output
    signals = mark_executed_signals(load_attribution_csv(attribution), trades, cfg)
    proposed, metrics, reasons = analyze(signals, trades, params, cfg)
    _write_offline_report(
        output,
        {
            "updated_at": _utcnow(),
            "params": asdict(proposed),
            "metrics": metrics,
            "reasons": reasons,
            "source": {"attribution": attribution, "trades_zip": trades_zip},
        },
    )
    if options.apply_state:
        if options.database is None:
            raise ValueError("apply_state requires a shared FreqLLMDatabase")
        cfg.enabled = True
        cfg.mode = "apply"
        manager = AdaptiveFeedbackManager(options.database, cfg)
        manager.apply_parameters(proposed, metrics, reasons or ["offline_apply_state"])


def main() -> None:
    """Run the offline adaptive-analysis command-line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze FreqLLM attribution and propose adaptive parameter overlay."
    )
    parser.add_argument("--attribution", required=True, help="Path to entry_attribution_*.csv")
    parser.add_argument("--trades-zip", default="", help="Optional Freqtrade backtest result zip")
    parser.add_argument("--output", default="user_data/freqllm/adaptive/adaptive_report.json")
    parser.add_argument(
        "--current-params", default="", help="Optional existing adaptive report/state JSON"
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional strategy config json to load adaptive settings/costs.",
    )
    parser.add_argument(
        "--db-url", default="", help="Unified FreqLLM database URL used with --apply-state."
    )
    parser.add_argument(
        "--apply-state", action="store_true", help="Persist proposed params to adaptive_state."
    )
    args = parser.parse_args()
    database = None
    try:
        if args.apply_state:
            database = FreqLLMDatabase(args.db_url) if args.db_url else FreqLLMDatabase()
        run_offline_analysis(
            args.attribution,
            args.trades_zip,
            args.output,
            OfflineAnalysisOptions(
                current_params=args.current_params or None,
                database=database,
                apply_state=args.apply_state,
                config_path=args.config,
            ),
        )
    finally:
        if database is not None:
            database.close()


if __name__ == "__main__":
    main()
