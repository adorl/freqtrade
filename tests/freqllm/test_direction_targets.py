"""Golden tests for policy-derived direction labels."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.freqllm.adaptive import AdaptiveConfig, AdaptiveFeedbackManager
from freqtrade.freqllm.attribution import SimpleAttributionWriter
from freqtrade.freqllm.decision import SimpleDecisionEngine, SimpleStrategyConfig
from freqtrade.freqllm.persistence import FreqLLMDatabase
from freqtrade.freqllm.strategy.contracts import PolicyTargets, StrategyCollaborators
from freqtrade.freqllm.strategy.targets import StrategyTargetsMixin


@contextmanager
def _target_builder(tmp_path: Path) -> Iterator[StrategyTargetsMixin]:
    """Build a target mixin with its explicit typed collaborators."""
    config = SimpleStrategyConfig()
    database = FreqLLMDatabase("sqlite:///:memory:")
    collaborators = StrategyCollaborators(
        simple_config=config,
        decision_engine=SimpleDecisionEngine(config),
        adaptive_manager=AdaptiveFeedbackManager(database, AdaptiveConfig(enabled=False)),
        attribution_writer=SimpleAttributionWriter(str(tmp_path)),
    )
    builder = StrategyTargetsMixin()
    builder.strategy_collaborators = collaborators
    try:
        yield builder
    finally:
        database.close()


def test_direction_target_golden_boundaries_and_missing_values(tmp_path: Path) -> None:
    """Direction labels preserve strict margin, inclusive gap, and missing-value semantics."""
    index = pd.Index(
        [
            "long",
            "short",
            "exact_gap",
            "below_gap",
            "exact_margin",
            "flat",
            "missing",
            "infinite",
        ]
    )
    policies = PolicyTargets(
        long=pd.DataFrame(
            {
                "policy_return": [
                    0.006,
                    -0.004,
                    0.004,
                    0.004,
                    0.001,
                    -0.001,
                    np.nan,
                    np.inf,
                ]
            },
            index=index,
        ),
        short=pd.DataFrame(
            {
                "policy_return": [
                    0.001,
                    0.006,
                    0.002,
                    0.0021,
                    -0.002,
                    -0.003,
                    0.001,
                    0.001,
                ]
            },
            index=index,
        ),
        long_early_fail=pd.Series(0.0, index=index),
        short_early_fail=pd.Series(0.0, index=index),
    )
    with _target_builder(tmp_path) as target_builder:
        result = vars(StrategyTargetsMixin)["_direction_target"](
            target_builder,
            index,
            policies,
            {"label_class_margin": 0.001, "label_class_edge_gap": 0.002},
        )

    expected = pd.Series(
        [1.0, -1.0, 1.0, 0.0, 0.0, 0.0, np.nan, np.nan],
        index=index,
    )
    pd.testing.assert_series_equal(result, expected)


def test_direction_target_clamps_negative_thresholds_to_zero(tmp_path: Path) -> None:
    """Negative user thresholds cannot turn a losing policy into a directional label."""
    index = pd.RangeIndex(3)
    policies = PolicyTargets(
        long=pd.DataFrame({"policy_return": [-0.001, 0.001, 0.0]}, index=index),
        short=pd.DataFrame({"policy_return": [-0.002, -0.001, 0.001]}, index=index),
        long_early_fail=pd.Series(0.0, index=index),
        short_early_fail=pd.Series(0.0, index=index),
    )
    with _target_builder(tmp_path) as target_builder:
        result = vars(StrategyTargetsMixin)["_direction_target"](
            target_builder,
            index,
            policies,
            {"label_class_margin": -1.0, "label_class_edge_gap": -1.0},
        )

    expected = pd.Series([0.0, 1.0, -1.0], index=index)
    pd.testing.assert_series_equal(result, expected)
