"""Tests for the allow-listed FreqLLM estimator backend configuration."""

from unittest import TestCase

import numpy as np

from freqtrade.freqllm.model_backends import EncodedClassifier, PolicyBackendSelection
from freqtrade.freqllm.pytorch_estimators import validate_pytorch_parameters


class _ProbabilityEstimator:
    """Minimal estimator used to verify backend-neutral class encoding."""

    def __init__(self) -> None:
        self.labels = np.asarray([])
        self.sample_weight = np.asarray([])

    def fit(self, features, labels, sample_weight=None):
        """Capture encoded labels and weights like a fitted estimator."""
        del features
        self.labels = np.asarray(labels)
        self.sample_weight = np.asarray(sample_weight)
        return self

    def predict_proba(self, features):
        """Return deterministic three-class probabilities."""
        return np.tile(np.asarray([[0.2, 0.3, 0.5]]), (len(features), 1))


def test_backend_defaults_to_lightgbm() -> None:
    """An omitted backend selects the documented LightGBM defaults."""
    selection = PolicyBackendSelection.from_training_parameters({})

    assert selection.name == "lightgbm"
    assert selection.classifier_parameters["num_leaves"] == 31
    assert selection.regression_parameters["n_estimators"] == 300


def test_backend_parameters_are_isolated() -> None:
    """Only parameters belonging to the selected backend are merged."""
    selection = PolicyBackendSelection.from_training_parameters(
        {
            "backend": "xgboost",
            "backends": {
                "lightgbm": {"classifier": {"num_leaves": 63}},
                "xgboost": {
                    "classifier": {"max_depth": 4},
                    "regression": {"max_depth": 5},
                    "quantile": {"max_bin": 128},
                },
            },
        }
    )

    assert selection.name == "xgboost"
    assert selection.classifier_parameters["max_depth"] == 4
    assert selection.regression_parameters["max_depth"] == 5
    assert selection.quantile_parameters == {"max_bin": 128}
    assert "num_leaves" not in selection.classifier_parameters


def test_backend_rejects_unknown_and_removed_configuration() -> None:
    """Unknown backends, sections and removed top-level fields fail closed."""
    assertions = TestCase()
    with assertions.assertRaisesRegex(ValueError, "Unsupported FreqLLM model backend"):
        PolicyBackendSelection.from_training_parameters({"backend": "arbitrary.module.Model"})
    with assertions.assertRaisesRegex(ValueError, "classifier_parameters"):
        PolicyBackendSelection.from_training_parameters({"classifier_parameters": {}})
    with assertions.assertRaisesRegex(ValueError, "unknown"):
        PolicyBackendSelection.from_training_parameters({"backends": {"lightgbm": {"unknown": {}}}})


def test_backend_rejects_non_data_parameter_values() -> None:
    """Estimator parameters reject Python objects and accept data only."""
    with TestCase().assertRaisesRegex(TypeError, "JSON-compatible"):
        PolicyBackendSelection.from_training_parameters(
            {"backends": {"lightgbm": {"classifier": {"callback": object()}}}}
        )


def test_pytorch_backends_are_allow_listed_and_isolated() -> None:
    """Both PyTorch architectures resolve without importing arbitrary classes."""
    mlp = PolicyBackendSelection.from_training_parameters(
        {
            "backend": "pytorch_mlp",
            "backends": {"pytorch_mlp": {"classifier": {"hidden_dim": 128}}},
        }
    )
    transformer = PolicyBackendSelection.from_training_parameters(
        {
            "backend": "pytorch_transformer",
            "backends": {"pytorch_transformer": {"regression": {"nhead": 2, "time_window": 8}}},
        }
    )

    assert mlp.name == "pytorch_mlp"
    assert mlp.classifier_parameters["hidden_dim"] == 128
    assert "time_window" not in mlp.classifier_parameters
    assert transformer.name == "pytorch_transformer"
    assert transformer.regression_parameters["nhead"] == 2
    assert transformer.regression_parameters["time_window"] == 8


def test_pytorch_parameters_fail_closed() -> None:
    """Unsafe devices, invalid ranges and unknown estimator options are rejected."""
    defaults = PolicyBackendSelection.from_training_parameters(
        {"backend": "pytorch_mlp"}
    ).classifier_parameters
    assertions = TestCase()
    with assertions.assertRaisesRegex(ValueError, "device"):
        validate_pytorch_parameters({**defaults, "device": "remote"}, "mlp")
    with assertions.assertRaisesRegex(ValueError, "dropout_percent"):
        validate_pytorch_parameters({**defaults, "dropout_percent": 1.5}, "mlp")
    with assertions.assertRaisesRegex(ValueError, "Unknown pytorch_mlp parameter"):
        PolicyBackendSelection.from_training_parameters(
            {
                "backend": "pytorch_mlp",
                "backends": {"pytorch_mlp": {"classifier": {"module_path": "unsafe.Model"}}},
            }
        )


def test_encoded_classifier_preserves_original_classes() -> None:
    """Contiguous training labels retain their original inference ordering."""
    estimator = _ProbabilityEstimator()
    classifier = EncodedClassifier(estimator)
    features = np.asarray([[1.0], [2.0], [3.0]])

    classifier.fit(features, np.asarray([-1.0, 1.0, 0.0]), sample_weight=np.ones(3))

    assert classifier.classes_.tolist() == [-1.0, 0.0, 1.0]
    assert estimator.labels.tolist() == [0, 2, 1]
    assert classifier.predict_proba(features).shape == (3, 3)
