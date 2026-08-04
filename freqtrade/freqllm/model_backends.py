"""Strict, allow-listed estimator backends for the FreqLLM policy model."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from freqtrade.exceptions import DependencyException
from freqtrade.freqllm.pytorch_estimators import (
    PyTorchPolicyClassifier,
    PyTorchPolicyRegressor,
    validate_pytorch_parameters,
)


try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
except ImportError:
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None


_BACKEND_NAMES = frozenset({"lightgbm", "pytorch_mlp", "pytorch_transformer", "sklearn", "xgboost"})
_TRAINING_OPTIONS = frozenset(
    {
        "backend",
        "backends",
        "key_level_weight_alpha",
        "key_level_weight_cap",
        "key_level_weight_scale_atr",
        "quantile_calibration_fraction",
    }
)
_BACKEND_SECTIONS = frozenset({"classifier", "quantile", "regression"})


class EncodedClassifier:
    """Expose original classes while fitting estimators on contiguous integer labels."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.classes_: npt.NDArray[Any] = np.asarray([], dtype=float)

    def fit(
        self,
        features: Any,
        labels: Any,
        sample_weight: npt.ArrayLike | None = None,
    ) -> EncodedClassifier:
        """Fit the estimator after encoding arbitrary numeric classes."""
        values = np.asarray(labels)
        self.classes_, encoded = np.unique(values, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("EncodedClassifier requires at least two target classes")
        self.estimator.fit(features, encoded, sample_weight=sample_weight)
        return self

    def predict_proba(self, features: Any) -> npt.NDArray[np.float64]:
        """Return probabilities ordered by the preserved original classes."""
        probabilities = np.asarray(self.estimator.predict_proba(features), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.classes_.size:
            raise ValueError("Classifier probability columns do not match fitted classes")
        return probabilities


class PolicyModelBackend(ABC):
    """Factory contract shared by every supported estimator library."""

    name: str
    classifier_defaults: Mapping[str, Any]
    regression_defaults: Mapping[str, Any]

    def validate_parameters(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate backend parameters before training starts."""
        return dict(parameters)

    @abstractmethod
    def create_classifier(self, parameters: Mapping[str, Any]) -> EncodedClassifier:
        """Create an unfitted classifier with original-label preservation."""

    @abstractmethod
    def create_regressor(self, parameters: Mapping[str, Any]) -> Any:
        """Create an unfitted point regressor."""

    @abstractmethod
    def create_quantile_regressor(
        self,
        parameters: Mapping[str, Any],
        alpha: float,
    ) -> Any:
        """Create an unfitted quantile regressor."""


class LightGBMPolicyBackend(PolicyModelBackend):
    """LightGBM estimator factories."""

    name = "lightgbm"
    classifier_defaults = {
        "colsample_bytree": 0.85,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "num_leaves": 31,
        "random_state": 7,
        "reg_lambda": 1.0,
        "subsample": 0.85,
        "verbosity": -1,
    }
    regression_defaults = classifier_defaults

    @staticmethod
    def _classes() -> tuple[type[Any], type[Any]]:
        if LGBMClassifier is None or LGBMRegressor is None:
            raise DependencyException(
                "The lightgbm backend requires the optional FreqAI LightGBM dependency"
            )
        return LGBMClassifier, LGBMRegressor

    def create_classifier(self, parameters: Mapping[str, Any]) -> EncodedClassifier:
        classifier, _ = self._classes()
        return EncodedClassifier(classifier(**parameters))

    def create_regressor(self, parameters: Mapping[str, Any]) -> Any:
        _, regressor = self._classes()
        return regressor(**parameters)

    def create_quantile_regressor(
        self,
        parameters: Mapping[str, Any],
        alpha: float,
    ) -> Any:
        _, regressor = self._classes()
        options = dict(parameters)
        options.update({"alpha": alpha, "objective": "quantile"})
        return regressor(**options)


class XGBoostPolicyBackend(PolicyModelBackend):
    """XGBoost estimator factories."""

    name = "xgboost"
    classifier_defaults = {
        "colsample_bytree": 0.85,
        "learning_rate": 0.03,
        "max_depth": 6,
        "n_estimators": 300,
        "random_state": 7,
        "reg_lambda": 1.0,
        "subsample": 0.85,
        "tree_method": "hist",
        "verbosity": 0,
    }
    regression_defaults = classifier_defaults

    @staticmethod
    def _classes() -> tuple[type[Any], type[Any]]:
        if XGBClassifier is None or XGBRegressor is None:
            raise DependencyException(
                "The xgboost backend requires the optional FreqAI XGBoost dependency"
            )
        return XGBClassifier, XGBRegressor

    def create_classifier(self, parameters: Mapping[str, Any]) -> EncodedClassifier:
        classifier, _ = self._classes()
        return EncodedClassifier(classifier(**parameters))

    def create_regressor(self, parameters: Mapping[str, Any]) -> Any:
        _, regressor = self._classes()
        return regressor(**parameters)

    def create_quantile_regressor(
        self,
        parameters: Mapping[str, Any],
        alpha: float,
    ) -> Any:
        _, regressor = self._classes()
        options = dict(parameters)
        options.update({"objective": "reg:quantileerror", "quantile_alpha": alpha})
        return regressor(**options)


class PyTorchPolicyBackend(PolicyModelBackend):
    """Weighted PyTorch MLP or Transformer estimator factories."""

    classifier_defaults = {
        "batch_size": 64,
        "device": "auto",
        "dropout_percent": 0.2,
        "gradient_clip_norm": 5.0,
        "hidden_dim": 256,
        "learning_rate": 0.0003,
        "n_epochs": 10,
        "n_layer": 2,
        "random_state": 7,
        "weight_decay": 0.0001,
    }
    regression_defaults = classifier_defaults

    def __init__(self, architecture: str) -> None:
        if architecture not in {"mlp", "transformer"}:
            raise ValueError("PyTorch policy architecture must be mlp or transformer")
        self.architecture = architecture
        self.name = f"pytorch_{architecture}"
        if architecture == "transformer":
            defaults = {**self.classifier_defaults, "nhead": 4, "time_window": 16}
            self.classifier_defaults = defaults
            self.regression_defaults = defaults

    def validate_parameters(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Reject unknown or invalid neural-network parameters eagerly."""
        return validate_pytorch_parameters(parameters, self.architecture)

    def create_classifier(self, parameters: Mapping[str, Any]) -> EncodedClassifier:
        estimator = PyTorchPolicyClassifier(self.architecture, self.validate_parameters(parameters))
        return EncodedClassifier(estimator)

    def create_regressor(self, parameters: Mapping[str, Any]) -> Any:
        return PyTorchPolicyRegressor(self.architecture, self.validate_parameters(parameters))

    def create_quantile_regressor(
        self,
        parameters: Mapping[str, Any],
        alpha: float,
    ) -> Any:
        return PyTorchPolicyRegressor(
            self.architecture,
            self.validate_parameters(parameters),
            alpha,
        )


class SklearnPolicyBackend(PolicyModelBackend):
    """Scikit-learn histogram gradient-boosting estimator factories."""

    name = "sklearn"
    classifier_defaults = {
        "l2_regularization": 1.0,
        "learning_rate": 0.03,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "random_state": 7,
        "verbose": 0,
    }
    regression_defaults = classifier_defaults

    @staticmethod
    def _classes() -> tuple[type[Any], type[Any]]:
        if HistGradientBoostingClassifier is None or HistGradientBoostingRegressor is None:
            raise DependencyException(
                "The sklearn backend requires the optional FreqAI scikit-learn dependency"
            )
        return HistGradientBoostingClassifier, HistGradientBoostingRegressor

    def create_classifier(self, parameters: Mapping[str, Any]) -> EncodedClassifier:
        classifier, _ = self._classes()
        return EncodedClassifier(classifier(**parameters))

    def create_regressor(self, parameters: Mapping[str, Any]) -> Any:
        _, regressor = self._classes()
        options = dict(parameters)
        options["loss"] = "squared_error"
        return regressor(**options)

    def create_quantile_regressor(
        self,
        parameters: Mapping[str, Any],
        alpha: float,
    ) -> Any:
        _, regressor = self._classes()
        options = dict(parameters)
        options.update({"loss": "quantile", "quantile": alpha})
        return regressor(**options)


_BACKENDS: Mapping[str, PolicyModelBackend] = {
    "lightgbm": LightGBMPolicyBackend(),
    "pytorch_mlp": PyTorchPolicyBackend("mlp"),
    "pytorch_transformer": PyTorchPolicyBackend("transformer"),
    "sklearn": SklearnPolicyBackend(),
    "xgboost": XGBoostPolicyBackend(),
}


@dataclass(frozen=True)
class PolicyBackendSelection:
    """Validated selected backend and its isolated estimator parameters."""

    backend: PolicyModelBackend
    classifier_parameters: dict[str, Any]
    regression_parameters: dict[str, Any]
    quantile_parameters: dict[str, Any]

    @property
    def name(self) -> str:
        """Return the canonical selected backend name."""
        return self.backend.name

    def create_classifier(self) -> EncodedClassifier:
        """Create a classifier using the selected backend parameters."""
        return self.backend.create_classifier(self.classifier_parameters)

    def create_regressor(self) -> Any:
        """Create a point regressor using the selected backend parameters."""
        return self.backend.create_regressor(self.regression_parameters)

    def create_quantile_regressor(self, alpha: float) -> Any:
        """Create a quantile regressor using merged regression parameters."""
        parameters = {**self.regression_parameters, **self.quantile_parameters}
        return self.backend.create_quantile_regressor(parameters, alpha)

    @classmethod
    def from_training_parameters(
        cls,
        training_parameters: Mapping[str, Any] | None,
    ) -> PolicyBackendSelection:
        """Validate model configuration and resolve its allow-listed backend."""
        raw = _validated_mapping(training_parameters or {}, "model_training_parameters")
        unknown_options = sorted(set(raw) - _TRAINING_OPTIONS)
        if unknown_options:
            raise ValueError(
                "Unknown FreqLLM model training option: "
                f"freqai.model_training_parameters.{unknown_options[0]}"
            )
        backend_name = str(raw.get("backend", "lightgbm")).strip().lower()
        if backend_name not in _BACKENDS:
            supported = ", ".join(sorted(_BACKEND_NAMES))
            raise ValueError(
                f"Unsupported FreqLLM model backend {backend_name!r}; choose {supported}"
            )
        configured_backends = _validated_mapping(raw.get("backends", {}), "backends")
        unknown_backends = sorted(set(configured_backends) - _BACKEND_NAMES)
        if unknown_backends:
            raise ValueError(f"Unknown FreqLLM model backend configuration: {unknown_backends[0]}")
        selected = _validated_mapping(configured_backends.get(backend_name, {}), backend_name)
        unknown_sections = sorted(set(selected) - _BACKEND_SECTIONS)
        if unknown_sections:
            raise ValueError(
                f"Unknown {backend_name} backend configuration section: {unknown_sections[0]}"
            )
        backend = _BACKENDS[backend_name]
        classifier = {
            **backend.classifier_defaults,
            **_validated_mapping(selected.get("classifier", {}), f"{backend_name}.classifier"),
        }
        regression = {
            **backend.regression_defaults,
            **_validated_mapping(selected.get("regression", {}), f"{backend_name}.regression"),
        }
        quantile = _validated_mapping(
            selected.get("quantile", {}),
            f"{backend_name}.quantile",
        )
        classifier = backend.validate_parameters(classifier)
        regression = backend.validate_parameters(regression)
        backend.validate_parameters({**regression, **quantile})
        return cls(backend, classifier, regression, quantile)


def _validated_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"freqai.model_training_parameters.{path} must be an object")
    result = dict(value)
    if any(not isinstance(key, str) or not key.strip() for key in result):
        raise ValueError(f"freqai.model_training_parameters.{path} contains an invalid key")
    for key, item in result.items():
        _validate_config_value(item, f"{path}.{key}")
    return result


def _validate_config_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"freqai.model_training_parameters.{path} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_config_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        _validated_mapping(value, path)
        return
    raise TypeError(
        f"freqai.model_training_parameters.{path} must contain only JSON-compatible values"
    )
