"""Sklearn-style PyTorch estimators used by FreqLLM policy backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from freqtrade.exceptions import DependencyException


try:
    from freqtrade.freqai.torch.PyTorchMLPModel import PyTorchMLPModel
    from freqtrade.freqai.torch.PyTorchTransformerModel import PyTorchTransformerModel
except ImportError:
    PyTorchMLPModel = None
    PyTorchTransformerModel = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


_Architecture = Literal["mlp", "transformer"]
_Loss = Literal["classification", "point", "quantile"]
_COMMON_PARAMETERS = frozenset(
    {
        "batch_size",
        "device",
        "dropout_percent",
        "gradient_clip_norm",
        "hidden_dim",
        "learning_rate",
        "n_epochs",
        "n_layer",
        "random_state",
        "weight_decay",
    }
)
_TRANSFORMER_PARAMETERS = frozenset({"nhead", "time_window"})


def _require_torch() -> None:
    if any(
        dependency is None
        for dependency in (
            torch,
            nn,
            DataLoader,
            TensorDataset,
            PyTorchMLPModel,
            PyTorchTransformerModel,
        )
    ):
        raise DependencyException(
            "PyTorch FreqLLM backends require the optional FreqAI-RL PyTorch dependencies"
        )


def _finite_matrix(values: Any, name: str) -> npt.NDArray[np.float32]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _finite_vector(values: Any, name: str) -> npt.NDArray[np.float32]:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return vector


def _validated_weights(values: Any, size: int) -> npt.NDArray[np.float32]:
    if values is None:
        return np.ones(size, dtype=np.float32)
    weights = _finite_vector(values, "sample_weight")
    if weights.size != size:
        raise ValueError("sample_weight length must match the number of samples")
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("sample_weight must be non-negative with a positive sum")
    return weights


def _positive_int(parameters: Mapping[str, Any], key: str) -> int:
    value = parameters[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"PyTorch parameter {key} must be a positive integer")
    return value


def _nonnegative_int(parameters: Mapping[str, Any], key: str) -> int:
    value = parameters[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"PyTorch parameter {key} must be a non-negative integer")
    return value


def _bounded_float(
    parameters: Mapping[str, Any],
    key: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = parameters[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"PyTorch parameter {key} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"PyTorch parameter {key} is outside its allowed range")
    return result


def validate_pytorch_parameters(
    parameters: Mapping[str, Any],
    architecture: _Architecture,
) -> dict[str, Any]:
    """Validate the strict PyTorch estimator parameter schema."""
    architecture_parameters = _TRANSFORMER_PARAMETERS if architecture == "transformer" else set()
    allowed = _COMMON_PARAMETERS | architecture_parameters
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"Unknown pytorch_{architecture} parameter: {unknown[0]}")
    validated = dict(parameters)
    for key in ("batch_size", "hidden_dim", "n_epochs", "n_layer"):
        _positive_int(validated, key)
    _nonnegative_int(validated, "random_state")
    if architecture == "transformer":
        _positive_int(validated, "nhead")
        _positive_int(validated, "time_window")
    _bounded_float(validated, "learning_rate", np.finfo(float).eps)
    _bounded_float(validated, "weight_decay", 0.0)
    _bounded_float(validated, "dropout_percent", 0.0, 1.0)
    _bounded_float(validated, "gradient_clip_norm", np.finfo(float).eps)
    device = validated.get("device")
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("PyTorch parameter device must be auto, cpu, cuda, or mps")
    return validated


class _PyTorchEstimator:
    """Shared weighted training and causal inference implementation."""

    def __init__(
        self,
        architecture: _Architecture,
        loss: _Loss,
        parameters: Mapping[str, Any],
        alpha: float | None = None,
    ) -> None:
        _require_torch()
        self.architecture = architecture
        self.loss = loss
        self.parameters = validate_pytorch_parameters(parameters, architecture)
        self.alpha = alpha
        if loss == "quantile" and (alpha is None or not 0.0 < alpha < 1.0):
            raise ValueError("Quantile alpha must be strictly between zero and one")
        self.model: Any = None
        self.feature_count = 0

    def fit_weighted(
        self,
        features: npt.NDArray[np.float32],
        target: npt.NDArray[Any],
        weights: npt.NDArray[np.float32],
        output_dim: int,
    ) -> None:
        """Fit the configured network with weighted batches."""
        self._fit_model(features, target, weights, output_dim)

    def predict_raw(self, features: Any) -> npt.NDArray[np.float32]:
        """Return raw network outputs after validating fitted state and input."""
        return self._raw_predict(features)

    def _device(self) -> Any:
        requested = self.parameters["device"]
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
            return torch.device("cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise DependencyException(
                "CUDA was requested for the PyTorch backend but is unavailable"
            )
        if requested == "mps" and not (
            torch.backends.mps.is_available() and torch.backends.mps.is_built()
        ):
            raise DependencyException(
                "MPS was requested for the PyTorch backend but is unavailable"
            )
        return torch.device(requested)

    def _network(self, output_dim: int) -> Any:
        model_options = {
            "dropout_percent": self.parameters["dropout_percent"],
            "hidden_dim": self.parameters["hidden_dim"],
            "n_layer": self.parameters["n_layer"],
        }
        if self.architecture == "mlp":
            return PyTorchMLPModel(self.feature_count, output_dim, **model_options)
        nhead = self.parameters["nhead"]
        time_window = self.parameters["time_window"]
        projected_dim = self.feature_count - (self.feature_count % nhead)
        if projected_dim <= 0:
            raise ValueError("Transformer feature count must be at least nhead")
        if projected_dim % 2:
            raise ValueError("Transformer projected feature dimension must be even")
        if self.parameters["hidden_dim"] < 4:
            raise ValueError("Transformer hidden_dim must be at least four")
        if time_window > projected_dim:
            raise ValueError(
                "Transformer time_window cannot exceed its projected feature dimension"
            )
        return PyTorchTransformerModel(
            input_dim=self.feature_count,
            output_dim=output_dim,
            nhead=nhead,
            time_window=time_window,
            **model_options,
        )

    def _windows(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if self.architecture == "mlp":
            return features
        width = self.parameters["time_window"]
        padded = np.pad(features, ((width - 1, 0), (0, 0)), mode="edge")
        return np.stack([padded[index : index + width] for index in range(len(features))])

    def _per_sample_loss(self, prediction: Any, target: Any) -> Any:
        if self.loss == "classification":
            return nn.functional.cross_entropy(prediction, target, reduction="none")
        prediction = prediction.reshape(-1)
        target = target.reshape(-1)
        error = target - prediction
        if self.loss == "quantile":
            return torch.maximum(self.alpha * error, (self.alpha - 1.0) * error)
        return error.square()

    def _training_loader(
        self,
        features: npt.NDArray[np.float32],
        target: npt.NDArray[Any],
        weights: npt.NDArray[np.float32],
        seed: int,
    ) -> Any:
        x_tensor = torch.as_tensor(self._windows(features), dtype=torch.float32)
        target_type = torch.long if self.loss == "classification" else torch.float32
        tensors = TensorDataset(
            x_tensor,
            torch.as_tensor(target, dtype=target_type),
            torch.as_tensor(weights, dtype=torch.float32),
        )
        return DataLoader(
            tensors,
            batch_size=self.parameters["batch_size"],
            generator=torch.Generator().manual_seed(seed),
            shuffle=True,
        )

    def _optimize(self, loader: Any, optimizer: Any, device: Any) -> None:
        self.model.train()
        for _ in range(self.parameters["n_epochs"]):
            for batch_features, batch_target, batch_weights in loader:
                prediction = self.model(batch_features.to(device))
                losses = self._per_sample_loss(prediction, batch_target.to(device))
                device_weights = batch_weights.to(device)
                loss = (losses * device_weights).sum() / device_weights.sum().clamp_min(1e-12)
                if not torch.isfinite(loss):
                    raise RuntimeError("PyTorch backend produced a non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.parameters["gradient_clip_norm"]
                )
                optimizer.step()

    def _fit_model(
        self,
        features: npt.NDArray[np.float32],
        target: npt.NDArray[Any],
        weights: npt.NDArray[np.float32],
        output_dim: int,
    ) -> None:
        self.feature_count = features.shape[1]
        seed = self.parameters["random_state"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = self._device()
        self.model = self._network(output_dim).to(device)
        loader = self._training_loader(features, target, weights, seed)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.parameters["learning_rate"],
            weight_decay=self.parameters["weight_decay"],
        )
        self._optimize(loader, optimizer, device)
        self.model = self.model.cpu().eval()

    def _raw_predict(self, features: Any) -> npt.NDArray[np.float32]:
        if self.model is None:
            raise RuntimeError("PyTorch estimator must be fitted before prediction")
        matrix = _finite_matrix(features, "features")
        if matrix.shape[1] != self.feature_count:
            raise ValueError("Prediction feature count does not match fitted data")
        tensor = torch.as_tensor(self._windows(matrix), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            result = self.model(tensor)
        values = np.asarray(result.detach().cpu().numpy(), dtype=np.float32)
        if self.architecture == "transformer":
            values = np.squeeze(values, axis=1)
        if not np.isfinite(values).all():
            raise RuntimeError("PyTorch backend produced non-finite predictions")
        return values


class PyTorchPolicyClassifier(_PyTorchEstimator):
    """Weighted classifier adapter exposing sklearn-compatible probabilities."""

    def __init__(self, architecture: _Architecture, parameters: Mapping[str, Any]) -> None:
        super().__init__(architecture, "classification", parameters)
        self.class_count = 0

    def fit(
        self,
        features: Any,
        labels: Any,
        sample_weight: npt.ArrayLike | None = None,
    ) -> PyTorchPolicyClassifier:
        """Fit a weighted neural classifier on contiguous integer labels."""
        matrix = _finite_matrix(features, "features")
        raw_labels = np.asarray(labels)
        if raw_labels.ndim != 1 or raw_labels.size != len(matrix):
            raise ValueError("Classifier labels must be a vector matching features")
        if not np.issubdtype(raw_labels.dtype, np.integer):
            raise TypeError("Classifier labels must be contiguous integers")
        encoded = raw_labels.astype(np.int64, copy=False)
        classes = np.unique(encoded)
        if not np.array_equal(classes, np.arange(classes.size)) or classes.size < 2:
            raise ValueError("Classifier labels must contain contiguous classes starting at zero")
        weights = _validated_weights(sample_weight, len(matrix))
        self.class_count = int(classes.size)
        self._fit_model(matrix, encoded, weights, self.class_count)
        return self

    def predict_proba(self, features: Any) -> npt.NDArray[np.float64]:
        """Return finite normalized class probabilities."""
        logits = self._raw_predict(features)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted.astype(np.float64))
        return exponentials / exponentials.sum(axis=1, keepdims=True)


class PyTorchPolicyRegressor(_PyTorchEstimator):
    """Weighted point or quantile regressor adapter."""

    def __init__(
        self,
        architecture: _Architecture,
        parameters: Mapping[str, Any],
        alpha: float | None = None,
    ) -> None:
        loss: _Loss = "quantile" if alpha is not None else "point"
        super().__init__(architecture, loss, parameters, alpha)

    def fit(
        self,
        features: Any,
        target: Any,
        sample_weight: npt.ArrayLike | None = None,
    ) -> PyTorchPolicyRegressor:
        """Fit weighted MSE or pinball regression."""
        matrix = _finite_matrix(features, "features")
        values = _finite_vector(target, "target")
        if values.size != len(matrix):
            raise ValueError("Regression target length must match features")
        weights = _validated_weights(sample_weight, len(matrix))
        self._fit_model(matrix, values, weights, 1)
        return self

    def predict(self, features: Any) -> npt.NDArray[np.float64]:
        """Return one finite prediction for each input row."""
        return np.asarray(self._raw_predict(features), dtype=np.float64).reshape(-1)
