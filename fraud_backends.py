"""Select a local scam classifier without loading transformer dependencies by default."""
import json
import math
from pathlib import Path
import threading

from fraud_ml import SCAM_LABELS, ScamClassifier, TrainingError


class OfflineScam:
    """An evaluated candidate is retrained offline, never from a dashboard request."""
    def __init__(self, model):
        self.model = model
        self.path = model.path
        self.enabled = model.enabled

    def status(self):
        return self.model.status()

    def predict(self, text):
        return self.model.predict(text)

    def train(self, reviewed):
        raise TrainingError("This model is trained offline with frozen splits. Evaluate a new candidate and restart with its serving configuration.")


class SetFitScam:
    """Load only an explicitly selected, trusted, locally trained SetFit directory."""
    def __init__(self, config, *, enabled=True):
        self.path = Path(config["path"])
        self.enabled = enabled
        self.labels = config["labels"]
        if (not isinstance(self.labels, list) or len(self.labels) != len(SCAM_LABELS)
                or set(self.labels) != set(SCAM_LABELS)):
            raise ValueError("SetFit configuration must contain exactly the scam labels.")
        self.threshold = float(config["threshold"])
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("SetFit score threshold must be between zero and one.")
        self.version = config["version"]
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("SetFit needs a nonempty model version.")
        self.metadata = {**config.get("metadata", {}), "version": self.version}
        self._lock = threading.Lock()
        self._error = None
        self.model = None
        if not enabled:
            return
        if not (self.path / "config_setfit.json").is_file() or not (self.path / "model_head.pkl").is_file():
            raise ValueError("SetFit model files are missing. Train/export a candidate first.")
        try:
            import numpy as np
            from setfit import SetFitModel
        except ImportError as exc:
            raise TrainingError("SetFit backend requires Python 3.11/3.12 and requirements-setfit.txt in a separate environment.") from exc
        self.np = np
        self.model = SetFitModel.from_pretrained(str(self.path), local_files_only=True)
        self.model.to("cpu")
        if list(self.model.labels or []) != self.labels:
            raise ValueError("Saved SetFit labels do not match the serving configuration.")
        if list(self.model.model_head.classes_) != list(range(len(self.labels))):
            raise ValueError("Saved SetFit class indices do not match the label mapping.")

    def status(self):
        return dict(ready=self.model is not None and self._error is None,
                    enabled=self.enabled, threshold=self.threshold, error=self._error,
                    algorithm="SetFit + MiniLM", metadata=dict(self.metadata))

    def predict(self, text):
        if not self.enabled or self.model is None or not isinstance(text, str) or not text.strip():
            return None
        try:
            with self._lock:
                scores = self.np.asarray(self.model.predict_proba([text], as_numpy=True), dtype=float)
            if (scores.shape != (1, len(self.labels)) or not self.np.isfinite(scores).all()
                    or (scores < 0).any() or (scores > 1).any()
                    or not self.np.allclose(scores.sum(axis=1), 1, atol=1e-5)):
                raise ValueError("Invalid class scores")
            index = int(scores[0].argmax())
            self._error = None
            confidence = float(scores[0, index])
            if confidence < self.threshold:
                return None
            return dict(verdict=self.labels[index], confidence=round(confidence, 4), model_version=self.version)
        except Exception as exc:
            self._error = "SetFit inference unavailable (" + type(exc).__name__ + ")"
            return None

    def train(self, reviewed):
        raise TrainingError("SetFit is trained offline; run the notebook or ml_workflow.train_fraud_setfit, evaluate, then restart with the new export.")


def load_fraud_model(fraud_config, *, enabled=True):
    path = Path(fraud_config).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("path"), str):
        raise ValueError("Invalid fraud-model serving configuration.")
    model_path = Path(config["path"])
    config["path"] = str(model_path if model_path.is_absolute() else (path.parent / model_path).resolve())
    if config.get("kind") == "setfit":
        return SetFitScam(config, enabled=enabled)
    if config.get("kind") == "tfidf":
        model = ScamClassifier(config["path"], enabled=enabled, threshold=float(config["threshold"]))
        if enabled and not model.status()["ready"]:
            raise ValueError("Fraud artifact is missing or invalid.")
        return OfflineScam(model)
    raise ValueError("Unknown fraud model kind. Use tfidf or setfit.")


def configured_fraud_model(default_path, *, config_path=None, enabled=True, threshold=0.6):
    if config_path:
        return load_fraud_model(config_path, enabled=enabled)
    return ScamClassifier(default_path, enabled=enabled, threshold=threshold)
