"""Select a local reason classifier without loading transformer dependencies by default."""
import json
import math
from pathlib import Path
import threading

from demand_ml import REASON_LABELS, ReasonClassifier, TrainingError


class OfflineReason:
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


class SetFitReason:
    """Load only an explicitly selected, trusted, locally trained SetFit directory."""
    def __init__(self, config, *, enabled=True):
        self.path = Path(config["path"])
        self.enabled = enabled
        self.labels = config["labels"]
        if (not isinstance(self.labels, list) or len(self.labels) != len(REASON_LABELS)
                or set(self.labels) != set(REASON_LABELS)):
            raise ValueError("SetFit configuration must contain exactly the six reason labels.")
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
        # A local SetFit head is a pickle: accept only your own trusted exports.
        # Inference never downloads weights, trains, or calls an external API.
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
            return dict(reason=self.labels[index], confidence=round(float(scores[0, index]), 4),
                        used=bool(scores[0, index] >= self.threshold), model_version=self.version)
        except Exception as exc:
            # Preserve the app's rule fallback if model inference fails. Never
            # expose exception messages that might include customer text.
            self._error = "SetFit inference unavailable (" + type(exc).__name__ + ")"
            return None

    def train(self, reviewed):
        raise TrainingError("SetFit is trained offline; run the notebook or ml_workflow.train_setfit, evaluate, then restart with the new export.")


def load_reason_model(reason_config, *, enabled=True):
    path = Path(reason_config).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("path"), str):
        raise ValueError("Invalid reason-model serving configuration.")
    model_path = Path(config["path"])
    config["path"] = str(model_path if model_path.is_absolute() else (path.parent / model_path).resolve())
    if config.get("kind") == "setfit":
        return SetFitReason(config, enabled=enabled)
    if config.get("kind") == "tfidf":
        model = ReasonClassifier(config["path"], enabled=enabled, threshold=float(config["threshold"]))
        if enabled and not model.status()["ready"]:
            raise ValueError("Reason artifact is missing or invalid.")
        return OfflineReason(model)
    raise ValueError("Unknown reason model kind. Use tfidf or setfit.")


def configured_reason_model(default_path, *, config_path=None, enabled=True, threshold=.55):
    if config_path:
        return load_reason_model(config_path, enabled=enabled)
    return ReasonClassifier(default_path, enabled=enabled, threshold=threshold)
