"""Train and serve a local TF-IDF + logistic regression reason classifier.

Artifacts contain JSON vocabulary and numeric weights, never executable pickle.
Only explicitly reviewed labels (plus the synthetic starter set) enter training.
"""
import json
import math
import os
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REASON_LABELS = (
    "out_of_stock", "price_too_high", "not_stocked",
    "quote_rejected", "lead_time", "abandoned",
)
STARTER_PATH = Path(__file__).with_name("training_data.json")
VECTORIZER_PARAMS = dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=6000)


class TrainingError(ValueError):
    pass


class ReasonClassifier:
    def __init__(self, path, *, enabled=True, threshold=0.55):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("GHOST_ML_THRESHOLD must be between 0 and 1")
        self.path = Path(path)
        self.enabled = enabled
        self.threshold = threshold
        self._stamp = None
        self._model = None
        self._error = None
        self._lock = threading.RLock()
        self._training_lock = threading.Lock()

    @staticmethod
    def _libraries():
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise TrainingError("Install requirements.txt to enable local machine learning.") from exc
        return np, TfidfVectorizer, LogisticRegression, accuracy_score, f1_score, train_test_split

    def _decode(self, artifact):
        np, Vectorizer, *_ = self._libraries()
        if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
            raise ValueError("Unsupported model format; retrain the model.")
        expected_params = {**VECTORIZER_PARAMS, "ngram_range": list(VECTORIZER_PARAMS["ngram_range"])}
        if artifact.get("vectorizer_params") != expected_params:
            raise ValueError("Unsupported text features; retrain the model.")
        vocabulary = artifact["vocabulary"]
        classes = artifact["classes"]
        if not isinstance(vocabulary, dict) or not 1 <= len(vocabulary) <= 6000:
            raise ValueError("Invalid model vocabulary")
        if (not all(isinstance(k, str) and type(v) is int for k, v in vocabulary.items())
                or set(vocabulary.values()) != set(range(len(vocabulary)))):
            raise ValueError("Invalid vocabulary indices")
        if not isinstance(classes, list) or sorted(classes) != sorted(REASON_LABELS):
            raise ValueError("Invalid model labels")
        idf = np.asarray(artifact["idf"], dtype=float)
        weights = np.asarray(artifact["weights"], dtype=float)
        intercept = np.asarray(artifact["intercept"], dtype=float)
        if (idf.shape != (len(vocabulary),) or weights.shape != (6, len(vocabulary))
                or intercept.shape != (6,) or not np.all(idf > 0)
                or not all(np.isfinite(a).all() for a in (idf, weights, intercept))):
            raise ValueError("Invalid model weights")
        metadata = artifact["metadata"]
        if (not isinstance(metadata, dict) or not isinstance(metadata.get("version"), str)
                or not metadata["version"] or not isinstance(metadata.get("trained_at"), str)):
            raise ValueError("Invalid model metadata")
        datetime.fromisoformat(metadata["trained_at"])
        if not isinstance(metadata.get("reviewed_through", ""), str):
            raise ValueError("Invalid review timestamp")
        evaluation = metadata.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("Invalid evaluation metadata")
        for values in (metadata, evaluation):
            for key in ("examples", "starter_examples", "reviewed_examples"):
                if type(values.get(key)) is not int or values[key] < 0:
                    raise ValueError("Invalid training sample counts")
            if values["examples"] != values["starter_examples"] + values["reviewed_examples"]:
                raise ValueError("Training sample counts do not match")
        for key in ("accuracy", "macro_f1"):
            if type(evaluation.get(key)) not in (float, int) or not 0 <= evaluation[key] <= 1:
                raise ValueError("Invalid evaluation scores")
        vectorizer = Vectorizer(vocabulary=vocabulary, **VECTORIZER_PARAMS)
        vectorizer.idf_ = idf
        return vectorizer, weights, intercept, classes, metadata

    def _refresh(self):
        """Reload after atomic replacement, including across app processes."""
        with self._lock:
            try:
                stat = self.path.stat()
                stamp = (stat.st_mtime_ns, stat.st_size)
                if stamp == self._stamp:
                    return
                artifact = json.loads(self.path.read_text(encoding="utf-8"))
                model = self._decode(artifact)
            except FileNotFoundError:
                self._model, self._stamp, self._error = None, None, None
                return
            except (OSError, ValueError, KeyError, TypeError, TrainingError) as exc:
                self._model, self._stamp = None, None
                self._error = f"Model unavailable: {exc}"
                return
            self._model, self._stamp, self._error = model, stamp, None

    def status(self):
        self._refresh()
        with self._lock:
            return dict(
                ready=self._model is not None, enabled=self.enabled,
                threshold=self.threshold, error=self._error,
                algorithm="TF-IDF + logistic regression",
                metadata=dict(self._model[4]) if self._model else None,
            )

    def predict(self, text):
        if not self.enabled:
            return None
        self._refresh()
        with self._lock:
            model = self._model
        if model is None:
            return None
        vectorizer, weights, intercept, classes, metadata = model
        np, *_ = self._libraries()
        features = vectorizer.transform([text])
        # Unknown vocabulary must not turn a class prior into a prediction.
        if features.nnz == 0:
            return None
        logits = np.asarray(features @ weights.T).ravel() + intercept
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        return dict(reason=classes[index], confidence=round(confidence, 4),
                    used=confidence >= self.threshold, model_version=metadata["version"])

    def train(self, reviewed):
        if not self._training_lock.acquire(blocking=False):
            raise TrainingError("Training is already in progress. Try again shortly.")
        try:
            return self._train(reviewed)
        finally:
            self._training_lock.release()

    def _train(self, reviewed):
        np, Vectorizer, Classifier, accuracy, f1, split = self._libraries()
        try:
            starters = json.loads(STARTER_PATH.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise TrainingError("training_data.json must contain a JSON list of labeled examples.") from exc
        if not isinstance(starters, list):
            raise TrainingError("training_data.json must contain a JSON list of labeled examples.")
        examples = {}
        # Reviewed labels take precedence; repeated messages count once.
        for origin, items in (("starter", starters), ("reviewed", reviewed)):
            for item in items:
                if not isinstance(item, dict):
                    raise TrainingError("Each training example must be a JSON object.")
                text, reason = item.get("text"), item.get("reason")
                if not isinstance(text, str) or not text.strip() or reason not in REASON_LABELS:
                    raise TrainingError("Training examples need text and a supported reason label.")
                key = " ".join(text.casefold().split())
                examples[key] = dict(text=text.strip(), reason=reason, origin=origin)
        data = list(examples.values())
        counts = Counter(e["reason"] for e in data)
        if any(counts[label] < 4 for label in REASON_LABELS):
            raise TrainingError("Training needs at least four distinct examples for each reason.")
        indices = list(range(len(data)))
        train_ids, test_ids = split(indices, test_size=0.25, random_state=42,
                                    stratify=[e["reason"] for e in data])

        def fit(ids):
            vectorizer = Vectorizer(**VECTORIZER_PARAMS)
            features = vectorizer.fit_transform([data[i]["text"] for i in ids])
            classifier = Classifier(C=8.0, class_weight="balanced", max_iter=1000, random_state=42)
            classifier.fit(features, [data[i]["reason"] for i in ids])
            return vectorizer, classifier

        # Fit the evaluation vocabulary only on the training partition.
        vectorizer, classifier = fit(train_ids)
        expected = [data[i]["reason"] for i in test_ids]
        predicted = classifier.predict(vectorizer.transform([data[i]["text"] for i in test_ids]))
        metadata = dict(
            version=uuid4().hex[:12], trained_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            examples=len(data), starter_examples=sum(e["origin"] == "starter" for e in data),
            reviewed_examples=sum(e["origin"] == "reviewed" for e in data), class_counts=dict(counts),
            reviewed_through=max((e.get("reviewed_at") or "" for e in reviewed), default=""),
            evaluation=dict(accuracy=round(float(accuracy(expected, predicted)), 4),
                            macro_f1=round(float(f1(expected, predicted, average="macro")), 4),
                            examples=len(test_ids),
                            starter_examples=sum(data[i]["origin"] == "starter" for i in test_ids),
                            reviewed_examples=sum(data[i]["origin"] == "reviewed" for i in test_ids),
                            method="Stratified 25% holdout; may include synthetic starter examples"),
        )
        # The serving model then uses all examples, including the held-out subset.
        vectorizer, classifier = fit(indices)
        artifact = dict(format_version=1, metadata=metadata, classes=classifier.classes_.tolist(),
                        vectorizer_params={**VECTORIZER_PARAMS, "ngram_range": list(VECTORIZER_PARAMS["ngram_range"])},
                        vocabulary={k: int(v) for k, v in vectorizer.vocabulary_.items()},
                        idf=vectorizer.idf_.tolist(), weights=classifier.coef_.tolist(),
                        intercept=classifier.intercept_.tolist())
        self._decode(artifact)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, suffix=".json", delete=False) as f:
                temporary = f.name
                json.dump(artifact, f, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        self._refresh()
        return metadata
