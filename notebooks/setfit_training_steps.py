"""Self-contained SetFit training steps, embedded into the Python notebook."""


def train_setfit_candidate(partitions, output, labels, *, base_model="sentence-transformers/all-MiniLM-L6-v2",
                           revision=None, epochs=1, iterations=10, seed=42,
                           target_accuracy=.90, min_accepted=5):
    """Fit on train, select abstention on validation, report test once, save/reload."""
    import csv
    import hashlib
    import json
    import os
    from collections import Counter
    from datetime import datetime, timezone
    from importlib.metadata import version
    from pathlib import Path
    import time

    import numpy as np
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from setfit import SetFitModel, Trainer, TrainingArguments
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from transformers import set_seed

    labels = list(labels)
    parts = {name: (value.to_dict("records") if hasattr(value, "to_dict") else list(value))
             for name, value in partitions.items()}
    if set(parts) != {"train", "validation", "test"}:
        raise ValueError("Provide the same frozen train, validation and test partitions used by the baseline.")
    memberships = {}
    for name, rows in parts.items():
        if not rows or set(row["reason"] for row in rows) != set(labels):
            raise ValueError(f"{name}: every blocker must be represented.")
        for row in rows:
            keys = [("id", str(row["id"])), ("text", " ".join(row["text"].casefold().split()))]
            for field in ("split_group", "group_id", "conversation_id"):
                if row.get(field) is not None and str(row[field]) not in {"", "nan", "None"}:
                    keys.append((field, str(row[field])))
            for key in keys:
                if key in memberships and memberships[key] != name:
                    raise ValueError("Data leakage: related examples cross frozen partitions.")
                memberships[key] = name

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["ACCELERATE_USE_CPU"] = "true"
    torch.set_num_threads(2)
    set_seed(seed)
    resolved_revision = HfApi().model_info(base_model, revision=revision, token=False).sha
    print("SetFit base checkpoint:", base_model, "revision:", resolved_revision, flush=True)
    model = SetFitModel.from_pretrained(base_model, revision=resolved_revision,
                                        labels=labels, token=False, trust_remote_code=False)
    model.to("cpu")
    dataset = Dataset.from_dict({"text": [row["text"] for row in parts["train"]],
        "label": [labels.index(row["reason"]) for row in parts["train"]]})
    arguments = TrainingArguments(output_dir=str(output / "checkpoints"), batch_size=8,
        num_epochs=epochs, num_iterations=iterations, max_length=128, seed=seed,
        report_to="none", save_strategy="no", show_progress_bar=False)
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset)
    started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - started

    def probabilities(texts, candidate=model):
        scores = np.asarray(candidate.predict_proba(texts, as_numpy=True), dtype=float)
        if (scores.shape != (len(texts), len(labels)) or not np.isfinite(scores).all()
                or (scores < 0).any() or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("SetFit returned invalid class scores.")
        return scores

    validation_scores = probabilities([row["text"] for row in parts["validation"]])
    validation_truth = np.asarray([row["reason"] for row in parts["validation"]])
    validation_predictions = np.asarray(labels)[validation_scores.argmax(axis=1)]
    curve = []
    for threshold in (.35, .45, .55, .65, .75, .85, .95):
        accepted = validation_scores.max(axis=1) >= threshold
        curve.append({"threshold": threshold, "accepted": int(accepted.sum()),
            "coverage": float(accepted.mean()), "accepted_accuracy": float(
                (validation_predictions[accepted] == validation_truth[accepted]).mean()) if accepted.any() else None})
    eligible = [row for row in curve if row["accepted"] >= min_accepted
                and row["accepted_accuracy"] >= target_accuracy]
    threshold = max(eligible, key=lambda row: row["coverage"])["threshold"] if eligible else 1.0

    truth = np.asarray([row["reason"] for row in parts["test"]])
    test_scores = probabilities([row["text"] for row in parts["test"]])
    predicted = np.asarray(labels)[test_scores.argmax(axis=1)]
    accepted = test_scores.max(axis=1) >= threshold
    metrics = {"examples": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "threshold": threshold, "accepted": int(accepted.sum()), "coverage": float(accepted.mean()),
        "accepted_accuracy": float(accuracy_score(truth[accepted], predicted[accepted])) if accepted.any() else None,
        "per_class": classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0),
        "class_order": labels, "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist()}

    model_dir = output / "model"
    model.save_pretrained(str(model_dir))
    reloaded = SetFitModel.from_pretrained(str(model_dir), local_files_only=True)
    reloaded.to("cpu")
    if list(reloaded.labels or []) != labels or list(reloaded.model_head.classes_) != list(range(len(labels))):
        raise ValueError("Saved SetFit class mapping changed during reload.")
    np.testing.assert_allclose(test_scores,
        probabilities([row["text"] for row in parts["test"]], reloaded), rtol=1e-5, atol=1e-6)

    latencies = []
    for row in parts["test"]:
        started = time.perf_counter()
        probabilities([row["text"]], reloaded)
        latencies.append((time.perf_counter() - started) * 1000)
    split_hashes = {}
    for name, rows in parts.items():
        content = [[str(row.get(key, "")) for key in ("id", "text", "reason", "source", "is_synthetic")]
                   for row in sorted(rows, key=lambda row: str(row["id"]))]
        split_hashes[name] = hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()
    artifact_hashes = {str(path.relative_to(model_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(model_dir.rglob("*")) if path.is_file()}
    model_version = "setfit-" + hashlib.sha256(json.dumps(artifact_hashes, sort_keys=True).encode()).hexdigest()[:12]
    environment = {name: version(name) for name in (
        "setfit", "torch", "sentence-transformers", "transformers", "datasets", "scikit-learn", "accelerate")}
    trained_at = datetime.now(timezone.utc).isoformat()
    report = {"algorithm": "SetFit + MiniLM", "model_version": model_version,
        "base_model": base_model, "resolved_revision": resolved_revision,
        "training_examples": len(parts["train"]), "split_sizes": {name: len(rows) for name, rows in parts.items()},
        "selected": {"epochs": epochs, "iterations": iterations, "threshold": threshold, "seed": seed},
        "validation_macro_f1": float(f1_score(validation_truth, validation_predictions,
            labels=labels, average="macro", zero_division=0)), "validation_threshold_curve": curve,
        "test": metrics, "training_seconds": training_seconds,
        "latency": {"median_ms": float(np.median(latencies)), "p95_ms": float(np.percentile(latencies, 95))},
        "reload_parity": True, "environment": environment,
        "provenance": {"created_at": trained_at, "split_sha256": split_hashes, "artifact_sha256": artifact_hashes},
        "warning": "Small synthetic practice results do not establish real-customer or multilingual accuracy. Scores are uncalibrated."}
    synthetic_count = sum(str(row.get("is_synthetic", "")) in {"1", "True", "true"} for row in parts["train"])
    serving = {"kind": "setfit", "path": "model", "labels": labels, "threshold": threshold,
        "version": model_version, "metadata": {"version": model_version, "trained_at": trained_at,
        "examples": len(parts["train"]), "starter_examples": synthetic_count,
        "reviewed_examples": len(parts["train"]) - synthetic_count, "reviewed_through": "",
        "class_counts": dict(Counter(row["reason"] for row in parts["train"])),
        "evaluation": metrics, "base_model": base_model, "resolved_revision": resolved_revision}}
    for name, value in (("evaluation.json", report), ("serving.json", serving)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "expected", "predicted", "score", "accepted"])
        writer.writeheader()
        for row, prediction, scores, used in zip(parts["test"], predicted, test_scores, accepted):
            writer.writerow({"id": row["id"], "expected": row["reason"], "predicted": prediction,
                             "score": float(scores.max()), "accepted": bool(used)})
    card = (f"# SetFit blocker candidate\n\nVersion: {model_version}\n\n"
        f"Base: {base_model} at {resolved_revision}\n\n"
        f"Split: {report['split_sizes']}; training partition only, no held-out refit.\n\n"
        f"Test macro F1: {metrics['macro_f1']:.3f}; accuracy: {metrics['accuracy']:.3f}; "
        f"coverage: {metrics['coverage']:.1%}.\n\n{report['warning']}\n\n"
        "Use only for blocker suggestions after confirming an unmet request, with human review. "
        "This model does not verify stock, payments or customer eligibility. "
        "Load only trusted local artifacts. Saving does not promote this candidate to the live app.\n")
    (output / "model_card.md").write_text(card, encoding="utf-8")
    (output / "requirements-training-tested.txt").write_text(
        "\n".join(f"{name}=={value}" for name, value in environment.items()) + "\n", encoding="utf-8")
    print("PASS: SetFit training, held-out evaluation, label mapping and reload parity.", flush=True)
    print("SetFit test macro F1:", round(metrics["macro_f1"], 3), "| coverage:", metrics["coverage"], flush=True)
    return reloaded, report
