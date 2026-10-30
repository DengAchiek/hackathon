"""Optional: fine-tune MiniLM with SetFit on fraud SMS data."""
import argparse
from pathlib import Path
import json

from fraud_ml import SCAM_LABELS
from .common import provenance, write_json
from .evaluation import choose_threshold, evaluate, save_report


def train(directory, output, base_model, revision=None):
    import numpy as np
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    from transformers import set_seed

    set_seed(42)
    # Load splits
    parts = {}
    for name in ("train", "validation", "test"):
        path = Path(directory) / f"{name}.csv"
        import csv
        rows = []
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        parts[name] = rows
    labels = sorted(SCAM_LABELS)
    dataset = Dataset.from_dict({"text": [row["text"] for row in parts["train"]],
                                 "label": [labels.index(row["label"]) for row in parts["train"]]})
    kwargs = {"revision": revision} if revision else {}
    model = SetFitModel.from_pretrained(base_model, labels=labels, **kwargs)
    output = Path(output)
    args = TrainingArguments(output_dir=str(output/"checkpoints"), batch_size=8, num_epochs=1,
                             num_iterations=10, seed=42, report_to="none")
    Trainer(model=model, args=args, train_dataset=dataset).train()
    validation = np.asarray(model.predict_proba([row["text"] for row in parts["validation"]], as_numpy=True))
    threshold, curve = choose_threshold([row["label"] for row in parts["validation"]], validation, labels)
    probabilities = np.asarray(model.predict_proba([row["text"] for row in parts["test"]], as_numpy=True))
    # Evaluate using reason-style
    from .evaluation import evaluate as eval_fn
    metrics = eval_fn(parts["test"], probabilities, labels, threshold, reason_key="label")
    target = output / "model"
    model.save_pretrained(str(target))
    record = provenance([Path(directory)/f"{name}.csv" for name in parts])
    record.update(base_model=base_model, revision=revision or "main")
    report = dict(algorithm="SetFit + MiniLM", selected=dict(epochs=1, threshold=threshold),
                  validation_threshold_curve=curve, test=metrics, provenance=record)
    save_report(output, report, parts["test"], probabilities, labels, reason_key="label")
    write_json(output / "serving.json", dict(kind="setfit", path="model", labels=labels,
               threshold=threshold, version="fraud-setfit-" + record["sha256"][str(Path(directory)/"train.csv")][:10]))
    print("SetFit fraud candidate saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="ml_workflow/work/fraud_splits")
    parser.add_argument("--out", default="ml_workflow/work/fraud_setfit")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--revision")
    args = parser.parse_args()
    train(args.splits, args.out, args.base_model, args.revision)
