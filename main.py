import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from config import FOLDS, SEED, VALIDATION_RECORDS
from data.ludb import load_ludb
from evaluate import evaluate, fit_gates, predict_logits, summarize
from train import fit


def record_folds(record_ids):
    ordered = np.argsort(record_ids)
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (remaining, test) in enumerate(splitter.split(ordered)):
        rng = np.random.default_rng(SEED + fold)
        validation = rng.choice(remaining, size=VALIDATION_RECORDS, replace=False)
        training = np.setdiff1d(remaining, validation)
        yield tuple(np.sort(ordered[indices]) for indices in (training, validation, test))


def format_score(value):
    if isinstance(value, dict):
        return f"{value['mean']:.3f} ± {value['std']:.3f}"
    return f"{value:.3f}"


def print_metrics(metrics):
    for group, rows in metrics.items():
        print(f"{group.capitalize()} (%)")
        for name, scores in rows.items():
            columns = "  ".join(f"{metric} {format_score(value)}" for metric, value in scores.items())
            print(f"  {name:12s}  {columns}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate CWL-ECG on LUDB.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ludb"), help="LUDB root (default: data/ludb)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Metrics directory (default: results)")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not torch.cuda.is_available():
        parser.error("CWL-ECG requires a CUDA-enabled NVIDIA GPU.")
    if not (data_dir / "RECORDS").is_file():
        parser.error(f"LUDB RECORDS file not found in {data_dir}")

    device = torch.device("cuda")
    record_ids, signals, labels = load_ludb(data_dir)
    fold_results = []
    for fold, (train_indices, val_indices, test_indices) in enumerate(record_folds(record_ids), start=1):
        print(f"\nFold {fold}/{FOLDS}", flush=True)
        model = fit(signals, labels, train_indices, val_indices, device)
        gates = fit_gates(labels[train_indices])
        logits = predict_logits(model, signals[test_indices], device)
        metrics = evaluate(logits, labels[test_indices], gates)
        fold_results.append(metrics)
        print_metrics(metrics)
        del model

    summary = summarize(fold_results)
    print("\nFive-fold mean ± standard deviation")
    print_metrics(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "metrics.json"
    results = {"folds": fold_results, "summary": summary}
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
