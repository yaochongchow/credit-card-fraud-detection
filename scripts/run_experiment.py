"""
run_experiment.py
-----------------
Train and evaluate all model × strategy combinations and save results.

Usage
-----
    python scripts/run_experiment.py --data creditcard.csv

Outputs (written to outputs/)
-------
    results.csv       — per-model metrics table
    *.png             — ROC, PR, threshold-sweep, calibration, and cost plots
"""

import argparse
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data, preprocess
from src.imbalance import prepare, get_class_weight, STRATEGIES
from src.models import get_model, get_param_grid
from src.train import tune, tune_threshold
from src.evaluate import (
    compute_metrics,
    false_positives_per_10k,
    plot_roc,
    plot_prc,
    plot_threshold_sweep,
    plot_calibration,
    plot_cost_curve,
)

MODEL_NAMES = ["lr", "rf", "xgb", "svm"]
OUTPUT_DIR = "outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="creditcard.csv", help="Path to creditcard.csv")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    parser.add_argument("--n_iter", type=int, default=20, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading data from {args.data} ...")
    df = load_data(args.data)
    X_train, X_test, y_train, y_test = preprocess(df, random_state=args.seed)
    n_negatives = int((y_test == 0).sum())

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Fraud rate (test): {y_test.mean():.4%}\n")

    records = []

    for model_name in args.models:
        for strategy in args.strategies:
            tag = f"{model_name.upper()}-{strategy}"
            print(f"[{tag}] Preparing data ...")

            X_res, y_res = prepare(X_train, y_train, strategy, random_state=args.seed)
            cw = get_class_weight(strategy)
            model = get_model(model_name, class_weight=cw, random_state=args.seed)
            param_grid = get_param_grid(model_name)

            print(f"[{tag}] Tuning ({args.n_iter} iterations, {args.cv}-fold CV) ...")
            best_model, best_params, cv_score = tune(
                model, param_grid, X_res, y_res,
                n_iter=args.n_iter, cv=args.cv, random_state=args.seed
            )
            print(f"[{tag}] CV PR-AUC: {cv_score:.4f} | Params: {best_params}")

            # Tune threshold on a small held-out slice of training data
            # to avoid leaking test labels
            from sklearn.model_selection import train_test_split
            _, X_val, _, y_val = train_test_split(
                X_res, y_res, test_size=0.1, stratify=y_res, random_state=args.seed
            )
            threshold = tune_threshold(best_model, X_val, y_val)

            metrics = compute_metrics(best_model, X_test, y_test, threshold)
            metrics["fp_per_10k"] = false_positives_per_10k(metrics["fp"], n_negatives)
            metrics["model"] = model_name.upper()
            metrics["strategy"] = strategy
            metrics["cv_pr_auc"] = cv_score
            records.append(metrics)

            # --- Plots ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            plot_roc(best_model, X_test, y_test, label=tag, ax=axes[0])
            plot_prc(best_model, X_test, y_test, label=tag, ax=axes[1])
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, f"{tag}-roc-prc.png"), dpi=120)
            plt.close(fig)

            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            plot_threshold_sweep(best_model, X_test, y_test, ax=axes[0])
            plot_calibration(best_model, X_test, y_test, label=tag, ax=axes[1])
            plot_cost_curve(best_model, X_test, y_test, ax=axes[2])
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, f"{tag}-analysis.png"), dpi=120)
            plt.close(fig)

            print(
                f"[{tag}] PR-AUC={metrics['pr_auc']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"Threshold={threshold:.3f} | "
                f"FP/10k={metrics['fp_per_10k']:.1f}\n"
            )

    # Save results table
    results_df = pd.DataFrame(records)
    col_order = [
        "model", "strategy", "pr_auc", "roc_auc", "f1",
        "precision", "recall", "brier", "threshold",
        "tp", "fp", "tn", "fn", "fp_per_10k", "cv_pr_auc",
    ]
    results_df = results_df[col_order]
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/results.csv")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
