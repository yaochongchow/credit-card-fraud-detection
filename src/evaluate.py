"""Evaluation metrics, threshold-aware prediction, and plots."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Return a dict of evaluation metrics at a given decision threshold.

    Includes PR-AUC, ROC-AUC, F1, precision, recall, and confusion-matrix
    values (TP, FP, TN, FN).
    """
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "brier": brier_score_loss(y_test, proba),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": threshold,
    }


def false_positives_per_10k(fp: int, total_negatives: int) -> float:
    """Express false-positive rate as FP per 10,000 legitimate transactions."""
    return fp / total_negatives * 10_000


# ---------------------------------------------------------------------------
# ROC / PR curves
# ---------------------------------------------------------------------------

def plot_roc(model, X_test, y_test, label: str = "", ax=None):
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax = ax or plt.gca()
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_prc(model, X_test, y_test, label: str = "", ax=None):
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    ax = ax or plt.gca()
    ax.step(recall, precision, where="post", label=f"{label} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def plot_threshold_sweep(model, X_test, y_test, ax=None):
    """Plot Precision, Recall, and F1 vs decision threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)

    ax = ax or plt.gca()
    ax.plot(thresholds, precision[:-1], label="Precision")
    ax.plot(thresholds, recall[:-1], label="Recall")
    ax.plot(thresholds, f1, label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs Decision Threshold")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def plot_calibration(model, X_test, y_test, label: str = "", n_bins: int = 10, ax=None):
    """Reliability diagram: mean predicted probability vs fraction of positives."""
    proba = model.predict_proba(X_test)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins)
    brier = brier_score_loss(y_test, proba)

    ax = ax or plt.gca()
    ax.plot(mean_pred, fraction_pos, "s-", label=f"{label} (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Plot")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Business cost analysis
# ---------------------------------------------------------------------------

def cost_at_threshold(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cost_fn: float = 100.0,
    cost_fp: float = 5.0,
    thresholds: np.ndarray | None = None,
) -> tuple:
    """
    Compute total business cost across a range of thresholds.

    Parameters
    ----------
    cost_fn : cost of a missed fraud (false negative), default $100
    cost_fp : cost of a false alarm (false positive), default $5

    Returns
    -------
    thresholds, costs, optimal_threshold
    """
    proba = model.predict_proba(X_test)[:, 1]
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 200)

    costs = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        costs.append(fn * cost_fn + fp * cost_fp)

    costs = np.array(costs)
    optimal = thresholds[np.argmin(costs)]
    return thresholds, costs, optimal


def plot_cost_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cost_fn: float = 100.0,
    cost_fp: float = 5.0,
    ax=None,
):
    """Plot total business cost vs decision threshold."""
    thresholds, costs, optimal = cost_at_threshold(
        model, X_test, y_test, cost_fn, cost_fp
    )
    ax = ax or plt.gca()
    ax.plot(thresholds, costs, label="Total cost")
    ax.axvline(optimal, color="r", linestyle="--", label=f"Optimal threshold={optimal:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(f"Cost (FN=${cost_fn}, FP=${cost_fp})")
    ax.set_title("Business Cost vs Decision Threshold")
    ax.legend()
    return ax
