"""
train_best_model.py
-------------------
Train the best configuration of each model and save artefacts for the demo.

Models saved
------------
  LR   + SMOTE         → model/lr_smote.pkl
  RF   + Class Weight  → model/rf_cw.pkl
  XGB  + SMOTE         → model/xgb_smote.pkl
  SVM  + SMOTE         → model/svm_smote.pkl  (trained on 30k subsample)

Shared artefacts
----------------
  model/scaler.pkl
  model/feature_names.txt
  model/thresholds.json   — optimal threshold per model key

Usage
-----
    python scripts/train_best_model.py --data creditcard.csv
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_data
from src.imbalance import apply_smote

MODEL_DIR = "model"


def tune_threshold(model, X_val, y_val) -> float:
    proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    return float(thresholds[np.argmax(f1)])


def evaluate(model, X_test, y_test, threshold, name):
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    print(
        f"  {name:<25} PR-AUC={average_precision_score(y_test, proba):.4f}  "
        f"ROC-AUC={roc_auc_score(y_test, proba):.4f}  "
        f"F1={f1_score(y_test, y_pred):.4f}  threshold={threshold:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="creditcard.csv")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load + preprocess ────────────────────────────────────────────────────
    print(f"Loading {args.data} ...")
    df = load_data(args.data)
    df = df.copy()
    df["Log_Amount"] = np.log1p(df["Amount"])
    df.drop(columns=["Amount", "Time"], inplace=True)

    feature_names = [c for c in df.columns if c != "Class"]
    X = df[feature_names].values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Validation slice for threshold tuning (never touches test set)
    _, X_val, _, y_val = train_test_split(
        X_train_s, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    # Apply SMOTE once — reused by LR, XGB, SVM
    print("Applying SMOTE ...")
    X_sm, y_sm = apply_smote(X_train_s, y_train)

    thresholds = {}
    print("\nTraining models ...")

    # ── Logistic Regression + SMOTE ──────────────────────────────────────────
    print("\n[1/4] Logistic Regression + SMOTE")
    lr = LogisticRegression(C=100, fit_intercept=True, tol=1e-3, max_iter=500, random_state=42)
    lr.fit(X_sm, y_sm)
    thresholds["lr_smote"] = tune_threshold(lr, X_val, y_val)
    evaluate(lr, X_test_s, y_test, thresholds["lr_smote"], "LR + SMOTE")
    joblib.dump(lr, os.path.join(MODEL_DIR, "lr_smote.pkl"))

    # ── Random Forest + Class Weight ─────────────────────────────────────────
    print("\n[2/4] Random Forest + Class Weight")
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=12,
        criterion="entropy",
        bootstrap=False,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_s, y_train)
    thresholds["rf_cw"] = tune_threshold(rf, X_val, y_val)
    evaluate(rf, X_test_s, y_test, thresholds["rf_cw"], "RF + Class Weight")
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_cw.pkl"))

    # ── XGBoost + SMOTE ──────────────────────────────────────────────────────
    print("\n[3/4] XGBoost + SMOTE")
    xgb = XGBClassifier(
        n_estimators=1200,
        max_depth=5,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        reg_alpha=0.1,
        gamma=0.0,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=42,
    )
    xgb.fit(X_sm, y_sm, verbose=False)
    thresholds["xgb_smote"] = tune_threshold(xgb, X_val, y_val)
    evaluate(xgb, X_test_s, y_test, thresholds["xgb_smote"], "XGB + SMOTE")
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_smote.pkl"))

    # ── SVM + SMOTE (subsample for speed) ────────────────────────────────────
    print("\n[4/4] SVM + SMOTE  (training on 30k subsample — may take ~2 min)")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_sm), size=min(30_000, len(X_sm)), replace=False)
    svm = SVC(C=10, kernel="rbf", gamma="scale", probability=True, random_state=42)
    svm.fit(X_sm[idx], y_sm[idx])
    thresholds["svm_smote"] = tune_threshold(svm, X_val, y_val)
    evaluate(svm, X_test_s, y_test, thresholds["svm_smote"], "SVM + SMOTE")
    joblib.dump(svm, os.path.join(MODEL_DIR, "svm_smote.pkl"))

    # ── Save shared artefacts ─────────────────────────────────────────────────
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))
    with open(os.path.join(MODEL_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nAll models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
