"""
train_best_model.py
-------------------
Train the best model (XGBoost + SMOTE) and save it for the Streamlit demo.

Usage
-----
    python scripts/train_best_model.py --data creditcard.csv

Outputs (written to model/)
-------
    xgb_smote.pkl   — trained XGBoost classifier
    scaler.pkl      — fitted StandardScaler
    threshold.txt   — optimal decision threshold
    feature_names.txt
"""

import argparse
import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data
from src.imbalance import apply_smote
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve

MODEL_DIR = "model"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="creditcard.csv")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = load_data(args.data)

    # --- Preprocessing ---
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

    # --- SMOTE ---
    print("Applying SMOTE ...")
    X_res, y_res = apply_smote(X_train_s, y_train)
    print(f"  After SMOTE: {y_res.sum()} fraud / {(y_res==0).sum()} legit")

    # --- Train with best known hyperparameters ---
    print("Training XGBoost ...")
    model = XGBClassifier(
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
    model.fit(X_res, y_res, verbose=False)

    # --- Threshold tuning on a validation slice ---
    _, X_val, _, y_val = train_test_split(
        X_train_s, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    proba_val = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    threshold = float(thresholds[np.argmax(f1_scores)])
    print(f"Optimal threshold: {threshold:.4f}")

    # --- Evaluate on test set ---
    proba_test = model.predict_proba(X_test_s)[:, 1]
    y_pred = (proba_test >= threshold).astype(int)
    pr_auc = average_precision_score(y_test, proba_test)
    roc_auc = roc_auc_score(y_test, proba_test)
    f1 = f1_score(y_test, y_pred)
    print(f"Test  PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  F1={f1:.4f}")

    # --- Save artefacts ---
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_smote.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "threshold.txt"), "w") as f:
        f.write(str(threshold))
    with open(os.path.join(MODEL_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))

    print(f"\nSaved to {MODEL_DIR}/")
    print("  xgb_smote.pkl, scaler.pkl, threshold.txt, feature_names.txt")


if __name__ == "__main__":
    main()
