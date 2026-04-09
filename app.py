"""
Credit Card Fraud Detection — Streamlit Demo
--------------------------------------------
Run:
    streamlit run app.py

Prerequisites:
    python scripts/train_best_model.py --data creditcard.csv
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Load model artefacts ─────────────────────────────────────────────────────
MODEL_DIR = "model"

@st.cache_resource
def load_artefacts():
    model   = joblib.load(os.path.join(MODEL_DIR, "xgb_smote.pkl"))
    scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "threshold.txt")) as f:
        threshold = float(f.read().strip())
    with open(os.path.join(MODEL_DIR, "feature_names.txt")) as f:
        feature_names = f.read().strip().splitlines()
    return model, scaler, threshold, feature_names


@st.cache_data
def load_sample_transactions():
    if os.path.exists("creditcard.csv"):
        df = pd.read_csv("creditcard.csv")
        df["Log_Amount"] = np.log1p(df["Amount"])
        fraud = df[df.Class == 1].sample(50, random_state=42).reset_index(drop=True)
        legit = df[df.Class == 0].sample(50, random_state=42).reset_index(drop=True)
        return fraud, legit
    return None, None


try:
    model, scaler, THRESHOLD, feature_names = load_artefacts()
    model_loaded = True
except Exception:
    model_loaded = False

fraud_samples, legit_samples = load_sample_transactions()
has_samples = fraud_samples is not None

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict(feature_values: np.ndarray):
    """Scale features and return (fraud_probability, is_fraud)."""
    X = feature_values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]
    return float(prob), prob >= THRESHOLD


def gauge_chart(prob: float, threshold: float):
    color = "#e74c3c" if prob >= threshold else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 48}},
        title={"text": "Fraud Probability", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 30],  "color": "#d5f5e3"},
                {"range": [30, 60], "color": "#fef9e7"},
                {"range": [60, 100],"color": "#fdecea"},
            ],
            "threshold": {
                "line": {"color": "#2c3e50", "width": 3},
                "thickness": 0.75,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def importance_chart(model, feature_names):
    scores = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": scores})
    df = df.sort_values("importance", ascending=True).tail(15)
    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        title="Top 15 Feature Importances (XGBoost)",
        color="importance",
        color_continuous_scale=["#aed6f1", "#1a5276"],
    )
    fig.update_layout(
        height=420,
        margin=dict(t=40, b=20, l=10, r=10),
        coloraxis_showscale=False,
        yaxis_title="",
        xaxis_title="Importance Score",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Fraud Detector")
    st.caption("XGBoost + SMOTE · PR-AUC 0.879 · F1 0.89")
    st.divider()

    st.subheader("Model performance (test set)")
    col1, col2 = st.columns(2)
    col1.metric("PR-AUC",  "0.879")
    col2.metric("ROC-AUC", "0.985")
    col1.metric("F1",      "0.89")
    col2.metric("Recall",  "86.7%")
    st.metric("False positives / 10k transactions", "3.9")
    st.divider()

    st.subheader("Decision threshold")
    if model_loaded:
        st.info(f"Optimal threshold: **{THRESHOLD:.3f}**\n\nTuned on a held-out validation slice to maximise F1.")

    st.divider()
    st.caption("Dataset: 284,807 transactions · 0.17% fraud rate\nModel: XGBoost + SMOTE oversampling")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Credit Card Fraud Detection")
st.markdown(
    "Enter transaction details manually **or** load a real sample from the dataset. "
    "The model returns a fraud probability in real time."
)

if not model_loaded:
    st.error(
        "Model not found. Run this first:\n\n"
        "```\npython scripts/train_best_model.py --data creditcard.csv\n```"
    )
    st.stop()

# ── Sample loader ─────────────────────────────────────────────────────────────
st.subheader("Load a sample transaction")

load_col1, load_col2, load_col3 = st.columns([1, 1, 4])

if has_samples:
    if load_col1.button("🚨 Load fraud sample", use_container_width=True):
        idx = np.random.randint(len(fraud_samples))
        st.session_state["sample"] = fraud_samples.iloc[idx]
    if load_col2.button("✅ Load legit sample", use_container_width=True):
        idx = np.random.randint(len(legit_samples))
        st.session_state["sample"] = legit_samples.iloc[idx]
else:
    st.info("Place `creditcard.csv` in the project root to enable sample loading.")

st.divider()

# ── Feature inputs ────────────────────────────────────────────────────────────
sample = st.session_state.get("sample", None)

st.subheader("Transaction features")

amount_default = float(np.expm1(sample["Log_Amount"])) if sample is not None else 100.0
amount = st.number_input(
    "Transaction Amount ($)",
    min_value=0.0,
    max_value=30000.0,
    value=amount_default,
    step=1.0,
)
log_amount = np.log1p(amount)

v_features = [f for f in feature_names if f.startswith("V")]
v_defaults = {f: float(sample[f]) if sample is not None else 0.0 for f in v_features}

# Render V features in a 4-column grid
st.markdown("**PCA Components (V1 – V28)**")
cols = st.columns(4)
v_values = {}
for i, feat in enumerate(v_features):
    with cols[i % 4]:
        v_values[feat] = st.slider(
            feat,
            min_value=-20.0,
            max_value=20.0,
            value=round(v_defaults[feat], 3),
            step=0.001,
            format="%.3f",
        )

# Assemble feature vector in correct order
feature_vector = np.array(
    [v_values[f] for f in v_features] + [log_amount],
    dtype=np.float64,
)

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Prediction")

prob, is_fraud = predict(feature_vector)

result_col, gauge_col = st.columns([1, 1])

with gauge_col:
    st.plotly_chart(gauge_chart(prob, THRESHOLD), use_container_width=True)

with result_col:
    if is_fraud:
        st.error("### 🚨 FRAUD DETECTED")
    else:
        st.success("### ✅ LEGITIMATE TRANSACTION")

    st.markdown(f"**Probability:** `{prob:.4f}`")
    st.markdown(f"**Threshold:** `{THRESHOLD:.3f}`")
    st.markdown(f"**Decision:** `{'FRAUD' if is_fraud else 'LEGITIMATE'}`")

    if sample is not None:
        true_label = int(sample.get("Class", -1))
        if true_label != -1:
            label_str = "🚨 Fraud" if true_label == 1 else "✅ Legitimate"
            correct = (true_label == 1) == is_fraud
            st.markdown(f"**True label:** {label_str}")
            if correct:
                st.success("Model prediction is **correct**")
            else:
                st.warning("Model prediction is **incorrect**")

# ── Feature importance ────────────────────────────────────────────────────────
st.divider()
st.plotly_chart(importance_chart(model, feature_names), use_container_width=True)
