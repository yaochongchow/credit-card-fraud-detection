"""
Credit Card Fraud Detection — Streamlit Demo
--------------------------------------------
Run:
    streamlit run app.py

Prerequisites:
    python scripts/train_best_model.py --data creditcard.csv
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide",
)

MODEL_DIR = "model"

MODEL_REGISTRY = {
    "lr_smote":  {"label": "Logistic Regression",  "strategy": "SMOTE",        "pr_auc": 0.800, "roc_auc": 0.977, "f1": 0.14},
    "rf_cw":     {"label": "Random Forest",         "strategy": "Class Weight", "pr_auc": 0.887, "roc_auc": 0.984, "f1": 0.87},
    "xgb_smote": {"label": "XGBoost",               "strategy": "SMOTE",        "pr_auc": 0.879, "roc_auc": 0.985, "f1": 0.89},
    "svm_smote": {"label": "SVM",                   "strategy": "SMOTE",        "pr_auc": 0.735, "roc_auc": 0.977, "f1": 0.00},
}

# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_names.txt")) as f:
        feature_names = f.read().strip().splitlines()
    with open(os.path.join(MODEL_DIR, "thresholds.json")) as f:
        thresholds = json.load(f)
    models = {}
    for key in MODEL_REGISTRY:
        path = os.path.join(MODEL_DIR, f"{key}.pkl")
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models, scaler, thresholds, feature_names


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
    models, scaler, thresholds, feature_names = load_artefacts()
    model_loaded = bool(models)
except Exception:
    model_loaded = False

fraud_samples, legit_samples = load_sample_transactions()
has_samples = fraud_samples is not None

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict_all(feature_vector: np.ndarray) -> dict:
    X = scaler.transform(feature_vector.reshape(1, -1))
    results = {}
    for key, model in models.items():
        prob = float(model.predict_proba(X)[0, 1])
        thr  = thresholds.get(key, 0.5)
        results[key] = {"prob": prob, "fraud": prob >= thr, "threshold": thr}
    return results


def mini_gauge(prob: float, threshold: float, title: str):
    color = "#e74c3c" if prob >= threshold else "#27ae60"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": title, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickfont": {"size": 9}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  30], "color": "#d5f5e3"},
                {"range": [30, 60], "color": "#fef9e7"},
                {"range": [60,100], "color": "#fdecea"},
            ],
            "threshold": {
                "line": {"color": "#2c3e50", "width": 3},
                "thickness": 0.75,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(t=50, b=0, l=10, r=10))
    return fig


def comparison_bar(results: dict):
    rows = []
    for key, r in results.items():
        meta = MODEL_REGISTRY[key]
        rows.append({
            "Model": f"{meta['label']}\n({meta['strategy']})",
            "Fraud Probability (%)": round(r["prob"] * 100, 2),
            "Verdict": "FRAUD" if r["fraud"] else "LEGIT",
        })
    df = pd.DataFrame(rows)
    colors = ["#e74c3c" if v == "FRAUD" else "#27ae60" for v in df["Verdict"]]
    fig = go.Figure(go.Bar(
        x=df["Model"],
        y=df["Fraud Probability (%)"],
        marker_color=colors,
        text=df["Fraud Probability (%)"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        title="Fraud Probability by Model",
        yaxis=dict(range=[0, 110], title="Fraud Probability (%)"),
        xaxis_title="",
        height=340,
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False,
    )
    return fig


def importance_chart(model, feature_names, title):
    if not hasattr(model, "feature_importances_"):
        return None
    scores = model.feature_importances_
    df = pd.DataFrame({"Feature": feature_names, "Importance": scores})
    df = df.sort_values("Importance", ascending=True).tail(15)
    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        title=title,
        color="Importance",
        color_continuous_scale=["#aed6f1", "#1a5276"],
    )
    fig.update_layout(
        height=400,
        margin=dict(t=40, b=10, l=10, r=10),
        coloraxis_showscale=False,
        yaxis_title="",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Fraud Detector")
    st.caption("4 models · best config per classifier")
    st.divider()

    st.subheader("Model performance (test set)")
    perf_df = pd.DataFrame([
        {
            "Model": f"{v['label']} ({v['strategy']})",
            "PR-AUC": v["pr_auc"],
            "ROC-AUC": v["roc_auc"],
            "F1": v["f1"],
        }
        for v in MODEL_REGISTRY.values()
    ])
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "Dataset: 284,807 transactions · 0.17% fraud rate\n\n"
        "Thresholds tuned on held-out validation slice to maximise F1."
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Credit Card Fraud Detection")
st.markdown(
    "Load a real transaction or adjust the sliders, then see all four models predict simultaneously."
)

if not model_loaded:
    st.error(
        "No models found. Run this first:\n\n"
        "```\npython scripts/train_best_model.py --data creditcard.csv\n```"
    )
    st.stop()

# ── Sample loader ─────────────────────────────────────────────────────────────
st.subheader("Load a sample transaction")
c1, c2, _ = st.columns([1, 1, 4])
if has_samples:
    if c1.button("🚨 Fraud sample", use_container_width=True):
        st.session_state["sample"] = fraud_samples.iloc[np.random.randint(len(fraud_samples))]
    if c2.button("✅ Legit sample", use_container_width=True):
        st.session_state["sample"] = legit_samples.iloc[np.random.randint(len(legit_samples))]
else:
    st.info("Place `creditcard.csv` in the project root to enable sample loading.")

st.divider()

# ── Feature inputs ────────────────────────────────────────────────────────────
sample = st.session_state.get("sample", None)
amount_default = float(np.expm1(sample["Log_Amount"])) if sample is not None else 100.0
amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=30000.0,
                          value=amount_default, step=1.0)
log_amount = np.log1p(amount)

v_features = [f for f in feature_names if f.startswith("V")]
v_defaults = {f: float(sample[f]) if sample is not None else 0.0 for f in v_features}

st.markdown("**PCA Components (V1 – V28)**")
cols = st.columns(4)
v_values = {}
for i, feat in enumerate(v_features):
    with cols[i % 4]:
        v_values[feat] = st.slider(feat, min_value=-20.0, max_value=20.0,
                                    value=round(v_defaults[feat], 3), step=0.001, format="%.3f")

feature_vector = np.array([v_values[f] for f in v_features] + [log_amount], dtype=np.float64)

# ── Predictions ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Predictions — all models")

results = predict_all(feature_vector)

# True label badge (if sample loaded)
if sample is not None and "Class" in sample:
    true_label = int(sample["Class"])
    label_str = "🚨 True label: **FRAUD**" if true_label == 1 else "✅ True label: **LEGITIMATE**"
    st.info(label_str)

# 4 model cards
model_keys = list(results.keys())
gauge_cols = st.columns(len(model_keys))

for col, key in zip(gauge_cols, model_keys):
    r    = results[key]
    meta = MODEL_REGISTRY[key]
    with col:
        title = f"{meta['label']}<br>({meta['strategy']})"
        st.plotly_chart(mini_gauge(r["prob"], r["threshold"], title),
                        use_container_width=True)
        if r["fraud"]:
            st.error("🚨 **FRAUD**", icon=None)
        else:
            st.success("✅ **LEGIT**", icon=None)
        st.caption(f"threshold: {r['threshold']:.3f}")

# Comparison bar chart
st.plotly_chart(comparison_bar(results), use_container_width=True)

# ── Feature importance ────────────────────────────────────────────────────────
st.divider()
st.subheader("Feature Importance")

importance_models = {k: v for k, v in models.items() if hasattr(v, "feature_importances_")}
if importance_models:
    selected_key = st.selectbox(
        "Select model",
        options=list(importance_models.keys()),
        format_func=lambda k: f"{MODEL_REGISTRY[k]['label']} ({MODEL_REGISTRY[k]['strategy']})",
    )
    fig = importance_chart(
        importance_models[selected_key],
        feature_names,
        f"Top 15 Feature Importances — {MODEL_REGISTRY[selected_key]['label']}",
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Feature importance is available for Random Forest and XGBoost.")
