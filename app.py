import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    roc_curve, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.utils import resample

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
section[data-testid="stSidebar"] { background-color: #f0f2f6; }
div[data-testid="stMetricValue"]  { font-size: 1.6rem !important; }
.fraud-box  { background:#fde8e8; border-left:4px solid #e53e3e; padding:16px 20px; border-radius:8px; }
.safe-box   { background:#e6f4ea; border-left:4px solid #2e7d32; padding:16px 20px; border-radius:8px; }
.fraud-title{ font-size:1.1rem; font-weight:600; color:#b91c1c; margin:0 0 4px; }
.safe-title { font-size:1.1rem; font-weight:600; color:#1b5e20; margin:0 0 4px; }
.sub-text   { color:#555; font-size:0.9rem; margin:0; }
.info-box   { background:#eff6ff; border-left:4px solid #3b82f6; padding:12px 16px;
              border-radius:8px; font-size:0.88rem; color:#1e3a5f; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH   = "fraud_model.pkl"
SCALER_PATH  = "scaler.pkl"
FEATURE_PATH = "feature_names.pkl"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💳 Fraud Detector")
    st.caption("Credit Card Fraud Detection — ML Project")
    st.divider()

    st.subheader("Step 1 — Upload Data")
    uploaded = st.file_uploader("Upload creditcard.csv", type=["csv"],
                                label_visibility="collapsed")
    st.markdown("""
    <div class="info-box">
    Download the free dataset from:<br>
    <b>Kaggle → Credit Card Fraud Detection</b><br>
    (by ULB Machine Learning Group)
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Step 2 — Settings")
    model_choice   = st.selectbox("Algorithm", ["Random Forest", "Logistic Regression"])
    balance_method = st.selectbox("Handle Imbalance", ["Undersampling", "Class Weights"])
    test_size      = st.slider("Test set size", 0.10, 0.35, 0.20, 0.05)
    n_trees        = st.slider("No. of trees (RF)", 50, 300, 100, 50) \
                     if model_choice == "Random Forest" else 100

    st.divider()
    train_btn = st.button("🚀  Train Model", use_container_width=True,
                          disabled=(uploaded is None))
    if uploaded is None:
        st.caption("Upload a CSV to enable training.")

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(f):
    return pd.read_csv(f)

def undersample(X_tr, y_tr):
    df  = pd.concat([X_tr, y_tr], axis=1)
    maj = df[df.Class == 0]
    mn  = df[df.Class == 1]
    maj_d = resample(maj, replace=False,
                     n_samples=min(len(maj), len(mn) * 10), random_state=42)
    bal = pd.concat([maj_d, mn])
    return bal.drop("Class", axis=1), bal["Class"]

def train_model(df, model_type, balance, ts, nt):
    sc = StandardScaler()
    df = df.copy()
    df["Amount"] = sc.fit_transform(df[["Amount"]])
    df = df.drop(columns=["Time"], errors="ignore")
    X, y = df.drop("Class", axis=1), df["Class"]
    feats = X.columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=ts, random_state=42, stratify=y)

    cw = "balanced" if balance == "Class Weights" else None
    if balance == "Undersampling":
        X_tr, y_tr = undersample(X_tr, y_tr)

    if model_type == "Random Forest":
        mdl = RandomForestClassifier(n_estimators=nt, random_state=42,
                                     class_weight=cw, n_jobs=-1)
    else:
        mdl = LogisticRegression(max_iter=1000, random_state=42, class_weight=cw)

    mdl.fit(X_tr, y_tr)
    yp  = mdl.predict(X_te)
    ypr = mdl.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, ypr)

    result = dict(
        acc   = accuracy_score(y_te, yp),
        prec  = precision_score(y_te, yp),
        rec   = recall_score(y_te, yp),
        f1    = f1_score(y_te, yp),
        auc   = roc_auc_score(y_te, ypr),
        cm    = confusion_matrix(y_te, yp),
        fpr   = fpr, tpr=tpr,
        report= classification_report(y_te, yp, output_dict=True),
    )
    if model_type == "Random Forest":
        result["fi"] = pd.Series(
            mdl.feature_importances_, index=X.columns
        ).sort_values(ascending=False).head(15)

    joblib.dump(mdl,   MODEL_PATH)
    joblib.dump(sc,    SCALER_PATH)
    joblib.dump(feats, FEATURE_PATH)
    return result

# ── Train ─────────────────────────────────────────────────────────────────────
if train_btn and uploaded is not None:
    df_raw = load_data(uploaded)
    with st.spinner("Training model — please wait (30–60 sec) ⏳"):
        res = train_model(df_raw, model_choice, balance_method, test_size, n_trees)
    st.session_state["results"] = res
    st.session_state["df"]      = df_raw
    st.success("Model trained and saved!", icon="✅")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_eda, tab_perf, tab_pred = st.tabs([
    "📊  Data Overview",
    "📈  Model Performance",
    "🔍  Predict a Transaction",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    if uploaded is None:
        st.info("Upload **creditcard.csv** from the sidebar to see the data overview.")
    else:
        df = load_data(uploaded)

        st.subheader("Dataset summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total transactions", f"{len(df):,}")
        c2.metric("Fraud cases",        f"{df['Class'].sum():,}")
        c3.metric("Normal cases",       f"{(df['Class']==0).sum():,}")
        c4.metric("Fraud rate",         f"{df['Class'].mean()*100:.2f}%")

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Class distribution")
            counts = df["Class"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(["Normal (0)", "Fraud (1)"], counts.values,
                   color=["#3b82f6", "#ef4444"], width=0.5, edgecolor="none")
            ax.set_ylabel("Count", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col_b:
            st.subheader("Amount distribution — fraud vs normal")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.hist(df[df.Class == 0]["Amount"], bins=60, alpha=0.6,
                     color="#3b82f6", label="Normal", density=True)
            ax2.hist(df[df.Class == 1]["Amount"], bins=60, alpha=0.8,
                     color="#ef4444", label="Fraud",  density=True)
            ax2.set_xlabel("Amount (USD)", fontsize=10)
            ax2.set_ylabel("Density",      fontsize=10)
            ax2.legend(fontsize=9)
            ax2.spines[["top", "right"]].set_visible(False)
            ax2.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig2); plt.close()

        st.divider()
        st.subheader("Top 10 features correlated with fraud")
        corr = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False).head(10)
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        ax3.barh(corr.index[::-1], corr.values[::-1],
                 color="#6366f1", edgecolor="none")
        ax3.set_xlabel("Absolute correlation with Class label", fontsize=10)
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig3); plt.close()

        with st.expander("Show raw data sample (first 20 rows)"):
            st.dataframe(df.head(20), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    if "results" not in st.session_state:
        st.info("Train the model using the sidebar button to view performance.")
    else:
        m = st.session_state["results"]

        st.subheader("Evaluation metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{m['acc']*100:.2f}%")
        c2.metric("Precision", f"{m['prec']*100:.2f}%")
        c3.metric("Recall",    f"{m['rec']*100:.2f}%")
        c4.metric("F1 Score",  f"{m['f1']*100:.2f}%")
        c5.metric("ROC-AUC",   f"{m['auc']:.4f}")

        st.caption(
            "💡 **Why ROC-AUC?** With only 0.17% fraud, "
            "a model saying 'all normal' gets 99.83% accuracy — but catches zero fraud. "
            "ROC-AUC measures real discrimination power regardless of class imbalance."
        )
        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Confusion matrix")
            fig4, ax4 = plt.subplots(figsize=(4, 3))
            sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal", "Fraud"],
                        yticklabels=["Normal", "Fraud"],
                        ax=ax4, annot_kws={"size": 11})
            ax4.set_xlabel("Predicted", fontsize=10)
            ax4.set_ylabel("Actual",    fontsize=10)
            ax4.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig4); plt.close()

        with col_b:
            st.subheader("ROC curve")
            fig5, ax5 = plt.subplots(figsize=(4, 3))
            ax5.plot(m["fpr"], m["tpr"], color="#6366f1", lw=2,
                     label=f"AUC = {m['auc']:.4f}")
            ax5.plot([0, 1], [0, 1], "--", color="#aaa", lw=1)
            ax5.set_xlabel("False Positive Rate", fontsize=10)
            ax5.set_ylabel("True Positive Rate",  fontsize=10)
            ax5.legend(fontsize=9)
            ax5.spines[["top", "right"]].set_visible(False)
            ax5.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig5); plt.close()

        if "fi" in m:
            st.divider()
            st.subheader("Top 15 important features (Random Forest)")
            fig6, ax6 = plt.subplots(figsize=(8, 3.5))
            m["fi"][::-1].plot(kind="barh", ax=ax6,
                               color="#6366f1", edgecolor="none")
            ax6.set_xlabel("Importance score", fontsize=10)
            ax6.spines[["top", "right"]].set_visible(False)
            ax6.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig6); plt.close()

        with st.expander("Full classification report"):
            rdf = pd.DataFrame(m["report"]).transpose().round(3)
            st.dataframe(rdf, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Predict
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    model_ready = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)

    if not model_ready:
        st.info("Train the model first (sidebar button), then return here to predict.")
    else:
        mdl   = joblib.load(MODEL_PATH)
        sc    = joblib.load(SCALER_PATH)
        feats = joblib.load(FEATURE_PATH)

        st.subheader("Enter transaction details")
        method = st.radio("Input method", ["Paste raw values", "Enter manually"],
                          horizontal=True)

        input_data = None

        # ── Option A: paste ────────────────────────────────────────────────
        if method == "Paste raw values":
            st.caption("Paste **29 comma-separated numbers**: V1, V2, … V28, Amount")
            raw = st.text_area("Values", height=80,
                placeholder="-1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396, "
                            "0.0987, 0.3637, 0.0907, -0.5516, -0.6178, -0.9913, -0.3111, "
                            "1.4681, -0.4704, 0.2079, 0.0257, 0.4039, 0.2514, -0.0183, "
                            "0.2779, -0.1105, 0.0669, 0.1285, -0.1891, 0.1335, -0.0210, 149.62")
            if raw.strip():
                try:
                    vals = [float(x.strip()) for x in raw.split(",")]
                    if len(vals) == 29:
                        keys       = [f"V{i}" for i in range(1, 29)] + ["Amount"]
                        input_data = dict(zip(keys, vals))
                    else:
                        st.error(f"Expected 29 values, got {len(vals)}.")
                except Exception:
                    st.error("Could not parse — ensure all values are numbers separated by commas.")

        # ── Option B: manual ───────────────────────────────────────────────
        else:
            amount = st.number_input("Transaction Amount (USD)",
                                     min_value=0.0, max_value=50000.0,
                                     value=149.62, step=0.01)
            st.caption("V1–V28 are PCA-anonymised. Enter 0 if you don't have the values.")
            cols   = st.columns(7)
            v_vals = {}
            for i in range(1, 29):
                v_vals[f"V{i}"] = cols[(i - 1) % 7].number_input(
                    f"V{i}", value=0.0, format="%.3f", key=f"vi{i}")
            input_data = {**v_vals, "Amount": amount}

        st.divider()
        run_btn = st.button("Run prediction →", type="primary")

        if run_btn and input_data:
            idf          = pd.DataFrame([input_data])
            idf["Amount"] = sc.transform(idf[["Amount"]])
            idf          = idf[[c for c in feats if c in idf.columns]]

            pred = mdl.predict(idf)[0]
            prob = mdl.predict_proba(idf)[0]

            if pred == 1:
                st.markdown(f"""
                <div class="fraud-box">
                  <p class="fraud-title">⚠️  Fraudulent transaction detected</p>
                  <p class="sub-text">Fraud probability: <b>{prob[1]*100:.1f}%</b></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-box">
                  <p class="safe-title">✅  Legitimate transaction</p>
                  <p class="sub-text">Normal probability: <b>{prob[0]*100:.1f}%</b></p>
                </div>""", unsafe_allow_html=True)

            st.write("")
            c1, c2 = st.columns(2)
            c1.metric("Normal probability", f"{prob[0]*100:.2f}%")
            c2.metric("Fraud probability",  f"{prob[1]*100:.2f}%")

            fig7, ax7 = plt.subplots(figsize=(5, 1.4))
            ax7.barh(["Normal", "Fraud"], [prob[0], prob[1]],
                     color=["#3b82f6", "#ef4444"], edgecolor="none", height=0.4)
            ax7.set_xlim(0, 1)
            ax7.set_xlabel("Probability", fontsize=9)
            ax7.spines[["top", "right"]].set_visible(False)
            ax7.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig7); plt.close()
