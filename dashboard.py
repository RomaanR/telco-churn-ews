"""Streamlit dashboard — Telco Customer Churn Early Warning System"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              precision_recall_curve, roc_curve, auc,
                              recall_score, precision_score, f1_score)

st.set_page_config(
    page_title="Telco Churn Early Warning System",
    page_icon="📡",
    layout="wide",
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    return joblib.load('dashboard_artifacts.pkl')

try:
    art = load_artifacts()
except FileNotFoundError:
    st.error("dashboard_artifacts.pkl not found. Run `python train_and_save.py` first.")
    st.stop()

models   = art['models']
results  = art['results']
features = art['features']
X_test   = art['X_test']
y_test   = art['y_test']
sv       = art['shap_values']
shap_df  = art['shap_df']
lr_coef  = art['lr_coef_df']
rf_fi    = art['rf_fi_df']
df_clean = art['df_clean']

MODEL_NAMES = list(models.keys())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Churn EWS")
    st.markdown("**CS 439 · Pranjal · Shlok · Romaan**")
    st.divider()

    selected_model = st.selectbox("Active model", MODEL_NAMES, index=1)

    st.markdown("**Decision threshold**")
    threshold = st.slider(
        "Probability cutoff (default 0.5)",
        min_value=0.01, max_value=0.99, value=0.50, step=0.01,
        help="Lower = catch more churners (higher recall, lower precision)"
    )

    y_proba   = results[selected_model]['y_proba']
    y_pred_t  = (y_proba >= threshold).astype(int)
    rec_t  = recall_score(y_test, y_pred_t)
    prec_t = precision_score(y_test, y_pred_t, zero_division=0)
    f1_t   = f1_score(y_test, y_pred_t, zero_division=0)

    st.markdown(f"""
    **At threshold {threshold:.2f}**
    - Recall: `{rec_t:.3f}`
    - Precision: `{prec_t:.3f}`
    - F1: `{f1_t:.3f}`
    """)
    st.divider()
    st.caption("FN cost >> FP cost → keep threshold low to catch churners")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Telco Customer Churn — Early Warning System")
st.markdown("Predicting at-risk customers and prioritising retention · IBM Telco Dataset (7,043 customers)")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🎯 Model Performance",
    "🔍 Feature Insights",
    "⚠️ At-Risk Customers",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset & Model Summary")

    # KPI cards
    best_recall  = max(r['recall']  for r in results.values())
    best_auc     = max(r['auc_roc'] for r in results.values())
    churn_rate   = y_test.mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",  "7,043")
    c2.metric("Churn Rate",        f"{churn_rate:.1%}")
    c3.metric("Best Recall",       f"{best_recall:.3f}",  help="Across all three models on test set")
    c4.metric("Best AUC-ROC",      f"{best_auc:.3f}",     help="Across all three models on test set")

    st.divider()
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("**Churn Distribution**")
        fig, ax = plt.subplots(figsize=(4, 4))
        counts = y_test.value_counts()
        ax.pie(counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
               colors=['#4e8df5', '#f55f4e'], startangle=90)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Model Comparison (Test Set, default threshold 0.5)**")
        summary = pd.DataFrame({
            name: {
                'Precision': f"{r['precision']:.3f}",
                'Recall':    f"{r['recall']:.3f}",
                'F1':        f"{r['f1']:.3f}",
                'AUC-ROC':   f"{r['auc_roc']:.3f}",
            }
            for name, r in results.items()
        }).T
        st.dataframe(summary, use_container_width=True)

        st.markdown("""
        **Takeaways**
        - **Random Forest** achieves the highest recall (0.826) — catches the most churners
        - **Gradient Boosting** achieves the highest AUC-ROC (0.853)
        - **Logistic Regression** is the most interpretable via its coefficients
        - All three far exceed the dummy baseline (0% churn recall)
        """)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Model Performance — {selected_model}")
    st.caption(f"Using threshold = {threshold:.2f}  (adjust in sidebar)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm = confusion_matrix(y_test, y_pred_t)
        disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f"{selected_model} (threshold={threshold:.2f})")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Precision vs Recall by Threshold**")
        prec_c, rec_c, thresh_c = precision_recall_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.plot(thresh_c, prec_c[:-1], label='Precision', lw=2, color='#4e8df5')
        ax.plot(thresh_c, rec_c[:-1],  label='Recall',    lw=2, color='#f55f4e')
        ax.axvline(threshold, color='gray', linestyle='--', lw=1,
                   label=f'Current ({threshold:.2f})')
        ax.set_xlabel('Threshold')
        ax.set_title('Precision / Recall Trade-off')
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**ROC Curves — All Models**")
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ['#4e8df5', '#f5a623', '#f55f4e']
        for (name, r), color in zip(results.items(), colors):
            ax.plot(r['fpr'], r['tpr'], lw=2, color=color,
                    label=f"{name} (AUC={r['auc_roc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("**Precision-Recall Curves — All Models**")
        baseline_pr = y_test.mean()
        fig, ax = plt.subplots(figsize=(5, 4))
        for (name, r), color in zip(results.items(), colors):
            pr_auc = auc(r['rec_curve'], r['prec_curve'])
            ax.plot(r['rec_curve'], r['prec_curve'], lw=2, color=color,
                    label=f"{name} (AUC={pr_auc:.3f})")
        ax.axhline(baseline_pr, color='k', linestyle='--', lw=1,
                   label=f'Random ({baseline_pr:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("What Drives Churn?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 15 Features by SHAP Impact (Random Forest)**")
        top_shap = shap_df.head(15).sort_values('mean_abs_shap')
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.barh(top_shap['feature'], top_shap['mean_abs_shap'],
                       color='#4e8df5')
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('Mean Absolute SHAP Impact')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Logistic Regression Coefficients (Top 15)**")
        top_coef = lr_coef.head(15).copy()
        top_coef = top_coef.sort_values('coefficient')
        colors_lr = ['#f55f4e' if c > 0 else '#4e8df5'
                     for c in top_coef['coefficient']]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top_coef['feature'], top_coef['coefficient'], color=colors_lr)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('Coefficient')
        ax.set_title('LR Coefficients\n(Red = increases churn risk)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.markdown("**Top 5 Churn Drivers — Plain English**")
    c1, c2, c3, c4, c5 = st.columns(5)
    drivers = [
        ("📄 Month-to-month contract", "Customers without a long-term contract churn at far higher rates."),
        ("⏱️ Short tenure", "New customers (< 12 months) are most at risk — loyalty hasn't built yet."),
        ("🌐 Fiber Optic internet", "Fiber customers churn more despite higher charges, suggesting unmet expectations."),
        ("👨‍👩‍👧 No dependents", "Customers without dependents have fewer reasons to stay."),
        ("💳 Electronic check", "This payment method correlates strongly with higher churn risk."),
    ]
    for col, (title, desc) in zip([c1, c2, c3, c4, c5], drivers):
        col.info(f"**{title}**\n\n{desc}")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — AT-RISK CUSTOMERS
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"At-Risk Customer List — {selected_model}")
    st.caption(f"Customers flagged as churn risk at threshold = {threshold:.2f}")

    y_proba_all = results[selected_model]['y_proba']
    risk_df = X_test.copy()
    risk_df['Churn Probability'] = np.round(y_proba_all, 4)
    risk_df['Actual Churn']      = y_test.values
    risk_df['Predicted Churn']   = (y_proba_all >= threshold).astype(int)
    risk_df['Risk Level'] = pd.cut(
        risk_df['Churn Probability'],
        bins=[0, 0.4, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    risk_df = risk_df.sort_values('Churn Probability', ascending=False)

    flagged = risk_df[risk_df['Predicted Churn'] == 1]
    total_flagged = len(flagged)
    true_churners = (flagged['Actual Churn'] == 1).sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers flagged",     total_flagged)
    m2.metric("True churners caught",  true_churners,
              help="Of the flagged customers who actually churned (on test set)")
    m3.metric("Missed churners",
              int((y_test == 1).sum()) - true_churners,
              delta_color="inverse")

    st.divider()

    risk_level_filter = st.multiselect(
        "Filter by risk level",
        ['Critical', 'High', 'Medium', 'Low'],
        default=['Critical', 'High']
    )

    display_cols = ['Churn Probability', 'Risk Level', 'Actual Churn',
                    'Tenure Months', 'Monthly Charges', 'Total Charges']
    display_cols = [c for c in display_cols if c in risk_df.columns]

    filtered = risk_df[risk_df['Risk Level'].isin(risk_level_filter)][display_cols]

    def colour_risk(val):
        colours = {'Critical': 'background-color: #ffcccc',
                   'High':     'background-color: #ffe5cc',
                   'Medium':   'background-color: #fff5cc',
                   'Low':      ''}
        return colours.get(val, '')

    st.dataframe(
        filtered.style.applymap(colour_risk, subset=['Risk Level']).format(
            {'Churn Probability': '{:.3f}'}),
        use_container_width=True,
        height=400,
    )

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download at-risk list as CSV",
        data=csv,
        file_name=f"at_risk_customers_{selected_model.replace(' ','_')}.csv",
        mime='text/csv',
    )

    st.divider()
    st.markdown("**Risk Level Distribution**")
    risk_counts = risk_df['Risk Level'].value_counts().reindex(
        ['Critical', 'High', 'Medium', 'Low'])
    fig, ax = plt.subplots(figsize=(6, 3))
    bar_colors = ['#d73027', '#f46d43', '#fee090', '#74add1']
    ax.bar(risk_counts.index, risk_counts.values, color=bar_colors)
    ax.set_ylabel('Number of customers')
    ax.set_title('Customer Risk Distribution (Test Set)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
