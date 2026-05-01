"""Trains all models and saves artifacts for the Streamlit dashboard."""
import pandas as pd
import numpy as np
import joblib, warnings
warnings.filterwarnings('ignore')

from scipy.stats import pointbiserialr
from sklearn.feature_selection import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, roc_curve, precision_recall_curve,
                              confusion_matrix)
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Load & preprocess ─────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel('Telco_customer_churn (1)-kaggle.xlsx')

NON_INFORMATIVE = ['CustomerID', 'Count', 'Country', 'State', 'City',
                   'Zip Code', 'Lat Long', 'Latitude', 'Longitude']
LEAKAGE = ['Churn Score', 'CLTV', 'Churn Reason', 'Churn Label']
df_clean = df.drop(columns=[c for c in NON_INFORMATIVE + LEAKAGE if c in df.columns])
df_clean['Total Charges'] = pd.to_numeric(df_clean['Total Charges'], errors='coerce').fillna(0)

y = df_clean['Churn Value']
X_raw = df_clean.drop(columns=['Churn Value'])

cat_cols = X_raw.select_dtypes(include='object').columns.tolist()
num_cols = X_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
bool_cols = X_encoded.select_dtypes(include='bool').columns
X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ── Feature selection ─────────────────────────────────────────────────────────
binary_cols = [c for c in X_encoded.columns if X_encoded[c].nunique() == 2]
_, chi2_pvals = chi2(X_train[binary_cols], y_train)
chi2_keep = [binary_cols[i] for i, p in enumerate(chi2_pvals) if p < 0.05]

pb_keep = [c for c in num_cols
           if pointbiserialr(X_train[c], y_train).pvalue < 0.05]

vif_cols = [c for c in pb_keep if c in X_train.columns]
if len(vif_cols) >= 2:
    vif_vals = [variance_inflation_factor(X_train[vif_cols].values, i)
                for i in range(len(vif_cols))]
    pb_keep = [c for c, v in zip(vif_cols, vif_vals) if v <= 10]

FINAL_FEATURES = list(dict.fromkeys(chi2_keep + pb_keep))
FINAL_FEATURES = [f for f in FINAL_FEATURES if f in X_encoded.columns]
X_train_f = X_train[FINAL_FEATURES]
X_test_f  = X_test[FINAL_FEATURES]

# ── Train models ──────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("Training Logistic Regression...")
lr_gs = GridSearchCV(
    ImbPipeline([('scaler', StandardScaler()),
                 ('clf', LogisticRegression(class_weight='balanced',
                                            max_iter=1000, random_state=RANDOM_STATE))]),
    {'clf__C': [0.01, 0.1, 1, 10]}, cv=cv, scoring='recall', n_jobs=-1)
lr_gs.fit(X_train_f, y_train)

print("Training Random Forest...")
rf_gs = GridSearchCV(
    ImbPipeline([('clf', RandomForestClassifier(class_weight='balanced',
                                                random_state=RANDOM_STATE))]),
    {'clf__n_estimators': [100, 200], 'clf__max_depth': [4, 8, None]},
    cv=cv, scoring='recall', n_jobs=-1)
rf_gs.fit(X_train_f, y_train)

print("Training Gradient Boosting...")
gb_gs = GridSearchCV(
    ImbPipeline([('clf', GradientBoostingClassifier(random_state=RANDOM_STATE))]),
    {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.05, 0.1],
     'clf__max_depth': [3, 5]},
    cv=cv, scoring='recall', n_jobs=-1)
gb_gs.fit(X_train_f, y_train)

models = {
    'Logistic Regression': lr_gs.best_estimator_,
    'Random Forest':       rf_gs.best_estimator_,
    'Gradient Boosting':   gb_gs.best_estimator_,
}

# ── Evaluation results ────────────────────────────────────────────────────────
results = {}
for name, model in models.items():
    y_pred  = model.predict(X_test_f)
    y_proba = model.predict_proba(X_test_f)[:, 1]
    fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
    prec, rec, pr_thresh  = precision_recall_curve(y_test, y_proba)
    results[name] = {
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'auc_roc':   roc_auc_score(y_test, y_proba),
        'cm':        confusion_matrix(y_test, y_pred),
        'fpr': fpr, 'tpr': tpr,
        'prec_curve': prec, 'rec_curve': rec,
    }

# ── SHAP (Random Forest) ──────────────────────────────────────────────────────
print("Computing SHAP values...")
rf_clf  = rf_gs.best_estimator_.named_steps['clf']
sv_raw  = shap.TreeExplainer(rf_clf).shap_values(X_test_f.values)
if isinstance(sv_raw, list):
    sv = sv_raw[1]
elif hasattr(sv_raw, 'ndim') and sv_raw.ndim == 3:
    sv = sv_raw[:, :, 1]
else:
    sv = sv_raw

mean_shap = np.abs(sv).mean(axis=0)
shap_df = pd.DataFrame({'feature': FINAL_FEATURES,
                         'mean_abs_shap': mean_shap}).sort_values(
    'mean_abs_shap', ascending=False)

lr_coef_df = pd.DataFrame({
    'feature': FINAL_FEATURES,
    'coefficient': lr_gs.best_estimator_.named_steps['clf'].coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

rf_fi_df = pd.DataFrame({
    'feature': FINAL_FEATURES,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

# ── Save ──────────────────────────────────────────────────────────────────────
print("Saving artifacts...")
joblib.dump({
    'models':      models,
    'results':     results,
    'features':    FINAL_FEATURES,
    'X_test':      X_test_f,
    'y_test':      y_test,
    'shap_values': sv,
    'shap_df':     shap_df,
    'lr_coef_df':  lr_coef_df,
    'rf_fi_df':    rf_fi_df,
    'df_clean':    df_clean,
}, 'dashboard_artifacts.pkl')

print("Done — dashboard_artifacts.pkl saved.")
