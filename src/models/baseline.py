# src/models/baseline.py
"""
Baseline classification model with optional SHAP feature importance.
Logistic Regression (default) is fast and interpretable for quant research.
XGBoost is available as a premium option when installed.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Extended feature set including technical momentum signals
FEATURE_SET = [
    # Macro: rates & curve
    "term_spread", "t10y2y", "fedfunds",
    # Macro: economy
    "unemp", "d_unemp", "indpro_yoy",
    # Macro: inflation
    "cpi_yoy", "cpi_mom", "infl_exp",
    # Macro: credit
    "cred_spread",
    # Price-based
    "ret_w", "rv_w",
    # Technical momentum
    "rsi_14", "mom_4w", "mom_12w", "price_sma52", "rv_zscore",
    # Regime label (from HMM)
    "regime",
]


def make_baseline(model_type: str = "logistic") -> Pipeline:
    """
    Build a sklearn Pipeline.
    model_type: 'logistic' (default, fast, interpretable) | 'xgboost'
    """
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            return Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("clf", clf)])
        except ImportError:
            warnings.warn("xgboost not installed, falling back to logistic regression.")

    clf = LogisticRegression(
        C=0.5,
        max_iter=500,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=None,
        random_state=42,
    )
    return Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("clf", clf)])


def fit_predict(pipe: Pipeline, X_train, y_train, X_test):
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def evaluate_cls(y_true, proba) -> dict:
    try:
        roc = roc_auc_score(y_true, proba)
    except ValueError:
        roc = np.nan
    acc = accuracy_score(y_true, (proba >= 0.5).astype(int))
    return {"auc": roc, "acc": acc}


def get_feature_importance(pipe: Pipeline, feature_names: list[str]) -> pd.Series:
    """
    Extract feature importance / coefficients from the fitted pipeline.
    Works for LogisticRegression (coefficients) and XGBoost (feature_importances_).
    Returns a Series sorted by absolute importance (descending).
    """
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        # Logistic Regression — use scaled coefficients as importance
        importance = np.abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    else:
        importance = np.ones(len(feature_names))

    s = pd.Series(importance, index=feature_names, name="importance")
    return s.sort_values(ascending=False)


def compute_shap_values(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame | None:
    """
    Compute SHAP values for the model in the pipeline.
    Returns a DataFrame of SHAP values (same shape as X), or None if shap not available.
    """
    try:
        import shap
        clf = pipe.named_steps["clf"]
        scaler = pipe.named_steps["scaler"]
        X_scaled = scaler.transform(X)

        if hasattr(clf, "coef_"):
            explainer = shap.LinearExplainer(clf, X_scaled, feature_perturbation="correlation_dependent")
        else:
            explainer = shap.TreeExplainer(clf)

        shap_vals = explainer.shap_values(X_scaled)
        # For binary classifiers, shap_values may return a list [class0, class1]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        return pd.DataFrame(shap_vals, columns=X.columns, index=X.index)
    except Exception:
        return None
