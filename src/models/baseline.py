import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURE_SET = ["term_spread","t10y2y","fedfunds","unemp","d_unemp","indpro_yoy","cpi_yoy","cpi_mom","infl_exp","cred_spread","ret_w","rv_w","regime"]

def make_baseline():
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=200, n_jobs=None))
    ])
    return pipe

def fit_predict(pipe, X_train, y_train, X_test):
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:,1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba

def evaluate_cls(y_true, proba):
    # AUC is robust for probabilistic outputs
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = np.nan
    acc = accuracy_score(y_true, (proba>=0.5).astype(int))
    return {"auc": auc, "acc": acc}
