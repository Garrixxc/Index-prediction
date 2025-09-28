# src/models/regimes.py
import ast
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def _to_feature_list(fc):
    if isinstance(fc, (list, tuple)): return list(fc)
    if isinstance(fc, str):
        s = fc.strip()
        if s.startswith("["):
            try:
                v = ast.literal_eval(s)
                return list(v) if isinstance(v, (list, tuple)) else [str(v)]
            except Exception:
                return [s]
        return [s]
    return ["ret_w", "rv_w"]

def fit_hmm(df, n_states=3, covariance_type="full", feature_cols=("ret_w","rv_w")):
    cols = _to_feature_list(feature_cols)
    X = df[cols].dropna()
    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                        n_iter=500, random_state=42)
    model.fit(X.values)
    states = model.predict(X.values)
    return model, pd.Series(states, index=X.index, name="regime")

def align_regimes(full_index, regimes):
    return regimes.reindex(full_index).ffill().bfill().astype(int)
