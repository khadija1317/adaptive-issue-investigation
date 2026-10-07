import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def make_s6(sigma, seed=0):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    rng = np.random.default_rng(seed)
    X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_te.copy()
    X_tr["aux_score"] = y_tr.values + rng.normal(0, sigma, len(y_tr))                      # leak: label-derived, train only
    X_val["aux_score"] = rng.permutation(y_val.values + rng.normal(0, sigma, len(y_val)))  # same values, relation destroyed
    X_te["aux_score"] = rng.permutation(y_te.values + rng.normal(0, sigma, len(y_te)))
    return X_tr, y_tr, X_val, y_val, X_te, y_te

def fit_eval(data, drop=(), **model_kwargs):
    X_tr, y_tr, X_val, y_val, X_te, y_te = data
    cols = [c for c in X_tr.columns if c not in drop]
    m = RandomForestClassifier(n_estimators=100, random_state=0, **model_kwargs).fit(X_tr[cols], y_tr)
    return [round(accuracy_score(y, m.predict(X[cols])), 3)
            for X, y in [(X_tr, y_tr), (X_val, y_val), (X_te, y_te)]]

for sigma in [0.0, 0.3, 0.6, 1.0]:
    d = make_s6(sigma)
    print(f"\nsigma={sigma}   (train, val, test accuracy)")
    print("  baseline          ", fit_eval(d))
    print("  regularized       ", fit_eval(d, max_depth=3, min_samples_leaf=5))
    print("  drop aux_score    ", fit_eval(d, drop=["aux_score"]))
    d = make_s6(sigma)
    print("  corr(aux, y) train:", round(np.corrcoef(d[0]["aux_score"], d[1])[0, 1], 3),
          " val:", round(np.corrcoef(d[2]["aux_score"], d[3])[0, 1], 3))