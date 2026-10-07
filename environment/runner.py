import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from .schema import ExperimentConfig, ExperimentRecord

FORBIDDEN = ("ground_truth", "decoy", "evidence", "make_imbalanced",
             "add_leaky_feature", "corrupt_scaling", "inject")

def assert_sanitized(rec):
    # regression test; the real protection is that Scenario/EVAL_META never enter the record
    assert set(rec.X) == {"train", "val"} and set(rec.y) == {"train", "val"}, "test leaked"
    visible = repr(rec.config) + repr(rec.feature_names) + repr(rec.metrics)
    bad = [w for w in FORBIDDEN if w in visible]
    assert not bad, f"forbidden words in record: {bad}"

def _load(name):
    d = load_breast_cancer() if name == "breast_cancer" else load_wine()
    return d.data.astype(float), d.target, list(d.feature_names)

def _split(X, y, strategy, seed):
    idx = np.arange(len(y))
    # test is carved FIRST, random and fixed by seed, so no knob can change it
    rest, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    if strategy == "random":
        tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=seed)
    elif strategy == "by_size":      # smallest 75% of tumors train, largest 25% validate
        order = rest[np.argsort(X[rest, 0])]          # column 0 = mean radius
        cut = int(0.75 * len(order))
        tr, va = order[:cut], order[cut:]
    else:
        raise ValueError(f"unknown split_strategy: {strategy}")
    return {"train": tr, "val": va, "test": test}

def _metrics(y, p):
    return {"accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"),
            "macro_recall": recall_score(y, p, average="macro")}

def run(scenario, config=None):
    cfg = config or scenario.base_config
    X, y, names = _load(scenario.dataset)
    rng_data, rng_inj = np.random.default_rng(cfg.seed), np.random.default_rng(cfg.seed + 1)
    if scenario.data_injector:
        X, y = scenario.data_injector(X, y, rng=rng_data)
    idx = _split(X, y, cfg.split_strategy, cfg.seed)
    splits = {k: (X[i], y[i]) for k, i in idx.items()}
    width = splits["train"][0].shape[1]
    if scenario.split_injector:
        splits = scenario.split_injector(splits, rng=rng_inj)
    if splits["train"][0].shape[1] > width:
        names = names + ["aux_score"]

    if cfg.train_fraction < 1.0:                       # "add/remove data" knob
        Xt, yt = splits["train"]
        keep = np.random.default_rng(cfg.seed + 2).permutation(len(yt))[:int(round(cfg.train_fraction * len(yt)))]
        splits["train"] = (Xt[keep], yt[keep])

    unknown = set(cfg.drop_features) - set(names)
    if unknown:
        raise ValueError(f"unknown features: {sorted(unknown)}")
    cols = [i for i, n in enumerate(names) if n not in cfg.drop_features]
    names = [names[i] for i in cols]
    splits = {k: (Xk[:, cols], yk) for k, (Xk, yk) in splits.items()}

    if cfg.scaling == "standard":      # legitimate: statistics fitted on train, applied to every split
        mu, sd = splits["train"][0].mean(0), splits["train"][0].std(0) + 1e-9
        splits = {k: ((Xk - mu) / sd, yk) for k, (Xk, yk) in splits.items()}
    elif cfg.scaling == "per_split":   # diagnostic only: each split uses its OWN statistics
        splits = {k: ((Xk - Xk.mean(0)) / (Xk.std(0) + 1e-9), yk) for k, (Xk, yk) in splits.items()}
    elif cfg.scaling != "none":
        raise ValueError(f"unknown scaling: {cfg.scaling}")

    if cfg.model == "tree":
        m = DecisionTreeClassifier(max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
                                   class_weight=cfg.class_weight, random_state=cfg.seed)
    else:
        m = LogisticRegression(C=cfg.C, class_weight=cfg.class_weight, max_iter=2000)
    m.fit(*splits["train"])

    pred = {k: m.predict(splits[k][0]) for k in ("train", "val")}
    rec = ExperimentRecord(
        config=cfg, feature_names=names,
        X={k: splits[k][0] for k in ("train", "val")},
        y={k: splits[k][1] for k in ("train", "val")},
        pred=pred, model=m,
        metrics={k: _metrics(splits[k][1], pred[k]) for k in ("train", "val")})
    assert_sanitized(rec)
    test_metrics = _metrics(splits["test"][1], m.predict(splits["test"][0]))
    return rec, test_metrics        # test_metrics goes to evaluation code ONLY