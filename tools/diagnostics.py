import numpy as np
from dataclasses import replace
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from environment.runner import run

SPLITS = ("train", "val")          # "test" is not requestable
TOOL_NAMES = ["get_metrics", "get_class_distribution", "compare_train_validation",
              "check_leakage", "inspect_feature_distribution",
              "analyze_confusion_matrix", "run_experiment"]

# --- helpers: uniform output format, rounding, error shape ---
def _r(x):
    if isinstance(x, dict): return {str(k): _r(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_r(v) for v in x]
    if isinstance(x, np.ndarray): return _r(x.tolist())
    if isinstance(x, (float, np.floating)): return round(float(x), 3)
    if isinstance(x, np.integer): return int(x)
    return x

def _wrap(tool, args, result): return {"tool": tool, "args": args, "result": _r(result)}
def _err(tool, args, msg): return _wrap(tool, args, {"error": msg})

def _size(model):   # lets the agent see whether a capacity change actually changed the model
    if hasattr(model, "get_depth"):
        return {"type": "tree", "depth": model.get_depth(), "leaves": model.get_n_leaves()}
    return {"type": "logreg", "C": model.C}

def _corr(X, y):    # Pearson corr of every column with the target, vectorised
    Xc, yc = X - X.mean(0), y - y.mean()
    return (Xc * yc[:, None]).sum(0) / (np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12)

class Session:
    MAX_EXPERIMENTS = 3
    KNOBS = {"max_depth", "C", "train_fraction", "class_weight",
             "drop_features", "scaling", "split_strategy"}

    def __init__(self, scenario):
        self.scenario = scenario
        self.rec, self.baseline_test = run(scenario)   # baseline run, test scores kept for evaluation only
        self.history = []                              # [(changes, test_metrics)] evaluation only, never shown

    # What the agent is told up front (basics live here, not in a tool).
    def task_info(self):
        c, r = self.scenario.base_config, self.rec
        hp = {"max_depth": c.max_depth} if c.model == "tree" else {"C": c.C}
        return {"complaint": self.scenario.complaint, "dataset": self.scenario.dataset,
                "model": c.model, "hyperparameters": hp, "class_weight": c.class_weight,
                "scaling": c.scaling, "n_train": len(r.y["train"]), "n_val": len(r.y["val"]),
                "n_features": len(r.feature_names), "n_classes": len(np.unique(r.y["train"]))}

    def call(self, name, args=None):   # single entry point: handles unknown tools and bad args
        if name not in TOOL_NAMES:
            return _err(name, args, f"unknown tool; available: {TOOL_NAMES}")
        if args is not None and not isinstance(args, dict):
            return _err(name, args, "args must be a JSON object")
        try:
            return getattr(self, name)(**(args or {}))
        except TypeError as e:
            return _err(name, args, f"bad arguments: {e}")

    # 1. headline numbers for one split
    def get_metrics(self, split="val"):
        a = {"split": split}
        if split not in SPLITS: return _err("get_metrics", a, "split must be 'train' or 'val'")
        return _wrap("get_metrics", a, {**self.rec.metrics[split], "model": _size(self.rec.model)})

    # 2. label balance per split
    def get_class_distribution(self, split="val"):
        a = {"split": split}
        if split not in SPLITS: return _err("get_class_distribution", a, "split must be 'train' or 'val'")
        c = np.bincount(self.rec.y[split])
        return _wrap("get_class_distribution", a, {"counts": c, "proportions": c / c.sum()})

    # 3. gap per metric + per-class recall on both splits
    def compare_train_validation(self):
        r = self.rec
        labels = np.unique(np.concatenate([r.y["train"], r.y["val"]]))
        out = {"gap_train_minus_val": {k: r.metrics["train"][k] - r.metrics["val"][k] for k in r.metrics["train"]}}
        for s in SPLITS:
            out[f"recall_per_class_{s}"] = dict(zip(labels.tolist(),
                recall_score(r.y[s], r.pred[s], labels=labels, average=None, zero_division=0)))
        return _wrap("compare_train_validation", {}, out)

    # 4. strongest feature-target correlations, train vs val side by side, + duplicate rows
    def check_leakage(self, top_k=5):
        a, r = {"top_k": top_k}, self.rec
        k = max(1, min(int(top_k), 10))
        ct, cv = _corr(r.X["train"], r.y["train"]), _corr(r.X["val"], r.y["val"])
        rows = [{"feature": r.feature_names[i], "corr_train": ct[i], "corr_val": cv[i]}
                for i in np.argsort(-np.abs(ct))[:k]]
        dup = len({x.tobytes() for x in r.X["train"]} & {x.tobytes() for x in r.X["val"]})
        return _wrap("check_leakage", a, {"top_features": rows, "shared_duplicate_rows": dup})

    # 5. per-feature train-vs-val comparison: mean, std, SMD, KS, std ratio, + all-feature summary
    def inspect_feature_distribution(self, top_k=5):
        a, r = {"top_k": top_k}, self.rec
        k = max(1, min(int(top_k), 10))
        Xt, Xv = r.X["train"], r.X["val"]
        st = Xt.std(0) + 1e-9
        smd, ratio = (Xv.mean(0) - Xt.mean(0)) / st, Xv.std(0) / st
        ks = np.array([ks_2samp(Xt[:, j], Xv[:, j]).statistic for j in range(Xt.shape[1])])
        rows = [{"feature": r.feature_names[i], "mean_train": Xt[:, i].mean(), "mean_val": Xv[:, i].mean(),
                 "std_train": Xt[:, i].std(), "std_val": Xv[:, i].std(),
                 "smd": smd[i], "ks": ks[i], "std_ratio": ratio[i]}
                for i in np.argsort(-np.abs(smd))[:k]]
        summary = {"mean_abs_smd_all": np.abs(smd).mean(), "median_std_ratio_all": np.median(ratio),
                   "n_features_abs_smd_gt_0.5": int((np.abs(smd) > 0.5).sum()), "n_features": len(smd)}
        return _wrap("inspect_feature_distribution", a, {"top_by_abs_smd": rows, "summary": summary})

    # 6. confusion matrix with per-class precision and recall
    def analyze_confusion_matrix(self, split="val"):
        a, r = {"split": split}, self.rec
        if split not in SPLITS: return _err("analyze_confusion_matrix", a, "split must be 'train' or 'val'")
        y, p = r.y[split], r.pred[split]
        labels = np.unique(np.concatenate([r.y["train"], r.y["val"]]))
        return _wrap("analyze_confusion_matrix", a, {
            "labels": labels, "matrix": confusion_matrix(y, p, labels=labels),
            "precision": precision_score(y, p, labels=labels, average=None, zero_division=0),
            "recall": recall_score(y, p, labels=labels, average=None, zero_division=0)})

    # 7. the only tool that changes anything: re-run the pipeline with changes, budget-capped
    def run_experiment(self, changes=None):
        a, base = {"changes": changes}, self.rec.config
        T = "run_experiment"
        if not isinstance(changes, dict) or not changes:
            return _err(T, a, f"changes must be a non-empty object; allowed knobs: {sorted(self.KNOBS)}")
        bad = set(changes) - self.KNOBS
        if bad: return _err(T, a, f"unknown knobs {sorted(bad)}; allowed: {sorted(self.KNOBS)}")
        if "max_depth" in changes and base.model != "tree": return _err(T, a, "max_depth applies to tree models only")
        if "C" in changes and base.model != "logreg": return _err(T, a, "C applies to logreg models only")
        tf = changes.get("train_fraction", 1.0)
        if not isinstance(tf, (int, float)) or not 0 < tf <= 1: return _err(T, a, "train_fraction must be in (0, 1]")
        if len(self.history) >= self.MAX_EXPERIMENTS:
            return _err(T, a, f"experiment budget used ({self.MAX_EXPERIMENTS}/{self.MAX_EXPERIMENTS})")
        ch = dict(changes)
        if "drop_features" in ch: ch["drop_features"] = tuple(ch["drop_features"])
        try:
            rec, test = run(self.scenario, replace(base, **ch))   # always baseline + changes, never stacked
        except (ValueError, TypeError) as e:
            return _err(T, a, str(e))
        self.history.append((changes, test))
        return _wrap(T, a, {"changed": changes, "train": rec.metrics["train"], "val": rec.metrics["val"],
                            "model": _size(rec.model),
                            "experiments_left": self.MAX_EXPERIMENTS - len(self.history)})