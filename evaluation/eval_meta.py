EVAL_META = {
 "S1": {"ground_truth": "class_imbalance",    "decoy": None,                   "difficulty": "easy"},
 "S2": {"ground_truth": "overfitting",        "decoy": None,                   "difficulty": "easy"},
 "S3": {"ground_truth": "data_leakage",       "decoy": None,                   "difficulty": "medium"},
 "S4": {"ground_truth": "bad_preprocessing",  "decoy": "distribution_shift",   "difficulty": "medium"},
 "S5": {"ground_truth": "distribution_shift", "decoy": "overfitting",          "difficulty": "medium"},
 "S6": {"ground_truth": "data_leakage",       "decoy": "overfitting",          "difficulty": "hard"},
}  # evidence sets get added on Day 6

def X(when, val=None):   # experiment atom: which changes count, and which val outcome makes it evidence
    return {"tool": "run_experiment", "when": when, "val": val}
def T(tool):             # plain tool-call atom
    return {"tool": tool, "when": None, "val": None}

# val outcomes: "up", "down", "moved" (differs from baseline), "no_gain" (not up AND model size changed)
drop_aux      = lambda c: "aux_score" in c.get("drop_features", ())
more_data     = lambda c: c.get("train_fraction", 0.15) > 0.15      # S2 base is 0.15
balanced      = lambda c: c.get("class_weight") == "balanced"
per_split     = lambda c: c.get("scaling") == "per_split"
rand_split    = lambda c: c.get("split_strategy") == "random"
less_capacity = lambda c: c.get("min_samples_leaf", 1) > 1 or c.get("max_depth") is not None

_S4 = [[T("inspect_feature_distribution")], [X(per_split, "up")]]
EVIDENCE = {
 "S1": {"true": [[T("get_class_distribution"), T("analyze_confusion_matrix")], [X(balanced, "up")]], "decoy": []},
 "S2": {"true": [[X(more_data, "up")]], "decoy": []},
 "S3": {"true": [[T("check_leakage"), X(drop_aux, "moved")]], "decoy": []},
 "S4": {"true": _S4, "decoy": _S4},
 "S5": {"true": [[T("get_class_distribution")], [T("inspect_feature_distribution")],
                 [X(rand_split, "up")], [X(per_split, "down")]],
        "decoy": [[X(rand_split, "up")], [X(less_capacity, "no_gain")]]},
 "S6": {"true": [[T("check_leakage"), X(drop_aux, "up")]], "decoy": [[X(less_capacity, "no_gain")]]},
}
for _sid, _e in EVIDENCE.items():
    EVAL_META[_sid]["evidence_true"], EVAL_META[_sid]["evidence_decoy"] = _e["true"], _e["decoy"]


FIX = {   # (kind, metric, reference fix); the reference is the scenario's designed repair
 "S1": ("gain",  "macro_recall", {"class_weight": "balanced"}),
 "S2": ("gain",  "accuracy",     {"train_fraction": 1.0}),
 "S3": ("gain",  "accuracy",     {"drop_features": ["aux_score"]}),
 "S4": ("agree", "accuracy",     {"scaling": "per_split"}),
 "S5": ("agree", "accuracy",     {"split_strategy": "random"}),
 "S6": ("gain",  "accuracy",     {"drop_features": ["aux_score"]}),
}
for _sid, (_k, _m, _r) in FIX.items():
    EVAL_META[_sid]["fix"] = {"kind": _k, "metric": _m, "reference": _r}