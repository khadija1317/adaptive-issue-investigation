def _fn(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": required or []}}}

SPLIT = {"type": "string", "enum": ["train", "val"]}
TOPK = {"type": "integer", "minimum": 1, "maximum": 10}

TOOL_SCHEMAS = [
    _fn("get_metrics", "Accuracy, macro-F1, macro-recall and model size for one split.", {"split": SPLIT}, ["split"]),
    _fn("get_class_distribution", "Class counts and proportions for one split.", {"split": SPLIT}, ["split"]),
    _fn("compare_train_validation", "Train-minus-validation gap per metric and per-class recall on both splits."),
    _fn("check_leakage", "Top-k features by |correlation with target| in train, with val correlation alongside; shared duplicate rows.", {"top_k": TOPK}),
    _fn("inspect_feature_distribution", "Train vs val per feature: mean, std, standardized mean difference, KS, std ratio; summary over all features.", {"top_k": TOPK}),
    _fn("analyze_confusion_matrix", "Confusion matrix with per-class precision and recall for one split.", {"split": SPLIT}, ["split"]),
    _fn("run_experiment",
        "Re-run training with changes applied to the ORIGINAL config (not cumulative). Max 3 per investigation. Returns train/val metrics and model size.",
        {"changes": {"type": "object", "additionalProperties": False, "properties": {
            "max_depth": {"type": "integer", "minimum": 1},
            "C": {"type": "number", "exclusiveMinimum": 0},
            "train_fraction": {"type": "number", "description": "fraction of the available training pool to use (1.0 = all)"},
            "class_weight": {"type": "string", "enum": ["balanced"]},
            "drop_features": {"type": "array", "items": {"type": "string"}},
            "scaling": {"type": "string", "enum": ["none", "standard", "per_split"],
                        "description": "standard = statistics fitted on train; per_split = each split normalized with its own statistics"},
            "split_strategy": {"type": "string", "enum": ["random", "by_size"]}}}},
        ["changes"]),
]