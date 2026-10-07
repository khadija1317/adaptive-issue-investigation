EVAL_META = {
 "S1": {"ground_truth": "class_imbalance",    "decoy": None,                   "difficulty": "easy"},
 "S2": {"ground_truth": "overfitting",        "decoy": None,                   "difficulty": "easy"},
 "S3": {"ground_truth": "data_leakage",       "decoy": None,                   "difficulty": "medium"},
 "S4": {"ground_truth": "bad_preprocessing",  "decoy": "distribution_shift",   "difficulty": "medium"},
 "S5": {"ground_truth": "distribution_shift", "decoy": "overfitting",          "difficulty": "medium"},
 "S6": {"ground_truth": "data_leakage",       "decoy": "overfitting",          "difficulty": "hard"},
}  # evidence sets get added on Day 6