from evaluation.score_fix import score_fix
from evaluation.eval_meta import EVAL_META
WRONG = {"S1": {"train_fraction": 1.0}, "S2": {"min_samples_leaf": 5}, "S3": {"min_samples_leaf": 5},
         "S4": {"scaling": "standard"}, "S5": {"scaling": "per_split"}, "S6": {"min_samples_leaf": 5},
         "S7": {"class_weight": "balanced"}}
print("sid seed | reference | no-op | wrong fix   (True=success, None=unscorable)")
for sid in EVAL_META:
    for seed in (0, 1, 2):
        r, n, w = (score_fix(sid, c, seed) for c in (EVAL_META[sid]["fix"]["reference"], {}, WRONG[sid]))
        print(sid, seed, "|", r["success"], f'{r["base_test"]:.3f}->{r["fixed_test"]:.3f}', "|", n["success"], "|", w["success"])