from dataclasses import replace
from environment.scenarios import SCENARIOS
from environment.runner import run

sc, N = SCENARIOS["S6"], 114
print("seed  lever                  depth  val    delta (rows of 114)")
for s in (0, 1, 2):
    rec0, _ = run(sc, replace(sc.base_config, seed=s))
    bv = rec0.metrics["val"]["accuracy"]
    print(f"{s}     {'baseline':<22} {rec0.model.get_depth():<5}  {bv:.3f}  -")
    for label, chg in [("max_depth=2", {"max_depth": 2}),
                       ("max_depth=3", {"max_depth": 3}),
                       ("max_depth=4", {"max_depth": 4}),
                       ("min_samples_leaf=5", {"min_samples_leaf": 5}),
                       ("depth=3 + leaf=5", {"max_depth": 3, "min_samples_leaf": 5}),
                       ("drop aux_score", {"drop_features": ("aux_score",)})]:
        rec, _ = run(sc, replace(sc.base_config, seed=s, **chg))
        v = rec.metrics["val"]["accuracy"]
        print(f"{s}     {label:<22} {rec.model.get_depth():<5}  {v:.3f}  {round((v - bv) * N):+d}")