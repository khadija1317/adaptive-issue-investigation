from dataclasses import replace
from environment.scenarios import SCENARIOS
from environment.runner import run

print("sid seed lever               depth  val    delta (rows)")
for sid in ("S2", "S3"):
    sc = SCENARIOS[sid]
    for s in (0, 1, 2):
        r0, _ = run(sc, replace(sc.base_config, seed=s))
        n, bv = len(r0.y["val"]), r0.metrics["val"]["accuracy"]
        print(f"{sid}  {s}    {'baseline':<19} {r0.model.get_depth():<5}  {bv:.3f}  -")
        chgs = [("min_samples_leaf=2", {"min_samples_leaf": 2}), ("min_samples_leaf=5", {"min_samples_leaf": 5})]
        if sid == "S2":
            chgs.append(("train_fraction=1.0", {"train_fraction": 1.0}))
        for label, chg in chgs:
            r, _ = run(sc, replace(sc.base_config, seed=s, **chg))
            v = r.metrics["val"]["accuracy"]
            print(f"{sid}  {s}    {label:<19} {r.model.get_depth():<5}  {v:.3f}  {round((v - bv) * n):+d}")