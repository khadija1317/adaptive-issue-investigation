import numpy as np
from dataclasses import replace
from sklearn.metrics import recall_score
from environment.scenarios import SCENARIOS
from environment.runner import run

def show(label, sid, **chg):
    sc = SCENARIOS[sid]
    rec, test = run(sc, replace(sc.base_config, **chg))
    print(f"  {label:<30} train={rec.metrics['train']['accuracy']:.3f} "
          f"val={rec.metrics['val']['accuracy']:.3f} test={test['accuracy']:.3f}")
    return rec

print("S1: is the minority class actually missed?")
for C in (0.1, 0.01, 0.001):
    rec = show(f"logreg C={C}", "S1", C=C)
    r = recall_score(rec.y["val"], rec.pred["val"], pos_label=0)
    print(f"     minority recall (val) = {r:.2f}  on {(rec.y['val'] == 0).sum()} rows")

print("S2: does regularization close the gap?")
show("baseline", "S2")
show("max_depth=2", "S2", max_depth=2)
show("max_depth=3", "S2", max_depth=3)

print("S3: regularization must NOT fix, dropping must")
show("baseline", "S3")
show("max_depth=3, leaf=5", "S3", max_depth=3, min_samples_leaf=5)
show("drop aux_score", "S3", drop_features=("aux_score",))

print("S6 over 3 seeds")
for s in (0, 1, 2):
    show(f"seed {s} baseline", "S6", seed=s)
    show(f"seed {s} regularized", "S6", seed=s, max_depth=3, min_samples_leaf=5)
    show(f"seed {s} drop aux_score", "S6", seed=s, drop_features=("aux_score",))

print("S4 and S5 fixes")
show("S4 baseline", "S4")
show("S4 scaling=standard", "S4", scaling="standard")
show("S4 scaling=per_split", "S4", scaling="per_split")
show("S5 baseline", "S5")
show("S5 split=random", "S5", split_strategy="random")
show("S5 scaling=per_split", "S5", scaling="per_split")

print("Shift profile (val vs train)")
for sid in ("S4", "S5", "S6", "S1"):
    rec, _ = run(SCENARIOS[sid])
    Xt, Xv = rec.X["train"], rec.X["val"]
    sd = Xt.std(0) + 1e-9
    smd = np.abs(Xv.mean(0) - Xt.mean(0)) / sd
    ratio = Xv.std(0) / sd
    print(f"  {sid}: features with SMD>0.5: {(smd > 0.5).sum()}/{len(smd)}  "
          f"std ratio val/train: median={np.median(ratio):.2f} min={ratio.min():.2f} max={ratio.max():.2f}")