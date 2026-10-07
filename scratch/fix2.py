import numpy as np
from dataclasses import replace
from environment.scenarios import SCENARIOS
from environment.runner import run

sc = SCENARIOS["S2"]
print("base_frac  lever         train   val    test   (mean of 5 seeds)")
for frac in (0.10, 0.15, 0.20):
    for label, chg in [("baseline", {}), ("max_depth=2", {"max_depth": 2}),
                       ("100% data", {"train_fraction": 1.0})]:
        tr, va, te = [], [], []
        for s in range(5):
            kw = {"train_fraction": frac, "seed": s, **chg}
            rec, test = run(sc, replace(sc.base_config, **kw))
            tr.append(rec.metrics["train"]["accuracy"])
            va.append(rec.metrics["val"]["accuracy"])
            te.append(test["accuracy"])
        print(f"{frac:.2f}       {label:<12}  {np.mean(tr):.3f}   {np.mean(va):.3f}  {np.mean(te):.3f}")