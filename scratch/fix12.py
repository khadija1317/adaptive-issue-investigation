import numpy as np
from dataclasses import replace
from functools import partial
from sklearn.metrics import recall_score
from environment.scenarios import SCENARIOS
from environment import injectors as inj
from environment.runner import run

print("S1: want high accuracy, LOW minority recall, and balanced weights recovering it")
print("frac   C      weight     val_acc  val_min_recall  val_min_rows  test_macro_recall")
for frac in (0.05, 0.10, 0.15):
    sc = replace(SCENARIOS["S1"], data_injector=partial(inj.make_imbalanced, minority_class=0, minority_frac=frac))
    for C in (0.01, 0.003):
        for cw in (None, "balanced"):
            acc, rc, n, tr = [], [], [], []
            for s in (0, 1, 2):
                rec, test = run(sc, replace(sc.base_config, C=C, class_weight=cw, seed=s))
                acc.append(rec.metrics["val"]["accuracy"])
                rc.append(recall_score(rec.y["val"], rec.pred["val"], pos_label=0))
                n.append(int((rec.y["val"] == 0).sum()))
                tr.append(test["macro_recall"])
            print(f"{frac:.2f}  {C:<6} {str(cw):<9}  {np.mean(acc):.3f}    {np.mean(rc):.2f}            {np.mean(n):.0f}            {np.mean(tr):.3f}")

print("\nS2: does reducing capacity OR adding data close the val gap?")
sc = SCENARIOS["S2"]
for s in (0, 1, 2):
    for label, chg in [("baseline (40% data)", {}), ("max_depth=2", {"max_depth": 2}),
                       ("70% data", {"train_fraction": 0.7}), ("100% data", {"train_fraction": 1.0})]:
        rec, test = run(sc, replace(sc.base_config, seed=s, **chg))
        print(f"  seed {s} {label:<22} train={rec.metrics['train']['accuracy']:.3f} "
              f"val={rec.metrics['val']['accuracy']:.3f} test={test['accuracy']:.3f}")