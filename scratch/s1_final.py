from dataclasses import replace
from sklearn.metrics import recall_score
from environment.scenarios import SCENARIOS
from environment.runner import run

sc = SCENARIOS["S1"]
for s in (0, 1, 2):
    for cw in (None, "balanced"):
        rec, test = run(sc, replace(sc.base_config, seed=s, class_weight=cw))
        r = recall_score(rec.y["val"], rec.pred["val"], pos_label=0)
        print(f"seed {s} weight={str(cw):<9} val_acc={rec.metrics['val']['accuracy']:.3f} "
              f"minority_recall={r:.2f} on {(rec.y['val'] == 0).sum()} rows  test_macro_recall={test['macro_recall']:.3f}")