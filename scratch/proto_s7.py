"""S7 prototype (Day 5, zero LLM tokens). Run: python -m scratch.proto_s7
HONEST NEGATIVES (decoy = class_imbalance, truth = data_leakage, tree, train-only leak):
 A1 wine, class 0 -> 25%, noise 0.3: leak strong (+16/+21/+11 val rows), balanced a no-op, BUT shares 25/45/29 are not
    visible imbalance in 3 classes, val only 32 rows. Decoy not tempting.
 A2 wine, class 0 -> 12%: val 27 rows (~3 minority); seed 2 baseline val 0.963, drop aux gains +1 row: no symptom.
 B  breast_cancer, class 0 -> 15%, noise 0.3: seed 1 drop aux test gain +0.036 (< MIN_EFFECT) while balanced gains
    +0.048 on test, so imbalance is partly causal there. Decoy fails.
 Sweep frac 0.20/0.25/0.30 x noise 0.3: all fail (leak effect inconsistent over seeds, e.g. test gain -0.021 at 0.25/seed 1).
CRITERIA (POST-HOC, written after seeing A1/A2/B; environment data only; per seed, all 3 seeds): (1) drop aux test gain
 >= 0.05 and val gain >= 4 rows; (2) balanced test gain <= 0.02 and val gain <= 2 rows; (3) min_samples_leaf=5, max_depth=2,
 max_depth=3 each recover <= half of drop's val gain.
NEGATIVE 20%: passes criteria 1-3 but balanced leaves the tree at 2 leaves with val flat, i.e. an uninformative flat result
 (no_gain needs the model size to change), so the decoy atom could never be satisfied. At 25% balanced grows the tree 2->3.
CHOSEN: breast_cancer, class 0 -> 25%, leak noise 0.1 (train only). Passes all criteria; no_gain is satisfiable on seeds 0,1,2.
 FRAGILE: the 2->3 leaf change under balanced is a split-tie artifact (absent at 20%); rerun this script after any sklearn upgrade.
 CAVEAT: baseline is a 2-leaf stump on aux_score, so criterion 3 passes only trivially (tree unchanged), as in S3."""
import numpy as np
from dataclasses import replace
from environment.scenarios import SCENARIOS
from environment.runner import run

sc = SCENARIOS["S7"]
CH = {"drop": dict(drop_features=("aux_score",)), "balanced": dict(class_weight="balanced"),
      "leaf5": dict(min_samples_leaf=5), "depth2": dict(max_depth=2), "depth3": dict(max_depth=3)}
for seed in (0, 1, 2):
    base = replace(sc.base_config, seed=seed)
    rec, t0 = run(sc, base)
    nva, v0 = len(rec.y["val"]), rec.metrics["val"]["accuracy"]
    cnt = np.bincount(rec.y["train"].astype(int))
    print(f"seed {seed}: train counts {cnt.tolist()} ({cnt[0]/cnt.sum():.0%} minority) | base leaves {rec.model.get_n_leaves()} "
          f"train {rec.metrics['train']['accuracy']:.3f} val {v0:.3f} test {t0['accuracy']:.3f}")
    for k, ch in CH.items():
        r, t = run(sc, replace(base, **ch))
        print(f"    {k:8s} leaves {r.model.get_n_leaves():2d} val {(r.metrics['val']['accuracy']-v0)*nva:+.0f} rows  test {t['accuracy']-t0['accuracy']:+.3f}")