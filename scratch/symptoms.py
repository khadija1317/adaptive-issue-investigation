import numpy as np
from environment.scenarios import SCENARIOS
from environment.runner import run

print("sid  train    val   test   train_counts  val_counts")
for sid, sc in SCENARIOS.items():
    rec, test = run(sc)
    rec2, _ = run(sc)
    assert rec.metrics == rec2.metrics, f"{sid} not reproducible"
    print(f"{sid}   {rec.metrics['train']['accuracy']:.3f}  {rec.metrics['val']['accuracy']:.3f}  "
          f"{test['accuracy']:.3f}   {np.bincount(rec.y['train'])}  {np.bincount(rec.y['val'])}")