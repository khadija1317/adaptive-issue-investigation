from environment.scenarios import SCENARIOS
from environment.runner import run

rec, _ = run(SCENARIOS["S2"])
print("S2 unconstrained tree depth:", rec.model.get_depth(), "| leaves:", rec.model.get_n_leaves(),
      "| train rows:", len(rec.y["train"]))