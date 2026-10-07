from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional
from .schema import ExperimentConfig
from . import injectors as inj

@dataclass(frozen=True)
class Scenario:
    sid: str
    dataset: str
    base_config: ExperimentConfig
    complaint: str
    data_injector: Optional[Callable] = None    # (X, y, rng=) before splitting
    split_injector: Optional[Callable] = None   # (splits, rng=) after splitting

SCENARIOS = {
 "S1": Scenario("S1", "breast_cancer", ExperimentConfig(model="logreg", C=0.003, scaling="standard"),
      "Accuracy is high but the model misses positive cases.",
      data_injector=partial(inj.make_imbalanced, minority_class=0, minority_frac=0.15)),
 "S2": Scenario("S2", "wine", ExperimentConfig(model="tree", train_fraction=0.15),
      "Perfect train score, weaker validation score."),
 "S3": Scenario("S3", "breast_cancer", ExperimentConfig(model="tree"),
      "Very high validation score, poor on held-out data.",
      split_injector=partial(inj.add_leaky_feature, signal_in=("train", "val"), noise=0.1)),
 "S4": Scenario("S4", "breast_cancer", ExperimentConfig(model="tree"),
      "Train score strong, validation score poor.",
      split_injector=partial(inj.corrupt_scaling, factor=3.0, target="val")),
 "S5": Scenario("S5", "breast_cancer", ExperimentConfig(model="tree", split_strategy="by_size"),
      "Train score strong, validation score weaker.",
      ),
 "S6": Scenario("S6", "breast_cancer", ExperimentConfig(model="tree"),
      "Train score perfect, validation score poor.",
      split_injector=partial(inj.add_leaky_feature, signal_in=("train",), noise=0.3)),
}

