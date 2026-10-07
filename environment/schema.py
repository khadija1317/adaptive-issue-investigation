from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

# What the agent may change. These are the only knobs (the legitimate levers).
@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "tree"                 # "tree" | "logreg"
    max_depth: Optional[int] = None     # tree capacity
    min_samples_leaf: int = 1
    C: float = 1.0                      # logreg regularization
    train_fraction: float = 1.0
    class_weight: Optional[str] = None  # None | "balanced"
    drop_features: Tuple[str, ...] = ()
    scaling: str = "none"               # "none" | "standard" | "per_split"
    split_strategy: str = "random"      # "random" | "by_size"
    seed: int = 0

# What tools read. Train/val only: test is NOT here.
@dataclass
class ExperimentRecord:
    config: ExperimentConfig
    feature_names: list
    X: dict = field(default_factory=dict)      # {"train": array, "val": array}
    y: dict = field(default_factory=dict)
    pred: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    model: Any = None