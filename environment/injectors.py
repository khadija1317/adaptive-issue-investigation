import numpy as np

# Data-stage injector (runs before splitting): S1.
def make_imbalanced(X, y, minority_class, minority_frac, rng):
    """Subsample one class so it is ~minority_frac of the data."""
    idx_min = np.where(y == minority_class)[0]
    idx_maj = np.where(y != minority_class)[0]
    n_min = int(round(minority_frac / (1 - minority_frac) * len(idx_maj)))
    keep_min = rng.choice(idx_min, size=min(n_min, len(idx_min)), replace=False)
    keep = np.sort(np.concatenate([keep_min, idx_maj]))
    return X[keep], y[keep]

# Split-stage injectors: take and return splits = {"train": (X, y), "val": ..., "test": ...}.
def add_leaky_feature(splits, signal_in, noise, rng):
    """Append aux_score. In splits listed in signal_in it is label + noise;
    elsewhere it is a permuted label + noise (carries no information).
    S6: signal_in=("train",)   S3: signal_in=("train", "val")"""
    out = {}
    for name, (X, y) in splits.items():
        base = y.astype(float) if name in signal_in else rng.permutation(y).astype(float)
        aux = base + rng.normal(0, noise, size=len(y))
        out[name] = (np.column_stack([X, aux]), y)
    return out

def corrupt_scaling(splits, factor, target="val", rng=None):
    """S4: multiply EVERY feature of one split by a constant factor."""
    out = dict(splits)
    X, y = out[target]
    out[target] = (X * factor, y)
    return out