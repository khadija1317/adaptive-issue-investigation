from dataclasses import replace
from environment.runner import run
from environment.scenarios import SCENARIOS
from evaluation.eval_meta import EVAL_META

MIN_EFFECT, SHARE, TEST_TOL = 0.05, 0.5, 0.03   # fixed before looking at numbers

def _measure(sid, changes, seed):   # baseline config + changes, same seed => same hidden test split
    sc, ch = SCENARIOS[sid], dict(changes or {})
    if "drop_features" in ch: ch["drop_features"] = tuple(ch["drop_features"])
    rec, test = run(sc, replace(sc.base_config, seed=seed, **ch))
    return rec.metrics["val"], test

def _agree(m, bv, bt, fv, ft):   # True/False, or None if the baseline val-vs-test gap is too small to measure
    g0, g1 = abs(bv[m] - bt[m]), abs(fv[m] - ft[m])
    if g0 < MIN_EFFECT: return None
    return (1 - g1 / g0) >= SHARE and ft[m] >= bt[m] - TEST_TOL

def score_fix(sid, changes, seed=0):
    rule = EVAL_META[sid]["fix"]; m = rule["metric"]
    bv, bt = _measure(sid, {}, seed)
    try: fv, ft = _measure(sid, changes, seed)
    except (ValueError, TypeError) as e: return {"success": False, "error": str(e)}
    out = {"sid": sid, "seed": seed, "base_test": bt[m], "fixed_test": ft[m]}
    if rule["kind"] == "gain":
        _, rt = _measure(sid, rule["reference"], seed)
        ref_gain = rt[m] - bt[m]
        out["success"] = None if ref_gain < MIN_EFFECT else (ft[m] - bt[m]) >= SHARE * ref_gain
    else:   # "agree": does val now tell the truth about test?
        rv, rt = _measure(sid, rule["reference"], seed)
        ref_ok = _agree(m, bv, bt, rv, rt)
        out["success"] = _agree(m, bv, bt, fv, ft) if ref_ok else None   # reference fails its own test => unscorable
    return out   # <- this line must sit at function level (4 spaces), after the if/else