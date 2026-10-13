# usage: python -m scratch.run_scored S4 0 low    |    python -m scratch.run_scored selftest
import sys
from evaluation.eval_meta import EVAL_META
from evaluation.score_fix import score_fix

PROSE_FAILS = ("tool_use_failed", "output_parse_failed")   # the model wrote prose instead of a tool call

def _last_error(r):
    errs = [t for t in r.get("trace", []) if t.get("event") == "llm_error"]
    return str(errs[-1].get("result", "")) if errs else ""

def is_invalid(r):   # INVALID = ended by a provider error that is NOT a model prose failure; rerun allowed
    return r.get("stop_reason") == "llm_error" and not any(c in _last_error(r) for c in PROSE_FAILS)

def scored(r):   # diagnosis vs ground truth, recommended fix vs hidden test
    if is_invalid(r):
        print(f"INVALID RUN {r['scenario']} seed {r['seed']}: provider error, not a model result (tokens {r['tokens_total']}). Not scored.")
        return None, None
    sid, seed, d = r["scenario"], r["seed"], r["diagnosis"]
    fix = score_fix(sid, d.get("recommended_config", {}), seed)
    ok = d["label"] == EVAL_META[sid]["ground_truth"]
    print(f"SCORE {sid} seed {seed}: diagnosis {'CORRECT' if ok else 'WRONG'} (said {d['label']} {d['confidence']}) | "
          f"fix {fix['success']} (recommended {d.get('recommended_config')}) | "
          f"experiments {r['experiments']} | tokens {r['tokens_total']}")
    if r.get("stop_reason") == "llm_error":
        print("  NOTE: run ended on repeated model prose failures (no submission): scored as a failed run, not INVALID.")
    return ok, fix

if __name__ == "__main__":
    if sys.argv[1] == "selftest":   # zero tokens: hand-made results through the scoring path
        fake = lambda cfg, label: {"scenario": "S1", "seed": 0, "experiments": 0, "tokens_total": 0,
                                   "diagnosis": {"label": label, "confidence": 0.9, "recommended_config": cfg}}
        assert scored(fake({"class_weight": "balanced"}, "class_imbalance"))[1]["success"] is True
        assert scored(fake({}, "overfitting"))[0] is False
        def ended(err):   # a run that stopped on a provider error; diagnosis is the loop's default
            r = fake({}, "other_or_inconclusive"); r["stop_reason"] = "llm_error"
            r["trace"] = [{"event": "llm_error", "result": err}]; return r
        assert scored(ended("Error code: 400 - 'code': 'tool_use_failed'")) [0] is False     # model failure: SCORED
        assert scored(ended("Error code: 400 - 'code': 'output_parse_failed'"))[0] is False  # model failure: SCORED
        assert scored(ended("Error code: 401 - invalid_api_key")) == (None, None)            # provider error: INVALID
        assert scored(ended("Connection error.")) == (None, None)                            # provider error: INVALID
        r = ended("x"); r["trace"] = []; assert scored(r) == (None, None)                    # no error text at all: INVALID
        print("SELFTEST OK")
    else:
        assert len(sys.argv) == 4 and sys.argv[1] in EVAL_META and sys.argv[3] in ("low", "medium", "high"), \
            "usage: python -m scratch.run_scored S4 0 low   (or: selftest)"
        from agents.loop import investigate, show
        r = investigate(sys.argv[1], int(sys.argv[2]), effort=sys.argv[3]); show(r); scored(r)