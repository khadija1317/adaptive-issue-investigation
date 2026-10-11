# usage: python -m scratch.run_scored S4 0 low    |    python -m scratch.run_scored selftest
import sys
from evaluation.eval_meta import EVAL_META
from evaluation.score_fix import score_fix


def scored(r):   # diagnosis vs ground truth, recommended fix vs hidden test
    if r.get("stop_reason") == "llm_error":
        print(f"INVALID RUN {r['scenario']} seed {r['seed']}: provider error, not a model result (tokens {r['tokens_total']}). Not scored.")
        return None, None
     # diagnosis vs ground truth, recommended fix vs hidden test
    sid, seed, d = r["scenario"], r["seed"], r["diagnosis"]
    fix = score_fix(sid, d.get("recommended_config", {}), seed)
    ok = d["label"] == EVAL_META[sid]["ground_truth"]
    print(f"SCORE {sid} seed {seed}: diagnosis {'CORRECT' if ok else 'WRONG'} (said {d['label']} {d['confidence']}) | "
          f"fix {fix['success']} (recommended {d.get('recommended_config')}) | "
          f"experiments {r['experiments']} | tokens {r['tokens_total']}")
    return ok, fix

if __name__ == "__main__":
    if sys.argv[1] == "selftest":   # zero tokens: hand-made results through the scoring path
        fake = lambda cfg, label: {"scenario": "S1", "seed": 0, "experiments": 0, "tokens_total": 0,
                                   "diagnosis": {"label": label, "confidence": 0.9, "recommended_config": cfg}}
        assert scored(fake({"class_weight": "balanced"}, "class_imbalance"))[1]["success"] is True
        assert scored(fake({}, "overfitting"))[0] is False
        print("SELFTEST OK")
    else:
        assert len(sys.argv) == 4 and sys.argv[1] in EVAL_META and sys.argv[3] in ("low", "medium", "high"), \
            "usage: python -m scratch.run_scored S4 0 low   (or: selftest)"
        from agents.loop import investigate, show
        r = investigate(sys.argv[1], int(sys.argv[2]), effort=sys.argv[3]); show(r); scored(r)