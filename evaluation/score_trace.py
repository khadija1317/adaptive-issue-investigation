# usage: python -m evaluation.score_trace logs/traces/S3_0_low.json [more.json ...]   |   python -m evaluation.score_trace selftest
# Automatic columns for one trace (zero tokens). Judgment columns are NOT computed here: they are hand-coded and labelled manual.
import json, os, sys
from evaluation.eval_meta import EVAL_META

CONF_WRONG = 0.75                      # frozen Oct 13 before any frozen-prompt trace was scored
PROSE_FAILS = ("tool_use_failed", "output_parse_failed")
EXCLUDED = {"S4_0_low_oct12_rev5_run1_llm_error.json"}   # INVALID by name
MAX_EXPS = 3

def _j(s):
    try: return json.loads(s)
    except Exception: return None

def _norm(c):   # order-free form of a config, so recommended == experiment is an exact comparison
    return json.dumps({k: (sorted(v) if isinstance(v, list) else v) for k, v in (c or {}).items()}, sort_keys=True)

def is_invalid(r):   # same rule as scratch/run_scored.py
    errs = [t for t in r.get("trace", []) if t.get("event") == "llm_error"]
    last = str(errs[-1].get("result", "")) if errs else ""
    return r.get("stop_reason") == "llm_error" and not any(c in last for c in PROSE_FAILS)

def _parse(r):
    metric = EVAL_META[r["scenario"]]["fix"]["metric"]
    base, calls, exps = None, [], []
    for t in r["trace"]:
        if t.get("event") != "ok" or t["tool"] == "submit_diagnosis": continue
        body = (_j(t.get("result", "")) or {}).get("result", {})
        if t["tool"] == "get_metrics" and t["args"].get("split") == "val" and base is None:
            base = body.get(metric)
        if t["tool"] == "run_experiment":
            ident, after = body.get("identical_to_baseline", {}), (body.get("val") or {}).get(metric)
            exps.append({"changes": t["args"].get("changes", {}), "after": after,
                         "up": base is not None and after is not None and after > base,
                         "down": base is not None and after is not None and after < base,
                         "moved": not ident.get("val_metrics", True),
                         "size_changed": not ident.get("model_size", True),
                         "noop": bool(ident.get("val_metrics")) and bool(ident.get("model_size"))})
        else:
            calls.append({"tool": t["tool"], "args": t.get("args") or {}})
    return base, calls, exps

def _val_ok(v, e):
    return {None: True, "up": e["up"], "down": e["down"], "moved": e["moved"],
            "no_gain": (not e["up"]) and e["size_changed"]}[v]

def _atom(a, calls, exps):
    if a["tool"] != "run_experiment": return any(c["tool"] == a["tool"] for c in calls)
    return any(a["when"](e["changes"]) and _val_ok(a["val"], e) for e in exps)

def _set_met(alts, calls, exps):   # alternatives: any one fully met is enough; no sets -> None
    return None if not alts else any(all(_atom(a, calls, exps) for a in alt) for alt in alts)

def _in_sets(kind, name, changes, sets):   # does this call match any atom of the sets (ignoring the val outcome)?
    for alt in sets:
        for a in alt:
            if kind == "tool" and a["tool"] == name: return True
            if kind == "exp" and a["tool"] == "run_experiment" and a["when"](changes): return True
    return False

def _ruled_out(items, calls, exps):   # frozen cause-to-tool map; 'manual' = needs the claim read by a human
    names = {c["tool"] for c in calls}
    cd = {c["args"].get("split") for c in calls if c["tool"] == "get_class_distribution"}
    xs = lambda f: [e for e in exps if f(e["changes"])]
    knob = lambda c: "scaling" in c or "split_strategy" in c
    out = []
    for it in items:
        cause, tools, ex = it.get("cause"), [], []
        if cause == "class_imbalance":
            tools = [n for n in ("get_class_distribution", "analyze_confusion_matrix") if n in names]; ex = xs(lambda c: "class_weight" in c)
        elif cause == "overfitting":
            tools = [n for n in ("compare_train_validation",) if n in names]
            ex = xs(lambda c: c.get("min_samples_leaf", 1) > 1 or c.get("max_depth") is not None or c.get("train_fraction") is not None)
        elif cause == "data_leakage":
            tools = [n for n in ("check_leakage",) if n in names]; ex = xs(lambda c: "drop_features" in c)
        elif cause in ("bad_preprocessing", "distribution_shift"):
            tools = [n for n in ("inspect_feature_distribution",) if n in names]; ex = xs(knob)
            if cause == "distribution_shift" and {"train", "val"} <= cd: tools.append("get_class_distribution(both)")
        if tools or ex: status = "supported"
        elif cause == "overfitting" and any(c["tool"] == "get_metrics" and c["args"].get("split") == "train" for c in calls): status = "manual"
        else: status = "unsupported"
        out.append({"cause": cause, "status": status, "noop_only": status == "supported" and not tools and all(e["noop"] for e in ex)})
    return out

def score_trace(r, filename="", fix_fn=None):
    sid, seed, d = r["scenario"], r["seed"], r.get("diagnosis", {})
    meta = EVAL_META[sid]
    base, calls, exps = _parse(r)
    submitted = any(t.get("event") == "submit" for t in r["trace"])
    row = {"sid": sid, "seed": seed, "file": filename, "stop_reason": r.get("stop_reason"),
           "invalid": is_invalid(r) or os.path.basename(filename) in EXCLUDED}
    if row["invalid"]:
        row["outcome"] = "invalid (not scored)"; return row
    label, conf = d.get("label"), d.get("confidence", 0.0)
    correct = label == meta["ground_truth"]
    row.update(label=label, confidence=conf, correct=correct, submitted=submitted,
               outcome="correct" if correct else ("failed_no_submission" if not submitted else
                       ("wrong_confident" if conf >= CONF_WRONG and label != "other_or_inconclusive" else "wrong_cautious")),
               decoy_fallen=label == meta["decoy"] if meta["decoy"] else None,
               steps=r.get("llm_calls"), experiments=len(exps), exps_unused=MAX_EXPS - len(exps), tokens=r.get("tokens_total"),
               llm_errors=sum(t.get("event") == "llm_error" for t in r["trace"]),
               true_set_met=_set_met(meta["evidence_true"], calls, exps), decoy_set_met=_set_met(meta["evidence_decoy"], calls, exps),
               waste=sum(1 for c in calls if c["tool"] != "get_metrics" and not _in_sets("tool", c["tool"], {}, meta["evidence_true"] + meta["evidence_decoy"]))
                     + sum(1 for e in exps if not _in_sets("exp", "", e["changes"], meta["evidence_true"] + meta["evidence_decoy"])),
               noop_experiments=sum(e["noop"] for e in exps), correct_zero_exps=correct and not exps)
    rec = d.get("recommended_config") or {}
    row["rec_tested"] = None if not rec else any(_norm(rec) == _norm(e["changes"]) for e in exps)
    text = json.dumps(d.get("supporting_evidence", [])).lower()
    row["empty_rec_no_note"] = (not rec) and submitted and row["exps_unused"] > 0 and not any(w in text for w in ("untest", "not tested", "not been tested"))
    ro = _ruled_out(d.get("alternatives_ruled_out", []), calls, exps) if submitted else []
    row["ruled_out"] = ro
    row["ro_supported"], row["ro_manual"], row["ro_total"] = (sum(x["status"] == "supported" for x in ro), sum(x["status"] == "manual" for x in ro), len(ro))
    full = (text + json.dumps(d.get("alternatives_ruled_out", []))).lower()
    row["cites_discriminating"] = None if not submitted else (("std_ratio" in full) if sid == "S4" else (("macro_recall" in full or "macro-recall" in full) if sid == "S5" else None))
    row["fix"] = fix_fn(sid, rec, seed)["success"] if fix_fn else "not computed"
    return row

def show_row(row):
    if row["invalid"]: print(f"{row['sid']} seed {row['seed']} [{row['file']}]: INVALID, not scored"); return
    print(f"{row['sid']} seed {row['seed']} [{row['file']}] {row['outcome'].upper()} ({row['label']} {row['confidence']}) fix={row['fix']} rec_tested={row['rec_tested']}")
    print(f"   steps {row['steps']} exps {row['experiments']} (unused {row['exps_unused']}) tokens {row['tokens']} llm_errors {row['llm_errors']} waste {row['waste']} noop_exps {row['noop_experiments']}")
    print(f"   true_set_met={row['true_set_met']} decoy_set_met={row['decoy_set_met']} decoy_fallen={row['decoy_fallen']} empty_rec_no_note={row['empty_rec_no_note']} cites_discriminating={row['cites_discriminating']}")
    print(f"   ruled-out: {row['ro_supported']} supported + {row['ro_manual']} manual of {row['ro_total']}: " +
          ", ".join(f"{x['cause']}={x['status']}{'(noop only)' if x['noop_only'] else ''}" for x in row["ruled_out"]))

def _selftest():
    def ev(tool, args, result=None, event="ok"):
        return {"event": event, "tool": tool, "args": args, "result": json.dumps({"tool": tool, "args": args, "result": result or {}})}
    gm = lambda s, acc: ev("get_metrics", {"split": s}, {"accuracy": acc, "macro_recall": acc, "model": {}})
    exp = lambda ch, va, same_v=False, same_m=False: ev("run_experiment", {"changes": ch}, {"changed": ch, "val": {"accuracy": va, "macro_recall": va},
              "identical_to_baseline": {"val_metrics": same_v, "model_size": same_m}})
    def run(sid, trace, label, conf, rec, ro=(), sup=("x",), stop="submitted", file="t.json"):
        r = {"scenario": sid, "seed": 0, "stop_reason": stop, "llm_calls": len(trace), "tokens_total": 1, "trace": trace,
             "diagnosis": {"label": label, "confidence": conf, "recommended_config": rec, "supporting_evidence": list(sup), "alternatives_ruled_out": list(ro)}}
        return score_trace(r, file)
    S = {"event": "submit", "tool": "submit_diagnosis", "args": {}, "result": "{}"}
    # 1 S3-like: leakage found, fix tested
    a = run("S3", [gm("train", 1.0), gm("val", 1.0), ev("check_leakage", {}), exp({"drop_features": ["aux_score"]}, 0.912), S], "data_leakage", 0.95, {"drop_features": ["aux_score"]})
    assert a["outcome"] == "correct" and a["true_set_met"] and a["rec_tested"] and a["waste"] == 0
    # 2 S2-like: two no-ops used to rule out causes; threshold 0.75
    b = run("S2", [gm("train", 1.0), gm("val", 0.806), exp({"max_depth": 3}, 0.806, True, True), exp({"scaling": "standard"}, 0.806, True, True), S],
            "other_or_inconclusive", 0.7, {}, ro=[{"cause": "overfitting", "evidence": "e"}, {"cause": "bad_preprocessing", "evidence": "e"}])
    assert b["outcome"] == "wrong_cautious" and b["noop_experiments"] == 2 and b["empty_rec_no_note"] and b["true_set_met"] is False
    assert all(x["status"] == "supported" and x["noop_only"] for x in b["ruled_out"])
    assert run("S2", [gm("val", 0.8), S], "distribution_shift", 0.75, {})["outcome"] == "wrong_confident"
    assert run("S2", [gm("val", 0.8), S], "distribution_shift", 0.74, {})["outcome"] == "wrong_cautious"
    # 3 failures: prose-failure ending is scored, provider error is INVALID, the named file is excluded
    err = lambda m: {"event": "llm_error", "tool": "-", "args": None, "result": m}
    assert run("S6", [gm("val", 0.8), err("code: output_parse_failed")], "other_or_inconclusive", 0.0, {}, stop="llm_error")["outcome"] == "failed_no_submission"
    assert run("S6", [err("Connection error.")], "other_or_inconclusive", 0.0, {}, stop="llm_error")["invalid"]
    assert run("S4", [gm("val", 0.4), S], "bad_preprocessing", 0.9, {}, file="logs/archive/S4_0_low_oct12_rev5_run1_llm_error.json")["invalid"]
    # 4 evidence atoms: informative flat (size changed) counts as no_gain, a no-op does not; S4 per_split up
    c = run("S6", [gm("val", 0.89), exp({"min_samples_leaf": 5}, 0.80), S], "overfitting", 0.8, {})
    assert c["decoy_set_met"] is True and c["decoy_fallen"] is True
    d = run("S6", [gm("val", 0.89), exp({"min_samples_leaf": 5}, 0.89, True, True), S], "overfitting", 0.8, {})
    assert d["decoy_set_met"] is False
    e = run("S4", [gm("val", 0.38), exp({"scaling": "per_split"}, 0.94), S], "bad_preprocessing", 0.9, {"scaling": "per_split"})
    assert e["true_set_met"] and e["rec_tested"] and e["cites_discriminating"] is False
    # 5 recommended config must equal ONE experiment exactly (union of tested changes is untested)
    f = run("S1", [gm("val", 0.5), exp({"class_weight": "balanced"}, 0.8), exp({"C": 1.0}, 0.6), S], "class_imbalance", 0.9, {"class_weight": "balanced", "C": 1.0})
    assert f["rec_tested"] is False
    print("SCORE_TRACE SELFTEST OK")

if __name__ == "__main__":
    if sys.argv[1] == "selftest": _selftest(); sys.exit()
    try:
        from evaluation.score_fix import score_fix as fx
    except Exception as e:
        print(f"[fix success not computed: {e}]"); fx = None
    for p in sys.argv[1:]:
        show_row(score_trace(json.load(open(p)), p, fx))