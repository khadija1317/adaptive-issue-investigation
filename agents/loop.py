# agents/loop.py

import copy, dataclasses, json, sys, time
from pathlib import Path
from agents.llm import llm_call
from agents.prompt import SYSTEM_PROMPT, SUBMIT_TOOL, LABELS
from environment.scenarios import SCENARIOS
from tools.diagnostics import Session
from tools.registry import TOOL_SCHEMAS

MAX_STEPS, MIN_DIAG, MAX_EXPERIMENTS, PROMPT_LIMIT = 12, 3, 3, 6000
FORCE = {"type": "function", "function": {"name": "submit_diagnosis"}}
TRACE_DIR = Path("logs/traces")
MAX_STALLS = 3

NUDGE = ("Nothing has been submitted: text replies are not recorded. To continue, call a diagnostic tool. "
         "To finish, call the submit_diagnosis tool with your answer.")

TRANSIENT = ("connection error", "timed out", "timeout", "502", "503", "504", "overloaded")
def _llm_retry(llm, *a, tries=3, **k):   # retry network blips; model-side errors still propagate
    for i in range(tries):
        try: return llm(*a, **k)
        except Exception as e:
            if i == tries - 1 or not any(t in str(e).lower() for t in TRANSIENT): raise
            print(f"[retry {i+1}] {str(e)[:80]}"); time.sleep(5 * (i + 1))

# Inject a required `hypothesis` arg into every diagnostic schema (registry.py stays untouched).
def _with_hypothesis(schemas):
    out = []
    for s in schemas:
        s = copy.deepcopy(s)
        p = s["function"]["parameters"]
        p["properties"] = {"hypothesis": {"type": "string", "description":
            "What you currently suspect and what this call tests. Cite earlier results."}, **p["properties"]}
        p["required"] = ["hypothesis"] + list(p.get("required", []))
        out.append(s)
    return out

TOOLS = _with_hypothesis(TOOL_SCHEMAS) + [SUBMIT_TOOL]

# Seed goes through base_config.seed (Scenario and config are dataclasses, so replace() works).
def make_scenario(sid, seed=0):
    sc = SCENARIOS[sid]
    return dataclasses.replace(sc, base_config=dataclasses.replace(sc.base_config, seed=seed))
# Code-side check of the recommended fix (the same rules run_experiment enforces, but never touches test).
def _check_config(cfg, sess):
    if not isinstance(cfg, dict): return "recommended_config must be an object ({} means no change)"
    bad = set(cfg) - sess.KNOBS
    if bad: return f"unknown knobs {sorted(bad)}; allowed: {sorted(sess.KNOBS)}"
    tree = sess.rec.config.model == "tree"
    isint = lambda v: isinstance(v, int) and not isinstance(v, bool)
    isnum = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    ok = {"max_depth": lambda v: tree and isint(v) and v >= 1,
          "min_samples_leaf": lambda v: tree and isint(v) and v >= 1,
          "C": lambda v: (not tree) and isnum(v) and v > 0,
          "train_fraction": lambda v: isnum(v) and 0 < v <= 1,
          "class_weight": lambda v: v == "balanced",
          "scaling": lambda v: v in ("none", "standard", "per_split"),
          "split_strategy": lambda v: v in ("random", "by_size"),
          "drop_features": lambda v: isinstance(v, list) and all(f in sess.rec.feature_names for f in v)}
    for k, v in cfg.items():
        if not ok.get(k, lambda v: False)(v): return f"invalid value for {k}: {v!r} (check the model type and the allowed range)"
    return None
# Code-side validation of the final answer (the prompt can't enforce this).
def _validate(a, sess):
    if not isinstance(a, dict): return "submit_diagnosis arguments must be a JSON object"
    if a.get("label") not in LABELS: return f"label must be one of {LABELS}"
    c = a.get("confidence")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0 <= c <= 1:
        return "confidence must be a number between 0 and 1"
    if not isinstance(a.get("supporting_evidence"), list) or not a["supporting_evidence"]:
        return "supporting_evidence must be a non-empty list"
    alts = a.get("alternatives_ruled_out")
    if not isinstance(alts, list): return "alternatives_ruled_out must be a list"
    for x in alts:
        if not isinstance(x, dict) or x.get("cause") not in LABELS or not str(x.get("evidence", "")).strip():
            return "each alternatives_ruled_out item needs a valid cause and a non-empty evidence string"
    if any(x["cause"] == a["label"] for x in alts):
        return "alternatives_ruled_out cannot include your own label"
    return _check_config(a.get("recommended_config"), sess)

def investigate(sid, seed=0, max_steps=MAX_STEPS, effort="low", llm=llm_call, save=True):
    sess = Session(make_scenario(sid, seed))
    base = sess.call("get_metrics", {"split": "val"})["result"]   # internal baseline, not shown to the agent
    history = [{"role": "user", "content": "Investigate this experiment.\n" + json.dumps(sess.task_info())}]
    trace, seen, n_exp, n_diag, tokens = [], set(), 0, 0, 0
    stalls = 0
    baseline_seen = False
    diagnosis, stop = None, "no_submit"

    for step in range(1, max_steps + 1):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        est = int((len(json.dumps(msgs)) + len(json.dumps(TOOLS))) / 3.5)   # rough token estimate
        force = step == max_steps or est > PROMPT_LIMIT                      # code forces the end
        # Structural force: on the last step the model is only OFFERED submit_diagnosis.
        call_tools, choice = ([SUBMIT_TOOL], "required") if force else (TOOLS, "required")
        try:
            resp = _llm_retry(llm, msgs, tools=call_tools, tool_choice=choice,
                              tag=f"{sid}_s{seed}_step{step}", reasoning_effort=effort)
        except Exception as e:   # provider error: keep the trace instead of losing the run
            txt = str(e)
            print(f"[llm_error] step {step}: {txt[:200]}")
            trace.append({"step": step, "event": "llm_error", "tool": "-", "args": None,
                          "result": txt[:500], "result_chars": 0})
            if any(c in txt for c in ("tool_use_failed", "output_parse_failed")) and stalls < MAX_STALLS - 1:   # model answered in prose: recover
                stalls += 1
                history.append({"role": "user", "content": NUDGE})
                continue
            stop = "llm_error"
            break
        u = resp.usage
        tokens += u.total_tokens
        msg = resp.choices[0].message
        calls = msg.tool_calls or []

        # Assistant message must be appended BEFORE its tool results (API is stateless).
        am = {"role": "assistant", "content": msg.content or ""}
        if calls:
            am["tool_calls"] = [{"id": tc.id, "type": "function",
                                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                for tc in calls]
        history.append(am)

        if not calls:   # text-only turn: explicit nudge; repeated stalls end the run
            trace.append({"step": step, "event": "no_tool_call", "text": (msg.content or "")[:300],
                          "prompt_tokens": u.prompt_tokens, "tokens": u.total_tokens})
            stalls += 1
            if stalls >= MAX_STALLS:
                stop = "stalled"
                break
            history.append({"role": "user", "content": NUDGE})
            continue
        stalls = 0

        done = False
        for i, tc in enumerate(calls):
            name, event, hyp, args = tc.function.name, "ok", "", None
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                event, content = "malformed_args", json.dumps({"error": "arguments were not valid JSON"})

            if event == "ok" and name == "submit_diagnosis":
                err = _validate(args, sess)
                if err:
                    event, content = "invalid_submit", json.dumps({"error": err})
                elif n_diag < MIN_DIAG and not force:
                    event, content = "rejected_early", json.dumps(
                        {"error": f"Too early: make at least {MIN_DIAG} diagnostic calls first (made {n_diag})."})
                else:
                    diagnosis, stop, done = args, ("forced_submit" if force else "submitted"), True
                    event, content = "submit", "{}"
            elif event == "ok":
                if isinstance(args, dict):
                    hyp = str(args.pop("hypothesis", ""))
                key = (name, json.dumps(args, sort_keys=True, default=str))
                if key in seen:
                    event, content = "duplicate", json.dumps({"note": "Already called with identical arguments; result unchanged. Use a different call or submit_diagnosis."})
                elif name == "run_experiment" and n_exp >= MAX_EXPERIMENTS:
                    event, content = "exp_cap", json.dumps({"error": f"experiment limit ({MAX_EXPERIMENTS}) reached"})
                elif name == "run_experiment" and not baseline_seen :
                    event, content = "rejected_no_baseline", json.dumps(
                        {"error": "Read the baseline validation metrics (get_metrics, split=val) before running experiments."})
                else:
                    seen.add(key)
                    out = sess.call(name, args)
                    content = out if isinstance(out, str) else json.dumps(out, default=str)
                    if '"error"' in content:
                        event = "tool_error"   # logged distinctly; a failed call doesn't burn an experiment
                    else:
                        if name == "run_experiment" and isinstance(out, dict) and isinstance(out.get("result"), dict):
                            r = out["result"]
                            r["identical_to_baseline"] = {
                                "val_metrics": all(r["val"].get(k) == base.get(k) for k in ("accuracy", "macro_f1", "macro_recall")),
                                "model_size": r.get("model") == base.get("model")}
                            content = json.dumps(out, default=str)
                        n_diag += 1
                        if name == "get_metrics" and isinstance(args, dict) and args.get("split") == "val":
                            baseline_seen = True
                        if name == "run_experiment": n_exp += 1
                    

            trace.append({"step": step, "hypothesis": hyp, "tool": name, "args": args, "event": event,
                          "result": content, "result_chars": len(content),
                          "prompt_tokens": u.prompt_tokens if i == 0 else None,
                          "tokens": u.total_tokens if i == 0 else None})
            history.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            if done: break
        if done: break

    if diagnosis is None:
        diagnosis = {"label": "other_or_inconclusive", "confidence": 0.0,
                     "supporting_evidence": [], "recommended_config": {},"alternatives_ruled_out": []}
    result = {"scenario": sid, "seed": seed, "effort": effort, "stop_reason": stop,
              "llm_calls": step, "experiments": n_exp, "tokens_total": tokens,
              "diagnosis": diagnosis, "trace": trace}
    if save:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        (TRACE_DIR / f"{sid}_{seed}_{effort}.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result

def show(r):   # compact human-readable trace
    print(f"== {r['scenario']} seed {r['seed']} effort {r['effort']} | {r['stop_reason']} | "
          f"{r['llm_calls']} calls, {r['experiments']} exps, {r['tokens_total']} tokens")
    for t in r["trace"]:
        print(f"[{t['step']}] {t.get('tool','-')} {json.dumps(t.get('args'))[:90]} <{t['event']}>")
        if t.get("hypothesis"): print("     H:", t["hypothesis"][:200])
    d = r["diagnosis"]
    print("DIAGNOSIS:", d["label"], d["confidence"])
    for e in d["supporting_evidence"]: print("  +", e[:160])
    for e in d["alternatives_ruled_out"]: print(f"  - {e['cause']}: {e['evidence'][:140]}")

if __name__ == "__main__":   # python -m agents.loop S1 0 low
    sid = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    show(investigate(sid, seed, effort=sys.argv[3] if len(sys.argv) > 3 else "low"))