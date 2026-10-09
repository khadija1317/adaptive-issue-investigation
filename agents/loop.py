# agents/loop.py
import copy, dataclasses, json, sys
from pathlib import Path
from agents.llm import llm_call
from agents.prompt import SYSTEM_PROMPT, SUBMIT_TOOL, LABELS
from environment.scenarios import SCENARIOS
from tools.diagnostics import Session
from tools.registry import TOOL_SCHEMAS

MAX_STEPS, MIN_DIAG, MAX_EXPERIMENTS, PROMPT_LIMIT = 12, 3, 3, 6000
FORCE = {"type": "function", "function": {"name": "submit_diagnosis"}}
TRACE_DIR = Path("logs/traces")

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

# Code-side validation of the final answer (the prompt can't enforce this).
def _validate(a):
    if not isinstance(a, dict): return "submit_diagnosis arguments must be a JSON object"
    if a.get("label") not in LABELS: return f"label must be one of {LABELS}"
    c = a.get("confidence")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0 <= c <= 1:
        return "confidence must be a number between 0 and 1"
    if not isinstance(a.get("supporting_evidence"), list) or not a["supporting_evidence"]:
        return "supporting_evidence must be a non-empty list"
    if not isinstance(a.get("alternatives_ruled_out"), list):
        return "alternatives_ruled_out must be a list"
    return None

def investigate(sid, seed=0, max_steps=MAX_STEPS, effort="low", llm=llm_call, save=True):
    sess = Session(make_scenario(sid, seed))
    history = [{"role": "user", "content": "Investigate this experiment.\n" + json.dumps(sess.task_info())}]
    trace, seen, n_exp, n_diag, tokens = [], set(), 0, 0, 0
    diagnosis, stop = None, "no_submit"

    for step in range(1, max_steps + 1):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        est = int((len(json.dumps(msgs)) + len(json.dumps(TOOLS))) / 3.5)   # rough token estimate
        force = step == max_steps or est > PROMPT_LIMIT                      # code forces the end
        resp = llm(msgs, tools=TOOLS, tool_choice=FORCE if force else "auto",
                   tag=f"{sid}_s{seed}_step{step}", reasoning_effort=effort)
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

        if not calls:   # text-only turn: nudge, the step cap still applies
            trace.append({"step": step, "event": "no_tool_call", "text": (msg.content or "")[:300],
                          "prompt_tokens": u.prompt_tokens, "tokens": u.total_tokens})
            history.append({"role": "user", "content": "Call a diagnostic tool or submit_diagnosis."})
            continue

        done = False
        for i, tc in enumerate(calls):
            name, event, hyp, args = tc.function.name, "ok", "", None
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                event, content = "malformed_args", json.dumps({"error": "arguments were not valid JSON"})

            if event == "ok" and name == "submit_diagnosis":
                err = _validate(args)
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
                else:
                    seen.add(key)
                    out = sess.call(name, args)
                    content = out if isinstance(out, str) else json.dumps(out, default=str)
                    if '"error"' in content:
                        event = "tool_error"   # logged distinctly; a failed call doesn't burn an experiment
                    else:
                        n_diag += 1
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
                     "supporting_evidence": [], "alternatives_ruled_out": []}
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
    for e in d["alternatives_ruled_out"]: print("  -", e[:160])

if __name__ == "__main__":   # python -m agents.loop S1 0 low
    sid = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    show(investigate(sid, seed, effort=sys.argv[3] if len(sys.argv) > 3 else "low"))