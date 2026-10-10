# scratch/fake_loop_test.py
import json
from types import SimpleNamespace as NS
from agents.loop import investigate, show

GOOD = json.dumps({"label": "class_imbalance", "confidence": 0.8,
                   "supporting_evidence": ["fake"], "alternatives_ruled_out": [{"cause": "overfitting", "evidence": "fake"}]})
BAD_LABEL = json.dumps({"label": "banana", "confidence": 0.8,
                        "supporting_evidence": ["x"], "alternatives_ruled_out": []})
H = '"hypothesis": "h"'

def make_fake(script):
    it = iter(script)
    def fake(messages, tools=None, tool_choice="auto", tag="", reasoning_effort="low"):
        # like the real API: a forced tool_choice always yields that tool call
        turn = [("submit_diagnosis", GOOD)] if tools and len(tools) == 1 else next(it, [])
        calls = [NS(id=f"c{tag}{i}", function=NS(name=n, arguments=a)) for i, (n, a) in enumerate(turn)]
        msg = NS(content="", tool_calls=calls or None)
        return NS(usage=NS(total_tokens=100, prompt_tokens=80), choices=[NS(message=msg)])
    return fake

# Run A: every guard should fire once
script_a = [
    [("get_metrics", "{not json")],                                  # malformed_args
    [("fake_tool", "{" + H + "}")],                                  # unknown tool (Session error)
    [("submit_diagnosis", GOOD)],                                    # rejected_early (0 diagnostic calls)
    [],                                                              # no_tool_call (text-only turn)
    [("get_metrics", "{" + H + ', "split": "val"}'),
     ("get_metrics", "{" + H + ', "split": "val"}')],                # ok + duplicate (two calls in one turn)
    [("get_metrics", "{" + H + ', "split": "train"}')],              # ok
    [("compare_train_validation", "{" + H + "}")],                   # ok (3rd diagnostic call)
    [("submit_diagnosis", BAD_LABEL)],                               # invalid_submit
    [("submit_diagnosis", GOOD)],                                    # accepted
]
r = investigate("S1", 0, llm=make_fake(script_a), save=False)
show(r)
ev = [t["event"] for t in r["trace"]]
print("EVENTS:", ev)
for e in ["malformed_args", "tool_error", "rejected_early", "no_tool_call", "duplicate", "invalid_submit", "submit"]:
    assert e in ev, f"missing event: {e}"
assert any("unknown tool" in str(t.get("result")) for t in r["trace"]), "unknown tool not reported"
assert r["stop_reason"] == "submitted"
print("RUN A OK\n")

# Run B: the step cap forces a submit even though the script never submits
script_b = [[("get_metrics", "{" + H + ', "split": "val"}')]] * 5
rb = investigate("S1", 0, max_steps=3, llm=make_fake(script_b), save=False)
print("RUN B:", rb["stop_reason"], rb["llm_calls"], "calls")
assert rb["stop_reason"] == "forced_submit" and rb["llm_calls"] == 3
print("RUN B OK")


# Run C: provider 'tool_use_failed' errors are recovered from, not fatal
def flaky(script, n_fail=2):
    base, st = make_fake(script), {"n": 0}
    def f(*a, **k):
        st["n"] += 1
        if st["n"] <= n_fail:
            raise RuntimeError("Error code: 400 - tool_use_failed")
        return base(*a, **k)
    return f
script_c = [[("get_metrics", "{" + H + ', "split": "val"}')],
            [("get_metrics", "{" + H + ', "split": "train"}')],
            [("compare_train_validation", "{" + H + "}")],
            [("submit_diagnosis", GOOD)]]
rc = investigate("S1", 0, llm=flaky(script_c), save=False)
evc = [t["event"] for t in rc["trace"]]
print("RUN C:", rc["stop_reason"], evc)
assert rc["stop_reason"] == "submitted" and evc.count("llm_error") == 2
print("RUN C OK")


from agents.loop import _validate
bad = {"label": "data_leakage", "confidence": 0.5, "supporting_evidence": ["x"],
       "alternatives_ruled_out": [{"cause": "data_leakage", "evidence": "x"}]}
assert _validate(bad) is not None and _validate(json.loads(GOOD)) is None
print("VALIDATE OK")


# Run D: run_experiment is rejected until baseline val metrics have been read
EXP = "{" + H + ', "changes": {"class_weight": "balanced"}}'
script_d = [[("run_experiment", EXP)],
            [("get_metrics", "{" + H + ', "split": "val"}')],
            [("run_experiment", EXP)],
            [("get_metrics", "{" + H + ', "split": "train"}')],
            [("submit_diagnosis", GOOD)]]
rd = investigate("S1", 0, llm=make_fake(script_d), save=False)
evd = [t["event"] for t in rd["trace"]]
print("RUN D:", rd["stop_reason"], evd)
assert evd[0] == "rejected_no_baseline" and rd["stop_reason"] == "submitted" and rd["experiments"] == 1
print("RUN D OK")
assert _validate({**json.loads(GOOD), "alternatives_ruled_out": ["overfitting"]}) is not None
print("VALIDATE-2 OK")


# Run E: experiment results carry a factual identical_to_baseline flag
def exp(ch): return "{" + H + ', "changes": ' + json.dumps(ch) + "}"
script_e = [[("get_metrics", "{" + H + ', "split": "val"}')],
            [("run_experiment", exp({"scaling": "standard"}))],
            [("run_experiment", exp({"scaling": "per_split"}))],
            [("submit_diagnosis", GOOD)]]
re_ = investigate("S4", 0, llm=make_fake(script_e), save=False)
flags = [json.loads(t["result"])["result"]["identical_to_baseline"] for t in re_["trace"] if t.get("tool") == "run_experiment"]
print("RUN E:", flags)
assert flags[0]["val_metrics"] is True and flags[1]["val_metrics"] is False
print("RUN E OK")