# scratch/fake_loop_test.py
import json
from types import SimpleNamespace as NS
from agents.loop import investigate, show

GOOD = json.dumps({"label": "class_imbalance", "confidence": 0.8,
                   "supporting_evidence": ["fake"],"recommended_config": {}, "alternatives_ruled_out": [{"cause": "overfitting", "evidence": "fake"}]})
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
from agents.loop import make_scenario
from tools.diagnostics import Session
s1 = Session(make_scenario("S1"))   # a real session, so the config check has a model and feature names
assert _validate(bad, s1) is not None and _validate(json.loads(GOOD), s1) is None
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
assert _validate({**json.loads(GOOD), "alternatives_ruled_out": ["overfitting"]}, s1) is not None
print("VALIDATE-2 OK")


_H = '"hypothesis": "h"'   # own helpers for Run H, so the script's existing names are untouched
_gv = lambda s: ("get_metrics", "{" + _H + f', "split": "{s}"}}')

def sub(cfg):
    return ("submit_diagnosis", json.dumps({"label": "class_imbalance", "confidence": 0.8, "supporting_evidence": ["x"],
            "alternatives_ruled_out": [], "recommended_config": cfg}))
rh = investigate("S1", 0, llm=make_fake([[_gv("val")], [_gv("train")], [("compare_train_validation", "{" + _H + "}")],
        [sub({"max_depth": 3})], [sub({"drop_features": ["nope"]})], [sub("balanced")], [sub({"class_weight": "balanced"})]]), save=False)
evh = [t["event"] for t in rh["trace"]]
assert evh.count("invalid_submit") == 3 and evh[-1] == "submit"
print("RUN H ok (bad fixes rejected: tree knob on logreg, unknown feature, non-object)")

import agents.loop as L
_sleep = L.time.sleep; L.time.sleep = lambda s: None   # no real waiting in the test
def conn_flaky(script, n_fail):
    base, st = make_fake(script), {"n": 0}
    def f(*a, **k):
        st["n"] += 1
        if st["n"] <= n_fail: raise RuntimeError("Connection error.")
        return base(*a, **k)
    return f
ri = investigate("S1", 0, llm=conn_flaky(script_c, 2), save=False)
assert ri["stop_reason"] == "submitted" and "llm_error" not in [t["event"] for t in ri["trace"]]
rj = investigate("S1", 0, llm=conn_flaky(script_c, 3), save=False)
assert rj["stop_reason"] == "llm_error"
L.time.sleep = _sleep
print("RUN I ok (transient errors retried; persistent ones still end the run)")
# Run J: output_parse_failed (model wrote prose instead of a tool call) is recovered like tool_use_failed
def flaky_msgs(script, msgs):
    base, st = make_fake(script), {"n": 0}
    def f(*a, **k):
        st["n"] += 1
        if st["n"] <= len(msgs):
            raise RuntimeError(msgs[st["n"] - 1])
        return base(*a, **k)
    return f
PARSE = "Error code: 400 - output_parse_failed"
TOOLF = "Error code: 400 - tool_use_failed"
rj1 = investigate("S1", 0, llm=flaky_msgs(script_c, [PARSE, PARSE]), save=False)
assert rj1["stop_reason"] == "submitted", rj1["stop_reason"]
rj2 = investigate("S1", 0, llm=flaky_msgs(script_c, [TOOLF, PARSE]), save=False)    # the real S4 sequence
assert rj2["stop_reason"] == "submitted", rj2["stop_reason"]
rj3 = investigate("S1", 0, llm=flaky_msgs(script_c, [PARSE, PARSE, PARSE]), save=False)   # persistent: still ends
assert rj3["stop_reason"] == "llm_error", rj3["stop_reason"]
rj4 = investigate("S1", 0, llm=flaky_msgs(script_c, ["Error code: 401 - invalid_api_key"]), save=False)   # other errors: not nudged
assert rj4["stop_reason"] == "llm_error" and rj4["llm_calls"] == 1
print("RUN J ok (output_parse_failed recovered; persistent and unrelated errors still end the run)")
