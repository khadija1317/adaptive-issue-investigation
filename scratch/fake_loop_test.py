# scratch/fake_loop_test.py
import json
from types import SimpleNamespace as NS
from agents.loop import investigate, show

GOOD = json.dumps({"label": "class_imbalance", "confidence": 0.8,
                   "supporting_evidence": ["fake"], "alternatives_ruled_out": ["fake"]})
BAD_LABEL = json.dumps({"label": "banana", "confidence": 0.8,
                        "supporting_evidence": ["x"], "alternatives_ruled_out": []})
H = '"hypothesis": "h"'

def make_fake(script):
    it = iter(script)
    def fake(messages, tools=None, tool_choice="auto", tag="", reasoning_effort="low"):
        # like the real API: a forced tool_choice always yields that tool call
        turn = [("submit_diagnosis", GOOD)] if isinstance(tool_choice, dict) else next(it, [])
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