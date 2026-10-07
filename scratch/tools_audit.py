import json
from environment.scenarios import SCENARIOS
from environment.runner import FORBIDDEN
from tools.diagnostics import Session, TOOL_NAMES
from tools.registry import TOOL_SCHEMAS

VERDICT = ("leak", "overfit", "imbalance", "shift", "bad", "problem", "suspicious", "likely", "cause", "fault")
ARGS = {"get_metrics": {"split": "val"}, "get_class_distribution": {"split": "val"},
        "analyze_confusion_matrix": {"split": "val"}}

print("schema chars:", len(json.dumps(TOOL_SCHEMAS)), "(tokens ~ chars/3.5)\n")

for sid in ("S6", "S2"):
    s = Session(SCENARIOS[sid])
    print(f"=== {sid} task_info ===\n{json.dumps(s.task_info())}")
    for t in TOOL_NAMES[:-1]:
        print(f"--- {t}\n{json.dumps(s.call(t, ARGS.get(t, {})))}")
    ch = {"max_depth": 2} if s.task_info()["model"] == "tree" else {"class_weight": "balanced"}
    print(f"--- run_experiment\n{json.dumps(s.call('run_experiment', {'changes': ch}))}\n")

print("AUDIT: scenario x tool, max chars and verdict words")
for sid, sc in SCENARIOS.items():
    s = Session(sc)
    assert not any(w in json.dumps(s.task_info()) for w in FORBIDDEN), f"{sid} task_info leaks"
    ch = {"max_depth": 2} if sc.base_config.model == "tree" else {"class_weight": "balanced"}
    sizes, hits = {}, []
    for t in TOOL_NAMES:
        out = s.call(t, {"changes": ch} if t == "run_experiment" else ARGS.get(t, {}))
        txt = json.dumps(out["result"])
        sizes[t] = len(txt)
        hits += [(t, w) for w in VERDICT if w in txt.lower()] + [(t, w) for w in FORBIDDEN if w in txt]
    assert "error" in s.call("get_metrics", {"split": "test"})["result"], "test reachable!"
    assert "error" in s.call("nonexistent")["result"] and "error" in s.call("get_metrics", {"bad": 1})["result"]
    print(f"{sid}: max={max(sizes.values())} chars, total={sum(sizes.values())}, verdict-word hits={hits or 'none'}")