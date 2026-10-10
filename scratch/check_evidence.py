from evaluation.eval_meta import EVAL_META
from tools.registry import TOOL_SCHEMAS
TOOLS = {s["function"]["name"] for s in TOOL_SCHEMAS}
EFFECTS = {None, "up", "down", "moved", "no_gain"}
for sid, m in EVAL_META.items():
    for key in ("evidence_true", "evidence_decoy"):
        for alt in m[key]:
            for a in alt:
                assert a["tool"] in TOOLS and a["val"] in EFFECTS, (sid, key, a)
                assert (a["tool"] == "run_experiment") == (a["when"] is not None), (sid, key, a)
    assert (m["decoy"] is None) == (m["evidence_decoy"] == []), sid
    print(sid, len(m["evidence_true"]), "true alts,", len(m["evidence_decoy"]), "decoy alts")