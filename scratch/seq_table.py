# scratch/seq_table.py
import json, glob

def lab(t):
    return t["tool"] if t["event"] in ("ok", "submit") else f"{t['tool']}[{t['event']}]"

for f in sorted(glob.glob("logs/traces/*.json")):
    r = json.load(open(f))
    seq = [lab(t) for t in r["trace"] if t.get("tool") not in (None, "-")]
    print(f"{r['scenario']} s{r['seed']} {r['effort']} | {r['diagnosis']['label']} | {r['stop_reason']} | " + " > ".join(seq))