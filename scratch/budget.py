# scratch/budget.py
import json, time, datetime as dt
rows = [json.loads(l) for l in open("logs/llm_calls.jsonl", encoding="utf-8")]
def ts(r):
    v = r["ts"]
    return float(v) if not isinstance(v, str) else dt.datetime.fromisoformat(v).timestamp()
now = time.time()
live = sorted((ts(r) + 86400, r["total_tokens"]) for r in rows if ts(r) + 86400 > now)
total = sum(n for _, n in live)
head = 200000 - total
print("in rolling window:", total, "| headroom now:", head)
marks = [30000, 60000, 100000]
for when, n in live:
    head += n
    while marks and head >= marks[0]:
        print(f"headroom {marks.pop(0)} from", dt.datetime.fromtimestamp(when).strftime("%a %H:%M"))