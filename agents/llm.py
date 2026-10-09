import os, json, time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-120b"
LOG_PATH = Path("logs/llm_calls.jsonl")

# FILL THESE from your Groq limits page for this model (don't guess):
RPM_LIMIT = 30
TPM_LIMIT = 8000
DAILY_TOKEN_BUDGET = 200_000
WARN_FRACTION = 0.8
SAFETY = 0.9                 # pace to 90% of the per-minute caps

assert RPM_LIMIT and TPM_LIMIT and DAILY_TOKEN_BUDGET, "Fill in the limits in agent/llm.py"


LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_recent = deque()   # (timestamp, tokens) for the last 60s
_last_total = 500   # token estimate for the next call


def _today():
    return datetime.now(timezone.utc).date().isoformat()   # UTC; check Groq's reset time in the headers


def _load_today_total():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    total = 0
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if datetime.fromisoformat(rec["ts"]) > cutoff:
                    total += rec["total_tokens"]
    return total


_daily_total = _load_today_total()   # survives restarts because it's read from the log


def _pace(est_tokens):
    while True:
        now = time.time()
        while _recent and now - _recent[0][0] > 60:
            _recent.popleft()
        tokens_in_window = sum(t for _, t in _recent)
        too_many_reqs = len(_recent) >= RPM_LIMIT * SAFETY
        too_many_tokens = tokens_in_window + est_tokens > TPM_LIMIT * SAFETY
        if not (too_many_reqs or too_many_tokens) or not _recent:
            return
        wait = 60 - (now - _recent[0][0]) + 0.5
        print(f"[pacing] sleeping {wait:.1f}s")
        time.sleep(max(wait, 0.5))


def llm_call(messages, tools=None, tool_choice="auto", tag="", reasoning_effort="low"):
    global _daily_total, _last_total
    if _daily_total >= DAILY_TOKEN_BUDGET:
        raise RuntimeError(f"Daily token budget reached ({_daily_total}). Switch provider or wait.")

    _pace(_last_total)

    kwargs = dict(model=MODEL, messages=messages, temperature=0, reasoning_effort=reasoning_effort)
    if tools:
        kwargs.update(tools=tools, tool_choice=tool_choice)

    t0 = time.time()
    for attempt in range(3):
        try:
            raw = client.chat.completions.with_raw_response.create(**kwargs)
            break
        except RateLimitError as e:
            wait = float(e.response.headers.get("retry-after", 20))
            print(f"[429] attempt {attempt+1}, waiting {wait}s")
            time.sleep(wait + 1)
    else:
        raise RuntimeError("Rate-limited 3 times in a row")

    resp = raw.parse()
    u = resp.usage
    details = getattr(u, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None

    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "date": _today(),
        "tag": tag,
        "model": MODEL,
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "reasoning_tokens": reasoning,
        "total_tokens": u.total_tokens,
        "finish_reason": resp.choices[0].finish_reason,
        "latency_s": round(time.time() - t0, 3),
        "headers": {k: v for k, v in raw.headers.items()
                    if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"},
    }

    _recent.append((time.time(), u.total_tokens))
    _last_total = u.total_tokens
    _daily_total += u.total_tokens
    rec["daily_total"] = _daily_total
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    if _daily_total >= WARN_FRACTION * DAILY_TOKEN_BUDGET:
        print(f"[WARNING] daily tokens at {_daily_total}/{DAILY_TOKEN_BUDGET}")

    return resp