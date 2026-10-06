import os, json
from dotenv import load_dotenv
from groq import Groq
from agents.llm import llm_call

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"  # pick from Groq's current models page

# --- a dummy tool: the real function ---
def get_class_distribution(split):
    return {"split": split, "counts": {"0": 90, "1": 10}}

REGISTRY = {"get_class_distribution": get_class_distribution}

# --- what the model sees: name + description + JSON schema ---
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_class_distribution",
        "description": "Returns per-class sample counts for the given data split.",
        "parameters": {
            "type": "object",
            "properties": {"split": {"type": "string", "enum": ["train", "val"]}},
            "required": ["split"],
        },
    },
}]

def run(prompt):
    print("\n" + "=" * 60 + f"\nPROMPT: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    r = llm_call(messages, tools=TOOLS, tag="smoke")
    choice = r.choices[0]
    msg = choice.message
    print("finish_reason:", choice.finish_reason)
    print("usage:", r.usage)

    if not msg.tool_calls:
        print("DIRECT ANSWER:", msg.content)
        return

    # 1) append the assistant message with its tool_calls, as received
    messages.append({
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [{
            "id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        } for tc in msg.tool_calls],
    })

    # 2) run each requested tool, answer each tool_call_id
    for tc in msg.tool_calls:
        print("TOOL CALL:", tc.id, tc.function.name, tc.function.arguments)
        args = json.loads(tc.function.arguments)   # it's a JSON *string*
        result = REGISTRY[tc.function.name](**args)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result),         # must be a string
        })

    # 3) call again with the full history
    r2 = llm_call(messages, tools=TOOLS, tag="smoke")
    
    print("FINAL finish_reason:", r2.choices[0].finish_reason)
    print("FINAL ANSWER:", r2.choices[0].message.content)
    print("usage (2nd call):", r2.usage)

run("How many samples are in each class in the training split?")  # should call the tool
run("In one sentence, what is overfitting?")                       # should NOT call it