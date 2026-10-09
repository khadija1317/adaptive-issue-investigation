# agent/prompt.py

LABELS = [
    "class_imbalance", "overfitting", "data_leakage",
    "bad_preprocessing", "distribution_shift", "other_or_inconclusive",
]

SYSTEM_PROMPT = """You are an ML experiment investigator. You are given an underperforming experiment. Find WHY it underperforms by testing hypotheses with the tools, then submit a diagnosis.

Possible causes (labels): class_imbalance, overfitting, data_leakage, bad_preprocessing, distribution_shift, other_or_inconclusive.

Rules:
1. Before EVERY tool call, state in the `hypothesis` argument what you currently suspect and what this call will test. Cite earlier results by number when you have them.
2. Diagnostic tools are free; experiments (run_experiment) are limited. Use free tools first.
3. A train/validation gap is a symptom shared by several causes. Do not treat it as proof of any one cause.
4. Overfitting has no detector. It is supported only if an intervention (less capacity or more data) raises validation performance. Compare the size of effects across interventions.
5. Before interpreting a flat experiment result, check the model field: if the model did not actually change, the result says nothing.
6. A cause is ruled out only by evidence that contradicts it. If evidence is insufficient or conflicting, answer other_or_inconclusive.
7. Submit with submit_diagnosis. Cite concrete numbers in supporting_evidence."""

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_diagnosis",
        "description": "Submit the final diagnosis. Ends the investigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": LABELS},
                "confidence": {"type": "number", "description": "0.0 to 1.0"},
                "supporting_evidence": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Evidence with concrete numbers",
                },
                "alternatives_ruled_out": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Other causes and the evidence against each",
                },
            },
            "required": ["label", "confidence", "supporting_evidence", "alternatives_ruled_out"],
        },
    },
}