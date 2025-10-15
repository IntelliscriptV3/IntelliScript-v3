import json
from langchain_openai import ChatOpenAI

_llm_eval = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

_PROMPT = """You are evaluating the quality of a database answer.

User asked: "{q}"
Agent answered: "{a}"

Return ONLY JSON with this schema:
{{
  "confidence": float,  // 0.0 to 1.0
  "reason": string
}}

Rules:
- If the answer admits uncertainty or says it cannot query the DB, score near 0.1.
- If the answer contains specific numbers/dates clearly derived from the DB, score >= 0.6.
- Penalize hallucinations or irrelevant content.
"""

def evaluate_confidence(q: str, a: str):
    try:
        res = _llm_eval.invoke(_PROMPT.format(q=q, a=a)).content
        data = json.loads(res)
        conf = float(data.get("confidence", 0.0))
        reason = data.get("reason", "")
    except Exception:
        conf, reason = 0.0, "parse_error"

    # simple heuristic penalty
    bad_phrases = ["I cannot", "not sure", "don’t have access", "unable to"]
    if any(p.lower() in a.lower() for p in bad_phrases):
        conf = min(conf, 0.3)
    return conf, reason
