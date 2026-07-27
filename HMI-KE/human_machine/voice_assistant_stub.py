from __future__ import annotations


def parse_text_command(text: str) -> dict:
    text = text.strip()
    if "确认" in text:
        return {"intent": "confirm_action", "raw_text": text}
    if "风险" in text:
        return {"intent": "query_risk", "raw_text": text}
    if "建议" in text:
        return {"intent": "query_recommendation", "raw_text": text}
    return {"intent": "unknown", "raw_text": text}
