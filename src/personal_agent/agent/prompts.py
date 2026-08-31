"""Prompts for agent — kept separate for versioning."""

SYSTEM_BASE = """You are Personal Agent, a privacy-first thinking partner.
You have access to retrieved personal memories (with provenance).
Never invent personal history. Expose uncertainty and assumptions.
Support decisions, do not make life decisions for the user."""

def build_chat_prompt(user_message: str, memory_context: str, assessment_context: str = "") -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_BASE}]
    if memory_context:
        msgs.append({"role": "system", "content": memory_context})
    if assessment_context:
        msgs.append({"role": "system", "content": assessment_context})
    msgs.append({"role": "user", "content": user_message})
    return msgs
