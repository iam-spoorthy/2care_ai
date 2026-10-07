"""LLM-played patient. It knows only its persona, its facts, and its goal.

It never sees the agent's prompt or the expected outcome, so it cannot steer
the agent toward a pass. The conversation is passed as a plain transcript in a
single user message rather than as alternating chat roles: with alternating
roles, gpt-oss models sometimes answered in the receptionist's voice or emitted
stray tool calls.
"""

from __future__ import annotations

from ..config import settings
from ..llm import ToolCallFormatError, chat

END = "<END>"

TEMPLATE = """You write the next line for a patient who is phoning a clinic's scheduling receptionist. Today is {today_label}.

Who the patient is: {persona}
What the patient knows: {facts}
The patient's goal: {goal}
How the patient behaves:
{behaviour}

Rules:
- Output only the patient's next spoken line: 1 or 2 short sentences, plain text, no labels.
- Give the phone number or date of birth only when the receptionist asks for it.
- Never invent facts the patient was not given. If asked for something unknown, say you don't have it.
- When the goal is done, the receptionist has ended the call, or there is nothing left to say, output exactly {end}."""


class PatientSimulator:
    def __init__(self, scenario: dict, today_label: str, seed: int):
        facts = scenario.get("facts") or {}
        fact_text = "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in facts.items()) or "nothing beyond the goal"
        self.system = TEMPLATE.format(
            today_label=today_label,
            persona=scenario["persona"],
            facts=fact_text,
            goal=scenario["goal"],
            behaviour="\n".join(f"- {item}" for item in scenario.get("behaviour", [])),
            end=END,
        )
        self.lines = [f"PATIENT: {scenario['opening']}"]
        self.seed = seed
        self.finished = False

    def reply(self, agent_text: str) -> str | None:
        if self.finished:
            return None
        self.lines.append(f"RECEPTIONIST: {agent_text}")
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": "Conversation so far:\n" + "\n".join(self.lines) + "\n\nWrite the patient's next line."},
        ]
        message = None
        for offset in range(3):
            try:
                message = chat(
                    model=settings().patient_model,
                    messages=messages,
                    purpose="patient",
                    temperature=0.0,
                    seed=self.seed + offset,
                    max_tokens=400,
                )
                break
            except ToolCallFormatError:
                if offset == 2:
                    raise
        text = (message["content"] or "").strip().split("\n")[0].strip()
        for label in ("PATIENT:", "Patient:"):
            if text.startswith(label):
                text = text[len(label):].strip()
        if END in text:
            self.finished = True
            text = text.replace(END, "").strip()
        if not text:
            self.finished = True
            return None
        self.lines.append(f"PATIENT: {text}")
        return text
