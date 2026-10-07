"""The scheduling agent: an LLM in a tool-calling loop over one conversation.

Conversation state has two halves. The message history (what was said) lives
here and is sent to the model each turn. The authoritative state (who is
verified, what is booked) lives in the ToolExecutor and the clinic, so the
model cannot talk its way into a state the backend does not agree with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .clinic import Clinic
from .config import settings
from .llm import ToolCallFormatError, chat
from .prompts import PromptVersion
from .tools import TOOL_SCHEMAS, ToolExecutor

MAX_TOOL_ROUNDS = 6
FALLBACK_REPLY = (
    "Sorry, something went wrong on my side. I've passed this to the front desk and someone will call you back."
)


@dataclass
class Turn:
    reply: str
    tool_events: list = field(default_factory=list)
    error: str | None = None


class Agent:
    def __init__(self, prompt: PromptVersion, clinic: Clinic, executor: ToolExecutor, seed: int | None = None):
        self.prompt = prompt
        self.clinic = clinic
        self.executor = executor
        self.model = settings().agent_model
        self.seed = seed
        self.messages: list = [{"role": "system", "content": prompt.render(clinic)}]

    def respond(self, text: str) -> Turn:
        self.messages.append({"role": "user", "content": text})
        start = len(self.executor.events)
        for _ in range(MAX_TOOL_ROUNDS):
            try:
                message = self._call()
            except ToolCallFormatError as error:
                return self._fail(start, f"malformed tool call: {error}")
            calls = message["tool_calls"]
            if not calls:
                reply = (message["content"] or "").strip()
                self.messages.append({"role": "assistant", "content": reply})
                return Turn(reply, [item.to_dict() for item in self.executor.events[start:]])
            self.messages.append({"role": "assistant", "content": message["content"], "tool_calls": calls})
            for call in calls:
                result = self.executor.execute(call["function"]["name"], call["function"].get("arguments") or "{}")
                self.messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
                )
        return self._fail(start, "too many tool rounds in one turn")

    def _call(self) -> dict:
        # One retry with a different seed: malformed tool calls are usually transient.
        base_seed = self.seed if self.seed is not None else settings().seed
        for offset in (0, 1):
            seed = base_seed + offset
            try:
                return chat(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOL_SCHEMAS,
                    purpose="agent",
                    seed=seed,
                )
            except ToolCallFormatError:
                if offset == 1:
                    raise
        raise AssertionError("unreachable")

    def _fail(self, start: int, reason: str) -> Turn:
        self.executor.execute("escalate_to_human", {"reason": f"agent error: {reason}"[:300], "urgency": "routine"})
        self.messages.append({"role": "assistant", "content": FALLBACK_REPLY})
        return Turn(FALLBACK_REPLY, [item.to_dict() for item in self.executor.events[start:]], error=reason)
