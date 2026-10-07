"""Turn judged failures into a structured prompt patch, validate it, apply it.

The improver model sees only train-split failures. It never sees holdout
scenarios, so holdout scores measure whether a patch generalizes.

Guards on what a patch may do:
- only add rules or edit non-core rules; core rules cannot be removed or edited
- each change must say which failure it addresses
- a bounded number of short, general rules
- no scenario ids, patient names, or phone numbers (overfitting to the test set)
Tool-level recommendations are recorded for a human, never auto-applied,
because changing backend guarantees should go through code review.
"""

from __future__ import annotations

import json
import re

from ..config import ROOT, settings
from ..llm import chat
from ..prompts import PromptVersion, Rule, next_version_name
from .judge import RUBRIC, format_transcript

MAX_CHANGES = 4
MAX_RULE_CHARS = 320
TRANSCRIPT_LINES = 30

SYSTEM = """You improve the system prompt of a clinic scheduling assistant using evidence from failed evaluation runs.

You will get the current rules and a list of failures. Each failure has the check that failed, the judge's reason, and the transcript (PATIENT, AGENT, and TOOL lines).

Find the root cause behind the failures and propose the smallest set of general rule changes that would fix them without breaking behaviour that already works.

Constraints:
- Write rules as short, general instructions for any caller. Never mention specific people, phone numbers, dates, or test names.
- You may add new rules or edit rules marked "core": false. You may not remove or edit core rules.
- At most {max_changes} changes. Each rule under {max_chars} characters.
- Each change must list the failure ids it addresses.
- If the right fix is in the backend tools rather than the prompt, put it in tool_recommendations instead of a rule.

Reply with JSON only:
{{
  "diagnosis": [{{"failure_ids": ["..."], "root_cause": "..."}}],
  "changes": [{{"op": "add", "id": "snake_case_id", "text": "...", "addresses": ["..."]}},
              {{"op": "edit", "id": "existing_rule_id", "text": "...", "addresses": ["..."]}}],
  "tool_recommendations": ["..."],
  "regression_risks": ["..."]
}}"""


class PatchRejected(ValueError):
    pass


def propose(prompt: PromptVersion, failures: list, feedback: str | None = None) -> dict:
    items = []
    for index, failure in enumerate(failures, start=1):
        failure_id = f"F{index}"
        failure["failure_id"] = failure_id
        lines = format_transcript(failure["transcript"]).splitlines()[-TRANSCRIPT_LINES:]
        question = RUBRIC.get(failure["check"], "The clinic's final state did not match the expected outcome.")
        items.append(
            f"[{failure_id}] check={failure['check']} ({failure['kind']})\n"
            f"What was checked: {question}\n"
            f"Judge's reason: {failure['detail']}\n"
            f"Transcript:\n" + "\n".join(lines)
        )
    rules = json.dumps([rule.to_dict() for rule in prompt.rules], indent=2)
    user = f"Current rules:\n{rules}\n\nFailures:\n\n" + "\n\n".join(items)
    if feedback:
        user += f"\n\nYour previous patch was rejected: {feedback}\nPropose a different patch."
    message = chat(
        model=settings().improver_model,
        messages=[
            {"role": "system", "content": SYSTEM.format(max_changes=MAX_CHANGES, max_chars=MAX_RULE_CHARS)},
            {"role": "user", "content": user},
        ],
        purpose="improver",
        json_mode=True,
        max_tokens=4096,
        reasoning_effort="medium",
    )
    try:
        return json.loads(message["content"] or "{}")
    except json.JSONDecodeError as error:
        raise PatchRejected(f"improver returned invalid JSON: {error}") from None


def _forbidden_terms() -> list:
    clinic = json.loads((ROOT / "data" / "clinic.json").read_text())
    scenarios = json.loads((ROOT / "evals" / "scenarios.json").read_text())["scenarios"]
    terms = [scenario["id"] for scenario in scenarios]
    for patient in clinic["patients"]:
        terms += [patient["first_name"], patient["last_name"], patient["phone"]]
    return [term.lower() for term in terms]


def validate(prompt: PromptVersion, patch: dict, failure_ids: set) -> list:
    changes = patch.get("changes")
    if not isinstance(changes, list) or not changes:
        raise PatchRejected("patch has no changes")
    if len(changes) > MAX_CHANGES:
        raise PatchRejected(f"patch has {len(changes)} changes; the limit is {MAX_CHANGES}")
    existing = {rule.id: rule for rule in prompt.rules}
    forbidden = _forbidden_terms()
    clean = []
    for change in changes:
        op = change.get("op")
        rule_id = re.sub(r"[^a-z0-9_]", "_", str(change.get("id", "")).lower()).strip("_")
        text = str(change.get("text", "")).strip()
        addresses = [item for item in change.get("addresses", []) if item in failure_ids]
        if op not in {"add", "edit"}:
            raise PatchRejected(f"unsupported op {op!r}")
        if not rule_id or not text:
            raise PatchRejected("change is missing an id or text")
        if len(text) > MAX_RULE_CHARS:
            raise PatchRejected(f"rule {rule_id} is {len(text)} characters")
        if not addresses:
            raise PatchRejected(f"rule {rule_id} does not address any listed failure")
        lowered = text.lower()
        leaked = [term for term in forbidden if re.search(rf"\b{re.escape(term)}\b", lowered)]
        if leaked:
            raise PatchRejected(f"rule {rule_id} mentions test-specific details: {leaked}")
        if op == "edit":
            if rule_id not in existing:
                raise PatchRejected(f"cannot edit unknown rule {rule_id}")
            if existing[rule_id].core:
                raise PatchRejected(f"rule {rule_id} is core and cannot be edited")
        if op == "add" and rule_id in existing:
            raise PatchRejected(f"rule {rule_id} already exists")
        clean.append({"op": op, "id": rule_id, "text": text, "addresses": addresses})
    return clean


def apply(prompt: PromptVersion, changes: list, patch: dict) -> PromptVersion:
    rules = [Rule(rule.id, rule.text, rule.core, rule.origin) for rule in prompt.rules]
    by_id = {rule.id: rule for rule in rules}
    version = next_version_name()
    for change in changes:
        if change["op"] == "edit":
            by_id[change["id"]].text = change["text"]
            by_id[change["id"]].origin = f"improver:{version}"
        else:
            rules.append(Rule(change["id"], change["text"], core=False, origin=f"improver:{version}"))
    entry = {
        "version": version,
        "summary": "; ".join(item.get("root_cause", "") for item in patch.get("diagnosis", []))[:500],
        "changes": changes,
        "tool_recommendations": patch.get("tool_recommendations", []),
    }
    return PromptVersion(version, prompt.version, prompt.base, rules, prompt.changelog + [entry])
