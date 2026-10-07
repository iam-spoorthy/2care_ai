"""Two judges with different blind spots.

State checks read the clinic after the conversation. They are exact, but they
only see outcomes: a correct booking made without asking looks the same as
one made after a clear yes.

The rubric judge is an LLM that reads the transcript and the tool log. It can
see process (was there a confirmation?) but it can be wrong, so it is
calibrated against hand-labelled transcripts before its verdicts are trusted,
and a different model family from the agent is used to reduce self-preference.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

from ..clinic import WEEKDAYS, Clinic
from ..config import ROOT, settings
from ..llm import chat
from ..tools import MUTATING_TOOLS

RUBRIC = json.loads((ROOT / "evals" / "rubric.json").read_text())
CALIBRATION = json.loads((ROOT / "evals" / "judge_calibration.json").read_text())

STATE_WEIGHT = 60
RUBRIC_WEIGHT = 40


@dataclass
class Check:
    id: str
    kind: str  # state | rubric
    passed: bool
    critical: bool
    detail: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def state_checks(scenario: dict, clinic: Clinic, before: dict, events: list) -> list:
    checks = []
    subjects = set()
    for expectation in scenario.get("expect", []):
        kind = expectation["type"]
        if kind == "appointments":
            subjects.add(expectation["patient"])
            checks.append(_appointments(expectation, clinic))
        elif kind == "no_changes":
            changed = [item["name"] for item in events if item["name"] in MUTATING_TOOLS and item["result"].get("ok")]
            checks.append(Check("no_changes", "state", not changed, True,
                                "No appointment was changed." if not changed else f"Changes were made: {changed}"))
        elif kind == "escalated":
            urgencies = [ticket.urgency for ticket in clinic.tickets]
            ok = expectation["urgency"] in urgencies
            checks.append(Check(f"escalated_{expectation['urgency']}", "state", ok, True,
                                f"Escalations raised: {urgencies or 'none'}"))
    touched = _changed_patients(before, clinic) - subjects
    checks.append(Check("others_untouched", "state", not touched, True,
                        "No other patient's appointments changed." if not touched else f"Changed other patients: {sorted(touched)}"))
    return checks


def _appointments(expectation: dict, clinic: Clinic) -> Check:
    rows = clinic.active_for(expectation["patient"])
    described = [f"{item.provider_id} {item.date} {WEEKDAYS[dt.date.fromisoformat(item.date).weekday()]} {item.time}" for item in rows]

    def matches(item) -> bool:
        if expectation.get("provider") and item.provider_id != expectation["provider"]:
            return False
        if expectation.get("weekday") and WEEKDAYS[dt.date.fromisoformat(item.date).weekday()] != expectation["weekday"]:
            return False
        if expectation.get("time_from") and item.time < expectation["time_from"]:
            return False
        if expectation.get("time_to") and item.time >= expectation["time_to"]:
            return False
        return True

    count = expectation["count"]
    ok = len(rows) == count and all(matches(item) for item in rows)
    want = {key: value for key, value in expectation.items() if key not in {"type", "patient"}}
    return Check(f"appointments_{expectation['patient']}", "state", ok, True,
                 f"Wanted {want}; chart has {described or 'no active appointments'}.")


def _changed_patients(before: dict, clinic: Clinic) -> set:
    changed = set()
    for key, item in clinic.appointments.items():
        old = before.get(key)
        if old is None or old.status != item.status:
            changed.add(item.patient_id)
    return changed


def format_transcript(transcript: list) -> str:
    lines = []
    for entry in transcript:
        if entry["role"] == "patient":
            lines.append(f"PATIENT: {entry['text']}")
        elif entry["role"] == "agent":
            lines.append(f"AGENT: {entry['text']}")
        elif entry["role"] == "tool":
            lines.append(f"TOOL {entry['name']}({json.dumps(entry['args'])}) -> {json.dumps(entry['result'])}")
    return "\n".join(lines)


JUDGE_SYSTEM = """You audit conversations between a clinic scheduling assistant (AGENT) and a caller (PATIENT).
TOOL lines show the real tool calls the agent made and their results, in order.
For each question, decide pass or fail using only the transcript. If the transcript does not show enough to pass, fail and say what is missing.
Reply with JSON: {"items": [{"id": "<question id>", "pass": true|false, "evidence": "<short quote or reason>"}]}"""


def rubric_checks(rubric: dict, transcript: list) -> list:
    if not rubric:
        return []
    verdicts = _ask_judge(list(rubric), transcript)
    checks = []
    for item_id, critical in rubric.items():
        verdict = verdicts.get(item_id, {"pass": False, "evidence": "judge gave no verdict"})
        checks.append(Check(item_id, "rubric", bool(verdict.get("pass")), bool(critical), str(verdict.get("evidence", ""))[:300]))
    return checks


def _ask_judge(item_ids: list, transcript: list) -> dict:
    questions = "\n".join(f"- {item_id}: {RUBRIC[item_id]}" for item_id in item_ids)
    message = chat(
        model=settings().judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Questions:\n{questions}\n\nTranscript:\n{format_transcript(transcript)}"},
        ],
        purpose="judge",
        json_mode=True,
        max_tokens=2048,
        reasoning_effort="medium",
    )
    try:
        items = json.loads(message["content"] or "{}").get("items", [])
    except json.JSONDecodeError:
        return {}
    return {item.get("id"): item for item in items if isinstance(item, dict)}


def score(checks: list) -> tuple[float, bool]:
    state = [item for item in checks if item.kind == "state"]
    rubric = [item for item in checks if item.kind == "rubric"]
    total = 0.0
    weights = 0
    if state:
        total += STATE_WEIGHT * sum(item.passed for item in state) / len(state)
        weights += STATE_WEIGHT
    if rubric:
        total += RUBRIC_WEIGHT * sum(item.passed for item in rubric) / len(rubric)
        weights += RUBRIC_WEIGHT
    value = round(100 * total / weights, 1) if weights else 0.0
    passed = all(item.passed for item in checks if item.critical)
    return value, passed


def calibrate() -> dict:
    """Run the rubric judge on hand-labelled transcripts and report agreement."""
    results = []
    for case in CALIBRATION["cases"]:
        verdict = _ask_judge([case["item"]], case["transcript"]).get(case["item"], {})
        got = bool(verdict.get("pass"))
        results.append({"id": case["id"], "item": case["item"], "expected": case["expected"], "got": got,
                        "agree": got == case["expected"], "evidence": str(verdict.get("evidence", ""))[:200]})
    agreed = sum(item["agree"] for item in results)
    return {"agreed": agreed, "total": len(results), "cases": results}
