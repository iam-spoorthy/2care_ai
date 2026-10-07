"""Play scenarios end to end and judge them."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Callable

from ..agent import Agent
from ..clinic import Clinic
from ..config import ROOT, settings
from ..prompts import PromptVersion
from ..tools import ToolExecutor
from . import judge
from .patient import PatientSimulator

SCENARIOS_PATH = ROOT / "evals" / "scenarios.json"
MAX_TURNS = 10


def load_scenarios(path: Path = SCENARIOS_PATH) -> tuple[dt.date, list]:
    raw = json.loads(path.read_text())
    return dt.date.fromisoformat(raw["clinic_today"]), raw["scenarios"]


def run_scenario(prompt: PromptVersion, scenario: dict, today: dt.date, seed: int) -> dict:
    clinic = Clinic.load(today, scenario.get("setup_appointments"))
    before = clinic.snapshot()
    executor = ToolExecutor(clinic, faults=scenario.get("faults") or {})
    agent = Agent(prompt, clinic, executor, seed=seed)
    patient = PatientSimulator(scenario, today.strftime("%A %d %B %Y"), seed=seed)

    transcript = []
    errors = []
    said = scenario["opening"]
    for _ in range(MAX_TURNS):
        transcript.append({"role": "patient", "text": said})
        turn = agent.respond(said)
        for event in turn.tool_events:
            transcript.append({"role": "tool", **event})
        transcript.append({"role": "agent", "text": turn.reply})
        if turn.error:
            errors.append(turn.error)
        said = patient.reply(turn.reply)
        if said is None:
            break
    else:
        errors.append("turn limit reached")

    events = [entry for entry in transcript if entry["role"] == "tool"]
    checks = judge.state_checks(scenario, clinic, before, events)
    checks += judge.rubric_checks(scenario.get("rubric", {}), transcript)
    value, passed = judge.score(checks)
    return {
        "id": scenario["id"],
        "title": scenario["title"],
        "split": scenario.get("split", "train"),
        "score": value,
        "passed": passed,
        "checks": [item.to_dict() for item in checks],
        "transcript": transcript,
        "errors": errors,
    }


def run_suite(
    prompt: PromptVersion,
    trials: int = 1,
    only: list | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    today, scenarios = load_scenarios()
    if only:
        scenarios = [item for item in scenarios if item["id"] in only]
    base_seed = settings().seed
    results = []
    for scenario in scenarios:
        runs = []
        for trial in range(trials):
            if progress:
                progress(f"{scenario['id']} (trial {trial + 1}/{trials})")
            runs.append(run_scenario(prompt, scenario, today, seed=base_seed + trial))
        passes = sum(run["passed"] for run in runs)
        # Majority vote across trials; the first failing trial is kept as evidence.
        representative = next((run for run in runs if not run["passed"]), runs[0]) if passes * 2 <= trials else runs[0]
        representative = dict(representative)
        representative["passed"] = passes * 2 > trials
        representative["score"] = round(sum(run["score"] for run in runs) / trials, 1)
        representative["trial_passes"] = f"{passes}/{trials}"
        results.append(representative)
    return summarize(prompt.version, results)


def summarize(version: str, results: list) -> dict:
    def block(rows: list) -> dict:
        if not rows:
            return {"count": 0, "passed": 0, "mean_score": 0.0}
        return {
            "count": len(rows),
            "passed": sum(row["passed"] for row in rows),
            "mean_score": round(sum(row["score"] for row in rows) / len(rows), 1),
        }

    return {
        "prompt_version": version,
        "overall": block(results),
        "train": block([row for row in results if row["split"] == "train"]),
        "holdout": block([row for row in results if row["split"] == "holdout"]),
        "scenarios": results,
    }


def failures(report: dict, split: str | None = "train") -> list:
    rows = []
    for scenario in report["scenarios"]:
        if scenario["passed"] or (split and scenario["split"] != split):
            continue
        for check in scenario["checks"]:
            if check["critical"] and not check["passed"]:
                rows.append({
                    "scenario": scenario["id"],
                    "title": scenario["title"],
                    "check": check["id"],
                    "kind": check["kind"],
                    "detail": check["detail"],
                    "transcript": scenario["transcript"],
                })
    return rows


def compare(before: dict, after: dict) -> dict:
    old = {row["id"]: row for row in before["scenarios"]}
    rows = []
    regressions = []
    fixed = []
    for row in after["scenarios"]:
        prior = old.get(row["id"])
        if prior is None:
            continue
        if prior["passed"] and not row["passed"]:
            regressions.append(row["id"])
        if not prior["passed"] and row["passed"]:
            fixed.append(row["id"])
        rows.append({
            "id": row["id"],
            "split": row["split"],
            "before": prior["score"],
            "after": row["score"],
            "passed_before": prior["passed"],
            "passed_after": row["passed"],
        })
    return {
        "rows": rows,
        "fixed": fixed,
        "regressions": regressions,
        "before": before["overall"],
        "after": after["overall"],
        "holdout_before": before["holdout"],
        "holdout_after": after["holdout"],
    }
