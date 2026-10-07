"""Versioned system prompts. A version is a base template plus a list of rules.

The improvement loop only ever produces a new version file; it never edits an
existing one. `prompts/CURRENT` points at the version the agent uses.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field

from .clinic import WEEKDAYS, Clinic
from .config import ROOT

PROMPTS_DIR = ROOT / "prompts"
CURRENT_FILE = PROMPTS_DIR / "CURRENT"


@dataclass
class Rule:
    id: str
    text: str
    core: bool = False
    origin: str = "human"

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "core": self.core, "origin": self.origin}


@dataclass
class PromptVersion:
    version: str
    parent: str | None
    base: str
    rules: list
    changelog: list = field(default_factory=list)

    @classmethod
    def load(cls, version: str) -> "PromptVersion":
        raw = json.loads((PROMPTS_DIR / f"{version}.json").read_text())
        return cls(
            version=raw["version"],
            parent=raw.get("parent"),
            base=raw["base"],
            rules=[Rule(**item) for item in raw["rules"]],
            changelog=raw.get("changelog", []),
        )

    def save(self) -> None:
        path = PROMPTS_DIR / f"{self.version}.json"
        if path.exists():
            raise FileExistsError(f"{path.name} already exists; versions are immutable")
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "parent": self.parent,
            "base": self.base,
            "rules": [rule.to_dict() for rule in self.rules],
            "changelog": self.changelog,
        }

    def render(self, clinic: Clinic) -> str:
        today = clinic.today
        calendar = []
        for offset in range(0, clinic.window_days + 1):
            day = today + dt.timedelta(days=offset)
            label = "today" if offset == 0 else WEEKDAYS[day.weekday()]
            calendar.append(f"{day.isoformat()} {label}")
        rules = "\n".join(f"{index}. {rule.text}" for index, rule in enumerate(self.rules, start=1))
        return (
            self.base.format(clinic=clinic.name, today=today.isoformat(), weekday=today.strftime("%A"))
            + "\n\nCalendar (use it to turn weekday names into dates):\n"
            + ", ".join(calendar)
            + "\n\nRules:\n"
            + rules
        )


def current_version() -> str:
    return CURRENT_FILE.read_text().strip()


def set_current(version: str) -> None:
    CURRENT_FILE.write_text(version + "\n")


def next_version_name() -> str:
    numbers = [int(path.stem[1:]) for path in PROMPTS_DIR.glob("v*.json") if path.stem[1:].isdigit()]
    return f"v{max(numbers, default=0) + 1}"
