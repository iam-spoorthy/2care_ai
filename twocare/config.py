"""Settings come from the environment, optionally loaded from a local .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    agent_model: str
    patient_model: str
    judge_model: str
    improver_model: str
    seed: int
    cache: bool
    timezone: str


def settings() -> Settings:
    return Settings(
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        agent_model=os.environ.get("AGENT_MODEL", "qwen/qwen3.8-27b"),
        patient_model=os.environ.get("PATIENT_MODEL", "openai/gpt-oss-20b"),
        judge_model=os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b"),
        improver_model=os.environ.get("IMPROVER_MODEL", "openai/gpt-oss-120b"),
        seed=int(os.environ.get("LLM_SEED", "7")),
        cache=os.environ.get("LLM_CACHE", "1") != "0",
        timezone=os.environ.get("CLINIC_TIMEZONE", "Asia/Kolkata"),
    )
