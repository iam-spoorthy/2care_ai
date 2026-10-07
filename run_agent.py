"""Talk to the scheduling agent.

    python3 run_agent.py                # uses prompts/CURRENT, today's real date
    python3 run_agent.py --prompt v1    # a specific prompt version
    python3 run_agent.py --today 2026-10-12
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from twocare.agent import Agent
from twocare.clinic import Clinic
from twocare.config import settings
from twocare.llm import LLMError
from twocare.prompts import PromptVersion, current_version
from twocare.tools import ToolExecutor

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def new_session(prompt: PromptVersion, today: dt.date) -> Agent:
    clinic = Clinic.load(today)
    return Agent(prompt, clinic, ToolExecutor(clinic))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the 2Care scheduling agent.")
    parser.add_argument("--prompt", help="prompt version (default: prompts/CURRENT)")
    parser.add_argument("--today", help="clinic date, YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    if not settings().groq_api_key:
        sys.exit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

    prompt = PromptVersion.load(args.prompt or current_version())
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    agent = new_session(prompt, today)
    print(f"{BOLD}2Care scheduling assistant{RESET} {DIM}· prompt {prompt.version} · clinic date {today}{RESET}")
    print(f"{DIM}Synthetic patients only, e.g. phone 9811111111, date of birth 7 July 1990. /reset starts a new call, /quit exits.{RESET}\n")

    while True:
        try:
            text = input("you   > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text == "/quit":
            return
        if text == "/reset":
            agent = new_session(prompt, today)
            print(f"{DIM}  new call{RESET}\n")
            continue
        try:
            turn = agent.respond(text)
        except LLMError as error:
            print(f"{DIM}  LLM error: {error}{RESET}\n")
            continue
        for event in turn.tool_events:
            print(f"{DIM}  tool {event['name']}({json.dumps(event['args'])}) -> {json.dumps(event['result'])[:220]}{RESET}")
        print(f"agent > {turn.reply}\n")


if __name__ == "__main__":
    main()
