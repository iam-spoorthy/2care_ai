"""Evaluate the agent, learn from its failures, and re-run.

    python3 run_eval.py                # start from prompts/CURRENT
    python3 run_eval.py --from v1      # start from a specific version
    python3 run_eval.py --step         # pause between stages
    python3 run_eval.py --trials 3     # majority vote over 3 runs per scenario

Stages: check the judge against human labels, run every scenario, collect
failures from the train split, ask the improver for a structured patch,
validate it, apply it as a new prompt version, re-run every scenario, and
accept the new version only if it fixes something and breaks nothing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from twocare.config import ROOT, settings
from twocare.evals import improver, judge, runner
from twocare.llm import LLMError, usage_summary
from twocare.prompts import PromptVersion, current_version, set_current

BOLD, DIM, GREEN, RED, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def stage(title: str, pause: bool) -> None:
    if pause:
        input(f"\n{DIM}Enter → {title}{RESET}")
    print(f"\n{BOLD}== {title} =={RESET}")


def progress(label: str) -> None:
    sys.stdout.write(f"\r{DIM}  running {label:<50}{RESET}")
    sys.stdout.flush()


def clear_progress() -> None:
    sys.stdout.write("\r" + " " * 70 + "\r")


def print_report(report: dict) -> None:
    clear_progress()
    for row in report["scenarios"]:
        mark = f"{GREEN}PASS{RESET}" if row["passed"] else f"{RED}FAIL{RESET}"
        failed = [check["id"] for check in row["checks"] if check["critical"] and not check["passed"]]
        note = f"  {DIM}failed: {', '.join(failed)}{RESET}" if failed else ""
        print(f"  {mark} {row['score']:>5}  {row['split']:<7} {row['id']:<22}{note}")
    overall, holdout = report["overall"], report["holdout"]
    print(f"\n  passed {BOLD}{overall['passed']}/{overall['count']}{RESET}   mean {overall['mean_score']}"
          f"   {DIM}(holdout {holdout['passed']}/{holdout['count']}){RESET}")


def print_failures(rows: list) -> None:
    for row in rows:
        print(f"  {RED}{row['scenario']}{RESET} · {row['check']} ({row['kind']})")
        print(f"    {row['detail']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate, improve, re-run.")
    parser.add_argument("--from", dest="start", help="prompt version to start from (default: prompts/CURRENT)")
    parser.add_argument("--step", action="store_true", help="pause between stages")
    parser.add_argument("--trials", type=int, default=1, help="runs per scenario; pass is a majority vote")
    parser.add_argument("--attempts", type=int, default=2, help="patch attempts before giving up")
    parser.add_argument("--max-judge-disagreements", type=int, default=0)
    args = parser.parse_args()

    config = settings()
    if not config.groq_api_key:
        sys.exit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    base = PromptVersion.load(args.start or current_version())
    print(f"{BOLD}2Care eval loop{RESET}  {DIM}agent={config.agent_model} patient={config.patient_model} "
          f"judge={config.judge_model} improver={config.improver_model} cache={'on' if config.cache else 'off'}{RESET}")

    record: dict = {"started_at": dt.datetime.now().isoformat(timespec="seconds"), "base_version": base.version}
    try:
        stage("1. Check the judge against hand-labelled transcripts", args.step)
        calibration = judge.calibrate()
        record["judge_calibration"] = calibration
        for case in calibration["cases"]:
            mark = f"{GREEN}agree{RESET}" if case["agree"] else f"{RED}DISAGREE{RESET}"
            print(f"  {mark}  {case['id']:<28} expected {'pass' if case['expected'] else 'fail'}")
        disagreements = calibration["total"] - calibration["agreed"]
        print(f"\n  judge agrees on {calibration['agreed']}/{calibration['total']}")
        if disagreements > args.max_judge_disagreements:
            print(f"  {RED}The judge disagrees with human labels, so its verdicts are not safe to learn from. Stopping.{RESET}")
            return finish(record, 2)

        stage(f"2. Run every scenario on prompt {base.version}", args.step)
        before = runner.run_suite(base, trials=args.trials, progress=progress)
        record["before"] = before
        print_report(before)

        stage("3. What the eval flagged", args.step)
        train_failures = runner.failures(before, split="train")
        holdout_failures = runner.failures(before, split="holdout")
        if not train_failures:
            print(f"  {GREEN}No train failures. Nothing to learn from.{RESET}")
            if holdout_failures:
                print(f"  {YELLOW}Holdout failures exist but are never shown to the improver:{RESET}")
                print_failures(holdout_failures)
            return finish(record, 0)
        print_failures(train_failures)
        if holdout_failures:
            print(f"\n  {DIM}{len(holdout_failures)} holdout failure(s) are hidden from the improver.{RESET}")

        feedback = None
        for attempt in range(1, args.attempts + 1):
            stage(f"4. Improver proposes a patch (attempt {attempt}/{args.attempts})", args.step)
            try:
                patch = improver.propose(base, train_failures, feedback)
                ids = {row["failure_id"] for row in train_failures}
                changes = improver.validate(base, patch, ids)
            except improver.PatchRejected as error:
                print(f"  {RED}Patch rejected by validator: {error}{RESET}")
                feedback = str(error)
                record.setdefault("attempts", []).append({"attempt": attempt, "rejected": str(error)})
                continue
            for item in patch.get("diagnosis", []):
                print(f"  root cause {DIM}{item.get('failure_ids')}{RESET}: {item.get('root_cause')}")
            for change in changes:
                print(f"  {GREEN}{change['op']}{RESET} {change['id']}: {change['text']}  {DIM}(fixes {', '.join(change['addresses'])}){RESET}")
            for item in patch.get("tool_recommendations", []):
                print(f"  {YELLOW}tool recommendation (needs human review, not applied){RESET}: {item}")
            candidate = improver.apply(base, changes, patch)

            stage(f"5. Re-run every scenario on candidate {candidate.version}", args.step)
            after = runner.run_suite(candidate, trials=args.trials, progress=progress)
            print_report(after)
            comparison = runner.compare(before, after)
            record.setdefault("attempts", []).append(
                {"attempt": attempt, "patch": patch, "changes": changes, "candidate": candidate.to_dict(),
                 "after": after, "comparison": comparison}
            )

            stage("6. Gate: fixed something, broke nothing?", args.step)
            for row in comparison["rows"]:
                arrow = f"{row['before']:>5} → {row['after']:<5}"
                if row["passed_before"] and not row["passed_after"]:
                    tag = f"{RED}REGRESSED{RESET}"
                elif not row["passed_before"] and row["passed_after"]:
                    tag = f"{GREEN}FIXED{RESET}"
                else:
                    tag = f"{DIM}{'pass' if row['passed_after'] else 'fail'}{RESET}"
                print(f"  {row['id']:<22} {row['split']:<7} {arrow} {tag}")
            b, a = comparison["before"], comparison["after"]
            print(f"\n  passed {b['passed']}/{b['count']} → {BOLD}{a['passed']}/{a['count']}{RESET}"
                  f"   mean {b['mean_score']} → {BOLD}{a['mean_score']}{RESET}")
            if comparison["regressions"]:
                feedback = f"it made these previously passing scenarios fail: {comparison['regressions']}"
                print(f"  {RED}Rejected: regressions in {', '.join(comparison['regressions'])}.{RESET}")
                continue
            if not comparison["fixed"]:
                feedback = "it did not fix any failing scenario"
                print(f"  {RED}Rejected: nothing was fixed.{RESET}")
                continue
            candidate.save()
            set_current(candidate.version)
            record["accepted"] = candidate.version
            print(f"  {GREEN}Accepted. Saved prompts/{candidate.version}.json and made it current.{RESET}")
            return finish(record, 0)

        print(f"\n  {YELLOW}No patch passed the gate. prompts/{base.version}.json stays current.{RESET}")
        return finish(record, 1)
    except LLMError as error:
        clear_progress()
        print(f"\n{RED}LLM error: {error}{RESET}")
        return finish(record, 3)


def finish(record: dict, code: int) -> None:
    record["usage"] = usage_summary()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = ROOT / "runs" / f"{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str))
    calls = {name: f"{value['calls']} live / {value['cached']} cached" for name, value in record["usage"].items()}
    print(f"\n{DIM}LLM calls: {calls}\nFull run record: runs/{path.name}{RESET}")
    sys.exit(code)


if __name__ == "__main__":
    main()
