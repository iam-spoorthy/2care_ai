# 2Care front desk

A patient-appointment scheduling agent, an evaluation harness that scores it against the appointment table, and an improvement loop that closes once with a before/after result.

Python 3.9+, standard library only. The clinic is synthetic and the clock is fixed at Monday 12 October 2026.

## Run the agent

```bash
python3 run_agent.py                                    # policy v1, before the patch
python3 run_agent.py --policy v2                        # policy v2, after the patch
python3 run_agent.py --scenario defer_after_preference  # replay a scripted caller and judge it
```

Try phone `9811111111` and date of birth `7 July 1990`. Each turn prints the tool calls and the session state.

## Run the eval loop

```bash
python3 run_eval.py          # all stages at once
python3 run_eval.py --step   # pause between stages
```

It runs twelve scenarios on v1, flags the failures, turns them into a structured patch, applies it, re-runs the same twelve, and rejects the patch if anything that passed before now fails.

Result: mean 88.3 → 100, critical passes 10/12 → 12/12, no regressions. Full evidence in `runs/loop.json`.

## Layout

| File | What it does |
| --- | --- |
| `src/clinic.py` | Synthetic patients, doctors, slots, and the appointment table |
| `src/tools.py` | The tools the agent may call, with scope enforced in code |
| `src/policy.py` | Versioned rules: base safety rules, the v1 rule, the v2 rules |
| `src/nlu.py` | Deterministic reader for what the caller said |
| `src/agent.py` | The multi-turn agent |
| `src/scenarios.py` | Twelve scenarios: three happy paths, nine hard cases |
| `src/usersim.py` | Simulated caller that answers the agent's questions |
| `src/judge.py` | Scores the chart, the tool log, and the words; also a transcript-only judge for comparison |
| `src/improve.py` | Turns a failure code into a structured patch and applies it |
| `src/harness.py` | Plays scenarios and compares two runs |

Design note: `docs/DESIGN.md`.
