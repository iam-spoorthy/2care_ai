# Design note

2Care is a front-desk agent and a harness. The agent books, moves, and cancels visits. The harness scores twelve callers on the appointment table, turns one failure into a patch, and runs the same twelve again.

The language layer is a deterministic tagger, not a sampled model, so a score change is the patch. The clinic clock is Monday 12 October 2026. Charts are synthetic. A phone number is not identity. Date of birth is the second factor, and it is a weak one. Emergency words route the caller to 108 or 112. They are not a diagnosis engine. Wording that misses the list is booked as an ordinary visit. That limit is real.

One interpreter reads a versioned policy. Base rules — emergency first, no override, no other chart, no diagnosis, second factor, confirm a cancellation, tell the truth when a tool fails, honor a correction — cannot be dropped. v1 adds one product rule: a named time is consent, to save a turn. A deferral in the same breath is ignored, and the book tool accepts a direct write. A line that only said “be careful” would leave that rule and that tool in place, so it was rejected.

The session remembers the verified patient, the intent, and at most one hold. Tools cannot take another patient’s id, and open slots do not carry other people’s names. The patch is the only way v2 exists: a back-off beats a yes, a named time is a hold, and a later “yes, book it” books from that hold. Direct booking then returns `DIRECT_BOOK_DISABLED`.

The judge is 50 points on the chart, 30 on the tool log, 20 on language. Only the chart is critical. Under v1 the premature book still scores the 30, because the call was legal. The chart fails, so the scenario scores 30. A shallow judge reads only the words. On these two scenarios it uses the happy-path rubric, where “you’re booked” is success. It passes v1 and fails v2, whose correct line is “I have not booked that.” The loop does not chase that judge. Three fixtures, not agent runs, mark the other blinds: a polite confirmation over a failed tool, a refusal after a silent lookup, and a ban on “you have” that false-fails “you have Friday at 10am”.

The improver keys on `PREMATURE_BOOK`, not on a scenario name. The deferral and the one-breath retraction both emit it, so one patch moves both from 30 to 100. It will not invent a repair for an unknown code, and it will not apply if a base rule would be dropped. Same suite, again: mean 88.3 to 100, critical passes 10/12 to 12/12, no regression.

The harness caller answers a structured `ask`. A frozen transcript cannot survive a new confirmation turn. If the agent asked only in prose, the simulator would stall. The chart judge would still see the damage.

AI help: Claude, inside Cursor, drafted the code, the PRD, and the UI. My judgment overrode it on: [fill in before submitting — for example, choosing a deterministic tagger over Groq so a score change is attributable to the patch; making the judge read the appointment table instead of trusting the reply; refusing to optimize the transcript-only score].
