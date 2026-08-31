# Human adjudication — instructions

`adjudication_sheet.csv` has 174 rows: every item where Gemini 2.5 Flash's original judge verdict
and phi4-mini's cross-model judge verdict disagreed. These are the only items that can tell you
which judge is wrong on a given case — where they agree, you have no information about who's right,
just agreement.

## What to do

For each row, read `question`, `reference_answer`, and `model_prediction`, then fill in
**`human_verdict_CORRECT_or_INCORRECT`** with exactly `CORRECT` or `INCORRECT` — your own judgment
of whether `model_prediction` correctly answers `question` given `reference_answer`, using the same
standard the paper's judge rubric uses (semantically correct even with different formatting,
verbosity, or units — wrong only if the actual fact/number/date is wrong). `human_notes` is optional,
for anything you want to remember about a tricky call.

You don't have to do all 174 in one sitting — the CSV can be filled incrementally and re-run through
`scripts/score_adjudication.py` (write this once you're ready to score) at any point.

## Why this matters for the paper

Once filled in, this tells us three things no other check can:
1. **Gemini's true error rate** on the cases it got wrong (compare `human_verdict` to `gemini_verdict`).
2. **phi4-mini's true error rate** on the same cases (compare `human_verdict` to `phi4_verdict`).
3. Whether one judge is systematically better than the other, or they're both wrong in different
   directions (which is what the paper's manual-audit-of-28 already found for Gemini alone: it
   over-credits correct intermediate reasoning that reaches a wrong final answer — check if phi4-mini
   shares that bias or has a different one).

This is the one piece of the TMLR resubmission that cannot be automated — every other check in
`PRE_SUBMISSION_AUDIT.py` currently passes; this is the only failing one, and it's failing correctly.
