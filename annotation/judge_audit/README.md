# Human adjudication — instructions

`adjudication_sheet.csv` has 238 rows in two groups (see the `sample_group` column):

- **174 `disagreement` rows** — every REG/NUM/TMP item where Gemini 2.5 Flash's original judge
  verdict and phi4-mini's cross-model judge verdict disagreed. These are the only items that can
  tell you which judge is wrong on a given case.
- **64 `control_agreement` rows** — a stratified sample (8 items per stratum, fixed random seed,
  built by `scripts/build_adjudication_control_sample.py`) of cases where the available scoring
  methods *agree*: for REG/NUM/TMP this means Gemini and phi4-mini both judged the item incorrect
  the same way (`strict_incorrect_judges_agree`), or the item was strict-correct and phi4-mini did
  not dispute it (`strict_correct_phi4_agrees`, no Gemini verdict exists for these since Gemini only
  ever judged strict failures). For CON — which no judge reviews at all, strict or otherwise — the
  two control strata are direct spot-checks of strict-correct and strict-incorrect items
  (`CON_strict_*_no_judge`), since CON's exact Yes/No scoring has never been human-verified anywhere
  in this pipeline. See the `stratum` column for the exact pool each row was drawn from.

**What this sample is and isn't.** 238 items, stratified but not randomly sampled from the full
4,128-judgement space, is evidence about specific, informative cases — not a basis for a
population-level judge error rate. Report results from this sheet as human
adjudication/agreement evidence (e.g. "of the N control-agreement items a human reviewed, M matched
the automated verdict"), never as an estimated error rate over all items. The disagreement rows in
particular are deliberately oversampled relative to their true frequency (174/874 disagree, but the
full 4,128-item judgement space is >95% agreement by construction) precisely because they are the
most informative cases, not a representative draw.

## What to do

For each row, read `question`, `reference_answer`, and `model_prediction`, then fill in
**`human_verdict_CORRECT_or_INCORRECT`** with exactly `CORRECT` or `INCORRECT` — your own judgment
of whether `model_prediction` correctly answers `question` given `reference_answer`, using the same
standard the paper's judge rubric uses (semantically correct even with different formatting,
verbosity, or units — wrong only if the actual fact/number/date is wrong). `human_notes` is optional,
for anything you want to remember about a tricky call.

You don't have to do all 238 in one sitting — the CSV can be filled incrementally and re-run through
`scripts/score_adjudication.py` (write this once you're ready to score) at any point.

## Why this matters for the paper

Once filled in, this gives direct human-adjudication evidence on:
1. The 174 disagreement cases: which automated judge (Gemini or phi4-mini) was right, per item —
   compare `human_verdict` to `gemini_verdict` and `phi4_verdict`.
2. The 64 control-agreement cases: whether agreement between methods also means agreement with a
   human reader, including the two CON strata, which no judge or prior human check has ever touched.
3. Whether either judge shows a directional bias on the cases it's wrong about (the paper's
   manual-audit-of-28 already found Gemini over-credits correct intermediate reasoning that reaches
   a wrong final answer on NUM items — check if phi4-mini shares that bias or has a different one).

This is the one piece of the TMLR resubmission that cannot be automated — every other check in
`PRE_SUBMISSION_AUDIT.py` currently passes; this is the only failing one, and it's failing correctly.
