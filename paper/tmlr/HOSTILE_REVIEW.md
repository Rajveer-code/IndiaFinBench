# Hostile review — Phase 7

Three adversarial passes on the stable (non-judge-dependent) content, instructed to find reasons to
reject rather than to praise. Objection → evidence → fixed/claim-reduced, per the plan's own rule:
ship only when every serious objection has one or the other.

## Reviewer 1 — Methodology

**R1.1 — "Your CON exclusion from judging is convenient, not principled."** A skeptical reader could
say excluding CON from BOTH judges is exactly the task where the strict pipeline is least likely to
be wrong (binary Yes/No), so excluding it inflates the appearance of judge disagreement elsewhere by
removing the one task where the two scoring regimes would agree by construction.
**Response:** This is backwards, not evasive — CON's exclusion *removes* a source of trivial
agreement, so its absence makes the reported disagreement rate more conservative, not less. **Status:
addressed by the existing framing in Section 4.3; add one sentence making this explicit rather than
leaving the reader to work it out.** — `[TODO: draft_07, Scoring subsection]`

**R1.2 — "gemma3:4b vs the original 'gemma4' — how do we know THIS was the intended model, not
just the first one that happened to be pullable?"** Real objection: the re-run decision rested on
`/api/show` confirming the tag matches the paper's own *methods text* ("Gemma 4 E4B... 4B
parameters"), which is circular if that methods text was written to match whatever was run, not
independently.
**Response:** Not fully resolved — this is a genuine residual uncertainty, honestly stated in
`NUMBERS.md` §8 already ("could not verify from prior logs"). **Verdict: PLAUSIBLE, not fixable
further without primary records that don't exist. State the uncertainty in the paper's Limitations,
not just the ledger** — currently the paper states the Ollama tag as fact with no hedge.
`[TODO: draft_09 Limitations, one sentence]`

**R1.3 — "A 3.8B cross-model judge auditing much larger models — why would this be evidence rather
than noise?"** Legitimate methodological question about judge competence vs. judge independence.
**Response:** Already addressed directly — Section `sec:judge` explicitly declines to call phi4-mini
"independent" for exactly this reason, and the human-adjudication oversampling of disagreement cases
is the designed answer. **STATUS: addressed, no action.**

## Reviewer 2 — Statistical validity

**R2.1 — "twelve models is not enough to trust a Spearman/Kendall correlation; say so more
strongly."** Already done (`draft_02`/`draft_03`: "with twelve models these tests have little
power"). **STATUS: addressed.**

**R2.2 — "the connected-component clustering only tests 6-7 pre-selected comparisons, not all 66 —
cherry-picking which ones to re-test under clustering."** Real objection. The six/seven headline
pairs were chosen because they're the ones stated as findings in the paper, but a reviewer can
reasonably ask why the FULL 66-pair table wasn't reprocessed under clustering, especially since the
paper's own claim (Bonferroni: 15/66 survivors) is a whole-table statistic, not spot-checked.
**Response: this is a real gap. Either (a) run all 66 pairs under clustering — cheap, ~10min, no
judge dependency — or (b) explicitly scope the claim to "the comparisons this paper reports as
findings" rather than implying full-table validation.** `[ACTION: rerun clustered bootstrap on all
66 pairs before final submission — see follow-up task below]`

**R2.3 — "the 27-component clustering treats a REG item and a CON item on the same document as
equally dependent, which overstates the correction — REG items on document A are not obviously more
correlated with each other than with a REG item on document B."** Fair theoretical point about the
strength of item-document dependence, not just its existence.
**Response:** The paper doesn't claim the dependence structure is uniform — only that SOME
dependence exists and ignoring it entirely (item-level bootstrap) is the more indefensible default.
**STATUS: addressed by existing hedged framing ("not necessarily independent draws"); no overclaim
to walk back.**

## Reviewer 3 — Novelty, construction, reproducibility

**R3.1 — "Is 'strict scoring vs LLM-judge scoring disagree' actually novel, or well-known folklore
in the eval community?"** The single most dangerous objection for this paper at TMLR.
**Response:** The paper's actual contribution is not "judges and string-match disagree" (folklore) —
it is the *magnitude and structure* of the disagreement (11→1 rank reversal, one model), the
*effective-size* finding (independently interesting), and the fact that BOTH are measured on the
SAME 406 items under controlled conditions with a domain chosen specifically to block memorization.
**Action: the Introduction should state this distinction explicitly rather than let a reviewer
assume the weaker claim.** `[TODO: draft_02, one sentence after the first paragraph]`

**R3.2 — "Single annotator, personal-network human evaluators, no compensation — this reads like a
side project, not a rigorous benchmark."** Blunt but real: reviewers do form impressions from
methodology framing, independent of the actual rigor.
**Response:** The honesty here (already in the Ethics statement) is the correct choice and should
NOT be walked back or softened — TMLR's stated criterion rewards accurate disclosure over polish.
**STATUS: accept the risk; the alternative (obscuring participant recruitment) is worse.**

**R3.3 — "406 items is small for an LLM benchmark in 2026."** Standard objection to any
expert-annotated benchmark.
**Response:** Already pre-empted directly in the Dataset Construction section (comparison to
FinanceBench's 150 items, explicit quality-over-quantity framing, IAA coverage at 44.3% "exceeds
what most comparable benchmarks report"). **STATUS: addressed.**

**R3.4 — "Is the released code/data actually reproducible, or just claimed to be?"**
**Response: cannot fully verify until the anonymized supplementary package is built (Phase 6.5,
outstanding). This is the single largest remaining compliance gap for TMLR specifically (anonymity
requirement) — see NUMBERS.md and the plan's Phase 6.**

---

## Consolidated action items from this pass

- [x] Rerun connected-component clustered bootstrap on all 66 pairs, not the 6-7 headline ones
      (R2.2) — done (`scripts/provenance_and_clustered_bootstrap.py`, all 66 pairs, 100k resamples,
      5-seed stability check; Appendix A.3, Table 8).
- [x] Add one sentence to `draft_07`/`draft_08` making the CON-exclusion-strengthens-not-weakens
      point explicit (R1.1) — done (draft_08: "conservative for the disagreement rates we report
      below, not convenient... removes a source of trivial agreement rather than manufacturing
      disagreement elsewhere").
- [x] Add one hedged sentence to Limitations about the Gemma identity re-derivation being confirmed
      via the paper's own methods text rather than an independent primary record (R1.2) — done
      (draft_09: "not fully independent confirmation... flag the residual uncertainty rather than
      overstating it as settled fact").
- [x] Add one sentence to the Introduction distinguishing "scoring regimes disagree" (folklore) from
      this paper's actual claim (magnitude, structure, controlled same-items measurement) (R3.1) —
      done (draft_02, opening of paragraph 2).
- [ ] Build and verify the anonymized supplementary package (R3.4 / Phase 6.5) — not started, needs
      Phase 9 final assembly.

None of these were blocking discoveries that undermined a claim already made; all were strengthening
edits, now landed except the supplementary package.

---

## Addendum — 2026-09-03, second pass (post three-regime reframe, judge-truncation fix, matched-budget
re-run)

Everything above predates the three-regime reframe and is now historical. This pass targets what's
actually new since then: the retired "judge-audited" terminology, the judge-input-truncation fix and
its 52 verdict flips, the IAA rescoring, and the new matched-budget confound section.

**R4.1 — "You found and fixed four stale numbers in your own Discussion section this pass alone
(Gemini-phi4 raw agreement, three worked rank examples). If a careful re-read still turns up drift
after this many verification passes, how confident should a reviewer be that nothing else drifted?"**
Real question, not deflectable with "we checked again." **Response:** The honest answer is these
were caught precisely *because* of systematic re-verification (cross-checking every worked example
against the regenerated JSON directly, not re-reading prose for internal plausibility) — the process
that catches drift is the same process a reviewer should trust *more*, not less, once shown it works.
The alternative (claiming zero drift without having looked) would be the actual red flag. **STATUS:
addressed by disclosure, not fixable further — this is inherent to iterative correction, and the
audit script's regression guards (now pinning gemini\_vs\_phi4\_agreement.json alongside
regime\_three\_way.json and iaa\_summary.json) exist specifically so this class of drift can't
reappear silently on the next data regeneration.**

**R4.2 — "Your matched-budget re-run has an unexplained +10.59pp outlier (DeepSeek) right next to a
result you're using to support your central thesis. Doesn't an unexplained anomaly undermine
confidence in the other nine numbers in the same table?"**
*(Note: this table's "original" column had a bug — found and fixed 2026-09-03 — where the original
side was live-rescored against write-time-truncated CSV text instead of using the already-correct
full-text score. Every delta in the table changed as a result, most by under 1pp but LLaMA-3.3-70B's
delta shrank from +3.70 to +0.99. The corrected at-cap-rate correlation is r=-0.67, not the originally
reported r=0.09 — see draft_03 Section 5.1 and Appendix G.2 for the current, correct numbers and
interpretation. This note records that the underlying data changed; it does not change the answer to
R4.2 itself.)*
**Response:** The nine non-outlier deltas are individually small, mixed-sign, and each traces to a
disclosed original-budget value in the same table (models with a low original budget gain, models
already at or above 512 don't) — the outlier is flagged as an outlier precisely because it does not
fit that pattern, which is evidence the other nine *are* behaving as expected, not evidence they're
suspect too. **STATUS: addressed by explicit reporting of both the pattern and the exception; no
further action — resolving DeepSeek's specific mechanism would need a live A/B test against the
original OpenRouter response, which is not reproducible after the fact.**

**R4.3 — "Independent adjudication was scoped into this revision but the blind sample sits unfilled.
Does citing a 62-item stratified sample that hasn't been adjudicated read as claiming more progress
than has actually happened?"**
**UPDATE 2026-09-04 — resolved with real data, not just avoided.** Two independent, non-author
adjudicators (recruited by the author, blind to model identity, judge verdicts, and each other's
answers) completed the full 62-item sample. Reported in Appendix~F.7: 82.3% pairwise agreement
between them ($\kappa=0.652$), 87.1%/82.3% agreement with the author's own earlier verdicts on the
same items ($\kappa=0.739$/$0.647$) — all three comparisons in the "substantial agreement" band by
Landis & Koch (1977) convention — and both independently reproduce the author's own
51.6%/48.4% Gemini/phi4-mini split on disputed items almost exactly (51.6%/48.4% and 53.2%/46.8%).
The Limitations section (Section 8) points to this from the DeepSeek-specific single-adjudicator
caveat, and the wording discipline from the original response is preserved explicitly in the new
appendix text: this is reported as *replication of a disagreement pattern*, never as *validation of
either judge*. **STATUS: resolved. Original response (superseded, kept for the record below): the
manuscript made no claim about independent adjudication results anywhere — Section 6.5's wording
rule (never "the judge was validated by humans") already guarded against overclaiming, and the
existing 238-item sample was consistently labelled author adjudication throughout, so there was
nothing for a reviewer to read as premature even before real data existed.**

**R4.4 — "Four of your matched-budget re-run's models moved from Groq to OpenRouter mid-project
because Groq deprecated them. Is 'the same model' via a different inference provider actually the
same measurement?"**
Legitimate infrastructure-stability objection, not fully answerable with existing evidence.
**Response:** For open-weight models this is a real, disclosed risk (Appendix G.1 states it
explicitly for Kimi K2's dated snapshot) rather than an assumption. We cannot verify bit-identical
serving behaviour across providers post hoc. **STATUS: PLAUSIBLE, disclosed, not resolvable further
without re-running the original provider, which no longer serves these models at all — this is
exactly the DeepSeek-outlier discussion's second candidate explanation, generalised.**

No finding in this addendum changes a headline result. Two items (R4.1, R4.4) are disclosed
limitations rather than fixes; the rest were already addressed by the manuscript's existing framing.

---

## Addendum — 2026-09-04, external cross-check (ChatGPT review, two rounds against the live PDF)

User ran an independent review against the actual compiled manuscript (not from memory) and
compared it to this repo's own audit. Two rounds: round 1 flagged items mostly already resolved by
this point (see the resolved-vs-open triage in that session); round 2 reacted to the page-compression
pass and surfaced three genuinely new, actionable points.

**R5.1 — overclaim in the strict-vs-judge framing.** "What the strict number cannot do is answer
which model reasons best about the underlying regulatory content" is too absolute: strict scoring
isn't zero-evidence about reasoning, it conflates reasoning-correctness with format-compliance.
**Fixed**: reworded to "What the strict number cannot isolate is underlying regulatory correctness
from reference-format compliance" (`draft_03_results_core.tex`). Real, applied.

**R5.2 — "neither regime is more valid" should be stated more forcefully.** Checked: this is already
explicit in two places (`draft_03` Section 5 intro: "Neither regime is more correct than the
other... a benchmark that reports only one has silently chosen which question to answer"; Discussion
7.1: the judge's verdict "is its own imperfect measurement, not a ground truth the strict number
falls short of"). Adding a third restatement would reintroduce the kind of repetition the
page-compression pass just removed. **Declined, with reasoning given** — not a gap, already
addressed twice.

**R5.3 — abstract should reorder to separate the DeepSeek illustration from the main claim.**
Checked against the actual abstract text: DeepSeek already sits inside the same paragraph as the
"no positive rank correspondence" claim it illustrates, not elevated to its own spot. Separating them
would break flow (reader hits the claim, jumps to an unrelated topic, then jumps back for the
example). **Declined, with reasoning given.**

**R5.4 — related-work refresh with genuinely current (2025-2026) citations**, superseding this
session's earlier "skip it, can't verify ChatGPT's citation IDs" call. Independently re-searched
(not trusting the earlier list) and verified two real papers via direct arXiv fetch (title, authors,
abstract all confirmed, not just a matching ID):
- Ho, Huang, Boudin, Aizawa (2025), "Reassessing Extractive QA Datasets at Scale: LLM-as-a-Judge and
  In-Depth Analyses," arXiv:2504.11972 — EM/F1 correlate weakly with humans (0.22/0.40) vs.
  LLM-judge up to 0.85 on extractive QA specifically.
- Norman, Rivera, Hughes (2026), "Reliability without Validity: A Systematic, Large-Scale Evaluation
  of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias," arXiv:2606.19544 — raw judge
  agreement overstates discriminative reliability once corrected for chance.

Both added to `indiafinbench.bib` and cited in `draft_04_related_work.tex`'s "How benchmarks are
scored" paragraph. Compiles clean, 0 undefined citations, bibliography entries verified rendering
correctly in `main.bbl`. **Note for the record**: this session's earlier blanket dismissal of
ChatGPT-suggested citations as "unverified/risky" was too cautious — two of the originally-cited
arXiv IDs turned out to be real papers on independent re-check. The right response to an unverified
citation is to verify it, not to discard it by default.

**Self-correction, logged for transparency (per R5's own meta-point about trusting "done" claims):**
this session initially reported the manuscript's "pre-registered decision rule" paragraph as absent,
based on a grep for `"preregist"` that missed the actual hyphenated text `"pre-registered"`. Caught
and corrected mid-session when the paragraph was found (and tightened) in `draft_08`'s Section 6.5.
The paragraph itself was already well-handled (honest disclosure, no overclaim) — the process gap
was in the initial verification step, not the manuscript.

No finding in this addendum changes a headline result. R5.1 and R5.4 are real, applied fixes; R5.2
and R5.3 were checked and declined with stated reasoning, not skipped.

---

## Addendum — 2026-09-04, judge-tolerance sensitivity (R6)

**R6 — "the judge rubric's 1% rounding tolerance is stated with no sensitivity check, and this
paper's whole thesis is that scoring-rule parameters matter."** Real gap, flagged independently by
both the author's own review and the external ChatGPT cross-check. Closed with data, not argument:
re-ran the full-coverage phi4-mini judge (4,128 REG/NUM/TMP judgements) at two additional
tolerances -- exact numeric match and 0.1% -- under otherwise identical conditions, and compared
against the existing 1% run. New Appendix F.2, Table 11 (`scripts/judge_tolerance_comparison.py`,
`evaluation/judge_tolerance_comparison.json`).

**Finding, in two parts that cut in different directions:**
1. The paper's central claim is robust to the tolerance choice: strict-vs-judge-only shows no
   significant rank correspondence at any of the three tolerances ($\rho = -0.32, -0.13, -0.27$,
   all $p > 0.3$), and the judge-only spread stays in a similar 7.6-9.1pp band throughout.
2. Individual models' judge-only *ranks* are more tolerance-sensitive than the headline spread
   suggests -- 7 of 12 models move 4+ ranks across the three tolerances (LLaMA-3.3-70B moves 7
   ranks). DeepSeek-R1-Distill's own judge-only rank moves 2 -> 4 -> 7 across exact/0.1%/1%, though
   it never approaches its strict rank of 12. Under judge-augmented scoring specifically (the
   paper's more dramatic "reversal" framing), DeepSeek is rank 1 or 2 at every tolerance tested
   (98.0-98.8%) -- a precision check caught an early draft overstating this as "ties for 1st at
   every tolerance," which is false at exact tolerance (DeepSeek is 1st outright there, no tie);
   corrected before this landed anywhere final.

**STATUS: resolved, and the result reinforces rather than weakens the paper.** Framed explicitly as
such in the new appendix section: the finding is not "1% was a bad choice," it is that scoring-rule
sensitivity extends one level deeper than strict-vs-judge alone -- a single numeric rubric parameter
moves individual ranks by several positions even within one fixed judge. One-sentence pointer added
to the main-text judge section (Section 6.5) so a main-body reader knows this check exists without
main body absorbing the appendix's full word count.

No finding in this addendum changes a headline result; it strengthens the central one.
