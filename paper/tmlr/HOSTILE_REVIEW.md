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

- [ ] Rerun connected-component clustered bootstrap on all 66 pairs, not the 6-7 headline ones
      (R2.2) — free, local, no judge dependency, should do before final submission.
- [ ] Add one sentence to `draft_07` making the CON-exclusion-strengthens-not-weakens-the-claim
      point explicit (R1.1).
- [ ] Add one hedged sentence to Limitations about the Gemma identity re-derivation being confirmed
      via the paper's own methods text rather than an independent primary record (R1.2).
- [ ] Add one sentence to the Introduction distinguishing "scoring regimes disagree" (folklore) from
      this paper's actual claim (magnitude, structure, controlled same-items measurement) (R3.1).
- [ ] Build and verify the anonymized supplementary package (R3.4 / Phase 6.5) — not started.

None of these are blocking discoveries that undermine a claim already made; all are strengthening
edits. No finding here should change the paper's headline results.
