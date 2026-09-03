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
**Response:** The manuscript makes no claim about independent adjudication results anywhere — Section
6.5's wording rule (never "the judge was validated by humans") already guards against this, and the
existing 238-item sample is consistently labelled author adjudication throughout. **STATUS: no
overclaim exists to fix; the sample's existence is not mentioned in the manuscript at all, only in
the repository (`annotation/independent_adjudication/`), so there is nothing for a reviewer to read
as premature.**

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
