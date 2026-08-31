# TMLR submission rules — verified 2026-08-31

Sources: [jmlr.org/tmlr](https://jmlr.org/tmlr/) · [author guide](https://jmlr.org/tmlr/author-guide.html) ·
[editorial policies](https://jmlr.org/tmlr/editorial-policies.html) ·
[acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html) ·
[reviewer guide](https://jmlr.org/tmlr/reviewer-guide.html)

Re-verify before submitting. FinNLP cost a desk rejection because the page rule was assumed, not read.

## Format — the thing that killed the last submission

| Rule | Value |
|---|---|
| Page limit | **NONE.** "Submissions may be any length, but a paper's length should be justified by its content." |
| Practical ceiling | **≤12 pages of main content (before references)** → reviews due in 2 weeks. **>12 pages** → reviews due in 4 weeks, and unusually long papers "are likely to result in reviewing delays." |
| Template | **Mandatory** [TMLR LaTeX stylefile](https://github.com/JmlrOrg/tmlr-style-file/archive/refs/heads/main.zip). PDF generated from it. |
| Appendix | Allowed, **after** the references, inside the same PDF. Reviewers read it at their discretion. |
| Supplementary | ≤100MB, **PDF or ZIP only**, must be anonymized. Discretionary reading. |
| Anonymity | Double-blind. Submission must be anonymized. |

**Opposite problem from FinNLP: nothing needs cutting. Target ≤12pp main content to stay on the
2-week review track.**

## Acceptance criteria — both must be YES

1. **Are the claims made in the submission supported by accurate, convincing and clear evidence?**
   The most important criterion. Any gap between claims and evidence must be closed — either by more
   evidence, *or by reducing the claims*.
2. **Would at least some individuals in TMLR's audience be interested in the findings, and does the
   paper communicate them clearly?** A reviewer unsure about this "should assume that it does."

**Explicitly NOT grounds for rejection:** not state-of-the-art; not "novel enough" (novelty of the
method is *not* a necessary criterion); modest contribution or significance.

**Grounds for rejection:** bold statements unsupported by evidence; unclear writing; incorrectly
claiming novelty over existing published work; merely re-implementing an already-reproduced idea.

## Desk rejection — what the Action Editor can bounce

AE assigned within 1 week and may reject immediately for:
- being out of scope
- **TMLR having insufficient reviewer expertise to adequately handle it**
- being poor quality / unlikely to meet the acceptance criteria
- **format violations** (contingent on Editor-in-Chief approval)

## Scope

"Original papers that contribute to the understanding of the computational and mathematical
principles that enable intelligence through learning." Explicitly invited, and the clause that
covers a benchmark paper:

> formalization of new learning tasks (e.g., in the context of new applications) **and of methods for
> assessing performance on those tasks**

Also invited: "accounts of applications of existing techniques that shed light on the strengths and
weaknesses of the methods"; "experimental and/or theoretical studies yielding new insight into the
design and behavior of learning in intelligent systems."

## Dual submission and originality — READ BEFORE SUBMITTING

> TMLR only accepts original contributions that don't reuse the authors' own prior work. In
> particular, we do not accept submissions that are expanded versions of conference papers. There
> should not be any reuse of written text, figures or results between the submitted paper and any
> paper which has been **published, accepted for publication, or submitted in parallel** at another
> archival, peer-reviewed venue.

**Acceptable** overlap: venues/tracks publicly declared **in writing** to be non-archival (workshops),
and preprint servers (arXiv, bioRxiv).

A **desk-rejected** submission was never published or accepted → not a bar. But a paper *currently
under submission* at an archival venue that shares results/figures/text **is** a bar.

## Authorship quota — scarce for a sole author

Generalized Harmonic Quota Rule with **N_1 = 2, N_9 = 9**. Budget spent per submission depends on
author count; a sole-authored submission is the most expensive.

- Sole author → reads as **~2 submissions per year**. Verify with the official
  [quota calculator](https://www.cs.cmu.edu/~nihars/quota/author.html?rule=generalized&N1=2&A=9&NA=9)
  before relying on it.
- Reviewers and Action Editors get **doubled** quotas (N_1 = 4).
- **Budget is spent even on submissions that are desk rejected.** A format-violation desk reject
  costs a whole slot.
- Circumventing quotas (duplicate accounts) risks a lifetime ban.

## Review process and timeline

1. Submit to [OpenReview](https://openreview.net/group?id=TMLR), anonymized; **authors recommend an
   Action Editor**.
2. AE assigned within 1 week; may desk-reject (above). Otherwise assigns **≥3 reviewers** and the
   paper becomes public. Members of the public may volunteer to review.
3. Reviews due **2 weeks** (≤12pp) / **4 weeks** (>12pp). Reviews hidden from public and other
   reviewers until all are in. Then open rebuttal/discussion/revision; reviewers submit a final
   recommendation 2 weeks after discussion starts, no later than 1 month.
4. Decision: **accept as is** / **accept with minor revision** / **reject** (AE states whether they
   would consider a significantly revised version).

Historical median time to decision: ~76–91 days.

Withdrawal is author-triggered any time before decision; the record stays public, marked withdrawn.

## Acceptance rates (community-reported)

| Year | Excluding desk rejects/withdrawals | Including them |
|---|---|---|
| 2025 | 70.6% | 46.3% |
| 2024 | — | 50.6% |
| 2023 | ~66% | ~50% |

2025's drop is attributed to a higher desk-rejection rate after an influx of weak submissions.

## Other

- **Licensing:** CC BY 4.0 from submission onward; authors retain copyright.
- **Preprints:** arXiv allowed at any time, anonymous or named — but the submission must not *link*
  to a version carrying the author names.
- **Broader Impact Statement:** required only if the work carries significant risk of harm.
- **OpenReview profile** must be complete and active (affiliations, conflicts, publication history).
- **Certifications:** Outstanding, Featured, Reproducibility, Survey.
- **Survey papers are no longer considered as of 2026-09-01.**

## Pre-submission checklist

- [ ] Built with the TMLR LaTeX stylefile, PDF output
- [ ] Main content ≤12 pages before references
- [ ] Fully anonymized (paper, supplementary, and any linked repo)
- [ ] No linked arXiv/OpenReview version carrying the author name
- [ ] No text/figure/result overlap with anything published, accepted, or *currently under submission*
      at an archival peer-reviewed venue
- [ ] Quota budget confirmed on the official calculator
- [ ] Action Editor nominated
- [ ] Every claim in the abstract traceable to evidence at the stated strength
- [ ] OpenReview profile current
