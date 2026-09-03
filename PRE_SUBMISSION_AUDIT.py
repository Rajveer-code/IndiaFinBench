"""Single PASS/FAIL gate for TMLR submission (Phase 8 of the resubmission plan).

Run this before submitting and nothing else. Every check either passes or
prints exactly why it failed. No manual interpretation should be needed.
Checks are exact-equality where a real value is known; a "close enough" (>=)
check hides exactly the kind of silent regression this script exists to catch.

Usage: python PRE_SUBMISSION_AUDIT.py
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TEX_DIR = ROOT / "paper" / "tmlr"
SUB_DIR = TEX_DIR / "tmlr_submission"
MAIN_TEX = SUB_DIR / "main.tex"

failures = []
warnings = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def warn(name, detail=""):
    print(f"  [WARN] {name}" + (f" -- {detail}" if detail else ""))
    warnings.append(name)


print("=== SCIENTIFIC ===")

import sys as _sys
_sys.path.insert(0, str(ROOT))
from scripts.novel_methods_utils import MODEL_FILES, RESULTS_DIR  # noqa: E402

n_models = len(MODEL_FILES)
check("12 models in MODEL_FILES", n_models == 12, f"found {n_models}")

# Canonical item-id set from the dataset itself -- catches missing/substituted
# ids that a same-length, no-internal-dupes check alone would miss.
dataset = json.loads((ROOT / "annotation/raw_qa/indiafinbench_qa_combined_406.json").read_text(encoding="utf-8"))
if isinstance(dataset, dict):
    dataset = dataset.get("items", dataset.get("data", list(dataset.values())[0]))
canonical_ids = {item["id"] for item in dataset}
check("dataset has exactly 406 canonical item ids", len(canonical_ids) == 406, f"found {len(canonical_ids)}")
canonical_by_task = {}
for item in dataset:
    canonical_by_task.setdefault(item["task_type"], set()).add(item["id"])

all_406 = True
no_dupes = True
ids_match_canonical = True
total_fails = 0
for label, fname in MODEL_FILES.items():
    path = RESULTS_DIR / fname
    if not path.exists():
        check(f"results file exists: {fname}", False, "missing")
        all_406 = False
        continue
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if len(rows) != 406:
        all_406 = False
        warn(f"{label}: {len(rows)}/406 items")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        no_dupes = False
        warn(f"{label}: duplicate item ids")
    if set(ids) != canonical_ids:
        ids_match_canonical = False
        missing = canonical_ids - set(ids)
        extra = set(ids) - canonical_ids
        warn(f"{label}: id set does not match canonical dataset",
             f"missing={len(missing)} extra={len(extra)}")
    total_fails += sum(1 for r in rows if "FAIL" in r.get("prediction", ""))

check("all 12 models have exactly 406 items", all_406)
check("no duplicate item ids in any model", no_dupes)
check("every model's id set exactly matches the canonical dataset", ids_match_canonical)
# The paper documents 3 API failures (part of "61: 58 empty + 3 API failures" in
# Section 4.3) as an expected, scored-incorrect-by-design outcome, not a data bug.
check("FAIL-prediction count matches documented methodology (exactly 3)", total_fails == 3,
      f"found {total_fails}, paper documents 3 API failures -- investigate if this changed")

# Provenance: exactly 34 documents
prov_path = ROOT / "evaluation" / "provenance_summary.json"
if prov_path.exists():
    prov = json.load(open(prov_path, encoding="utf-8"))
    check("provenance: exactly 34 documents represented", prov["documents_represented_in_qa"] == 34,
          f"found {prov['documents_represented_in_qa']}")
else:
    check("provenance_summary.json exists", False)

# Gemma identity confirmed
gemma_path = RESULTS_DIR / "gemma4_e4b_results.csv"
if gemma_path.exists():
    rows = list(csv.DictReader(open(gemma_path, encoding="utf-8")))
    versions = {r.get("model_version", "") for r in rows}
    check("Gemma results use confirmed gemma3:4b identity, exactly", versions == {"gemma3:4b"},
          f"found versions: {versions}")

# Cross-model judge: exact full coverage, exact per-model coverage, exact id match
judge_dir = ROOT / "evaluation" / "results_judged_phi4"
reg_num_tmp_ids = canonical_by_task.get("regulatory_interpretation", set()) \
    | canonical_by_task.get("numerical_reasoning", set()) \
    | canonical_by_task.get("temporal_reasoning", set())
check("REG+NUM+TMP canonical ids total exactly 344", len(reg_num_tmp_ids) == 344,
      f"found {len(reg_num_tmp_ids)}")
if judge_dir.exists():
    judge_files = sorted(judge_dir.glob("*.csv"))
    total_judged = 0
    judge_all_344 = True
    judge_ids_match = True
    for f in judge_files:
        jrows = list(csv.DictReader(open(f, encoding="utf-8")))
        total_judged += len(jrows)
        if len(jrows) != 344:
            judge_all_344 = False
            warn(f"{f.name}: {len(jrows)}/344 judged items")
        if {r["id"] for r in jrows} != reg_num_tmp_ids:
            judge_ids_match = False
            warn(f"{f.name}: judged id set does not match canonical REG+NUM+TMP ids")
    check("cross-model judge: exactly 12 model files present", len(judge_files) == 12,
          f"found {len(judge_files)}")
    check("cross-model judge: exactly 344 items per model", judge_all_344)
    check("cross-model judge: every model's judged ids match canonical REG+NUM+TMP set", judge_ids_match)
    check("cross-model judge: exactly 4128-item total coverage", total_judged == 4128,
          f"{total_judged}/4128 -- STILL RUNNING or incomplete, this must be exactly 4128 before submission")
else:
    check("evaluation/results_judged_phi4/ exists", False, "judge has not been run")

# Document-clustered bootstrap: current, full 66-pair coverage
clustered_path = ROOT / "evaluation" / "clustered_bootstrap_full66.json"
if clustered_path.exists():
    clustered = json.load(open(clustered_path, encoding="utf-8"))
    check("clustered bootstrap: exactly 66 pairs (full table, not a subset)",
          clustered.get("n_pairs") == 66, f"found {clustered.get('n_pairs')}")
else:
    check("evaluation/clustered_bootstrap_full66.json exists", False)

# Human-adjudication sheet: correct structure (fill status is separately gated below)
adjud_path = ROOT / "annotation" / "judge_audit" / "adjudication_sheet.csv"
if adjud_path.exists():
    adjud_rows = list(csv.DictReader(open(adjud_path, encoding="utf-8")))
    groups = {r.get("sample_group", "") for r in adjud_rows}
    check("adjudication sheet: exactly 238 rows (174 disagreement + 64 control)",
          len(adjud_rows) == 238, f"found {len(adjud_rows)}")
    check("adjudication sheet: both sample groups present",
          groups == {"disagreement", "control_agreement"}, f"found groups: {groups}")
else:
    check("annotation/judge_audit/adjudication_sheet.csv exists", False)

# Figure/data consistency: every generated figure/table must be newer than the
# source data it was built from -- a stale copy is exactly the bug class this
# check exists to catch (fig_heatmap.png etc. sat 4+ hours stale, pre-Gemma-fix,
# undetected until a manual read caught it).
print("\n=== FIGURE/DATA CONSISTENCY ===")
SOURCE_FILES = list(RESULTS_DIR.glob("*.csv")) + [
    ROOT / "evaluation" / "task_accuracy_matrix.csv",
    ROOT / "evaluation" / "difficulty_breakdown.csv",
    ROOT / "evaluation" / "phi4_regime_table.json",
] + list((ROOT / "evaluation" / "results_judged_phi4").glob("*.csv"))
newest_source_mtime = max(f.stat().st_mtime for f in SOURCE_FILES if f.exists())

GENERATED_OUTPUTS = [
    SUB_DIR / "figures" / "fig_heatmap.png",
    SUB_DIR / "figures" / "fig_correlation.png",
    SUB_DIR / "figures" / "fig_difficulty.png",
    SUB_DIR / "figures" / "fig_radar.png",
    SUB_DIR / "figures" / "figure_regime_shift.png",
    SUB_DIR / "tables" / "table_regime.tex",
    SUB_DIR / "tables" / "table_errortax.tex",
]
for out in GENERATED_OUTPUTS:
    if out.exists():
        check(f"{out.name}: newer than all source data", out.stat().st_mtime >= newest_source_mtime,
              f"{out.name} predates a source file -- regenerate it")
    else:
        check(f"{out.name} exists", False)

print("\n=== THREE-REGIME CONSISTENCY (Plan v3 Phase 0) ===")
regime_path = ROOT / "evaluation" / "regime_three_way.json"
if regime_path.exists():
    regime = json.loads(regime_path.read_text(encoding="utf-8"))
    check("regime_three_way.json: 12 models", regime.get("n_models") == 12,
          f"found {regime.get('n_models')}")
    pm = regime.get("per_model", {})
    # Regression guard: judge_augmented is what earlier drafts computed as
    # "judge-audited" -- these numbers must not silently drift when
    # regime_table.py is rerun, independent of what the manuscript prose calls it.
    checks = [
        ("DeepSeek-R1-Distill", "strict_pct", 75.12), ("DeepSeek-R1-Distill", "judge_augmented_pct", 98.03),
        ("Gemini 2.5 Flash", "strict_pct", 89.66), ("Gemini 2.5 Flash", "judge_augmented_pct", 96.55),
        ("LLaMA-3.3-70B", "judge_augmented_pct", 98.03),
    ]
    for model, field, expected in checks:
        actual = pm.get(model, {}).get(field)
        check(f"regime: {model} {field} == {expected}", actual == expected,
              f"found {actual}")
    check("regime: strict spread == 14.5 pp (see F11 -- NOT 14.6, do not 'fix' this)",
          regime.get("spread", {}).get("strict_pp") == 14.54, f"found {regime.get('spread')}")
    corr = regime.get("correlations", {}).get("strict_vs_judge_only", {})
    # 2026-09-03: rho updated from -0.2238 to -0.2732 after the judge-input-truncation fix
    # (Plan v3 Phase 2.6 / cleanup item 7) changed 52/449 verdicts on 5 of 12 models' judged
    # rows. Paper now reports this as -0.273. Do not revert to -0.224/-0.2238.
    check("regime: strict vs judge-only Spearman rho == -0.2732 (paper reports as -0.273)",
          corr.get("spearman_rho") == -0.2732, f"found {corr}")
    check("regime: judge-only spread == 7.64 pp (paper reports as 7.6)",
          regime.get("spread", {}).get("judge_only_pp") == 7.64, f"found {regime.get('spread')}")
else:
    check("evaluation/regime_three_way.json exists", False,
          "run scripts/regime_table.py first")

print("\n=== IAA CONSISTENCY (cleanup item 8: rounds 2-3 scored for real) ===")
iaa_path = ROOT / "evaluation" / "iaa_summary.json"
if iaa_path.exists():
    iaa = json.loads(iaa_path.read_text(encoding="utf-8"))
    pooled = iaa.get("pooled_180", {})
    check("iaa_summary.json: pooled n == 180", pooled.get("n") == 180, f"found {pooled.get('n')}")
    check("iaa: pooled overall agreement == 86.1% (155/180)",
          round(pooled.get("overall_agreement", 0) * 100, 1) == 86.1,
          f"found {pooled.get('overall_agreement')}")
    check("iaa: pooled CON kappa == 0.712 (regression guard)",
          round(pooled.get("con_kappa", 0), 3) == 0.712, f"found {pooled.get('con_kappa')}")
    r1 = iaa.get("round1", {})
    check("iaa: round 1 unchanged at 85.0% (frozen prior work, never overwritten)",
          round(r1.get("overall_agreement", 0) * 100, 1) == 85.0, f"found {r1.get('overall_agreement')}")
else:
    check("evaluation/iaa_summary.json exists", False,
          "run scripts/score_iaa_rounds.py first")

print("\n=== GEMINI-VS-PHI4 AGREEMENT CONSISTENCY ===")
gvp_path = ROOT / "evaluation" / "gemini_vs_phi4_agreement.json"
if gvp_path.exists():
    gvp = json.loads(gvp_path.read_text(encoding="utf-8"))
    check("gemini_vs_phi4_agreement.json: total == 874", gvp.get("total") == 874, f"found {gvp.get('total')}")
    check("gemini_vs_phi4_agreement.json: agree == 698 (regression guard -- was 703 pre-truncation-fix)",
          gvp.get("agree") == 698, f"found {gvp.get('agree')}")
else:
    check("evaluation/gemini_vs_phi4_agreement.json exists", False,
          "run scripts/analyze_phi4_judge.py first")

print("\n=== MANUSCRIPT ===")

STALE_STRINGS = [
    "69.0", "n = 100", "42 pairs", "0.041", "0.455", "0.364", "0.413",
    "0.790", "0.910", "0.861", "0.057", "19.3-point", "26.3-point", "29.5-point",
    "DeepSeek R1 70B",  # must be DeepSeek-R1-Distill-Llama-70B / DeepSeek-R1-Distill
    "70.4--89.7", "79.8--96.6",  # pre-judge abstract accuracy ranges
    "corrected accuracy", "corrected scoring", "corrected regime",
    "11th of twelve", "11th to 1st", "strict-11th",
    "remaining eleven span",  # the removed "excluding Gemma" compression framing
    "86\\% reclassified", "86.1\\% reclassified",  # pre-phi4 DeepSeek reclassification figure
    "format non-compliance rather than",  # overclaims the judge verdict as fact, not "reclassified by the judge"
    "true accuracy lies between",  # neither regime is ground truth; there is no "true accuracy" to bound
    "near-chance",  # no defined chance baseline for open-ended numeric answers
    "genuine failures",  # judge-confirmed residual is not "genuine" (judge isn't ground truth either)
    "strict false-negative rate", "strict false-positive rate", "Strict FN", "Strict FP",
    # ^ presuppose phi4-mini is ground truth; use "overturn rate" / "judge rejection rate" instead
    "93.6--98.0", "0.113", "4.4 under judge-audited", "14.6-point",
    # ^ pre-llama3/mistral-truncation-fix headline stats, superseded by 94.1-98.0 / 0.067 / 3.9 /
    # 14.5-point (14.6-point was also independently wrong: 89.66-75.12=14.54, rounds to 14.5)
    "Ten of the twelve models move",  # superseded by "Eleven" once llama3's rank crossed the >=2 threshold
    "returns an independent verdict",  # phi4-mini is explicitly not independent/ground-truth (Section 6.5)
    "judge-confirmed",  # judge is not ground truth; use "judge-retained"
    "judge audit of every flagged error",  # the real methodology judges every REG/NUM/TMP prediction, not only flagged errors
    "is not a capability gap",  # overclaims beyond what judge-audited scoring (itself imperfect) establishes
    "denies models the option of recalling",  # unestablished training-data-contamination claim
    "only the weakest is significantly worse",  # superseded once Bonferroni was applied to the full 12-test family
    "12 items where both verdicts happened to match",  # the messy pre-freeze adjudication bookkeeping ChatGPT flagged
    "47.7\\%",  # the pre-freeze phi4-mini disagreement-match rate; frozen analysis uses 43.1%
    # --- Plan v3 (2026-09-02) retirements. These are EXPECTED to fail until Phase 1-4 land;
    # that is the point -- this blocklist is what makes "Phase 1 done" checkable rather than
    # a matter of opinion. See paper/tmlr/../../../../.claude/plans/... Plan v3 Section 1 (F1-F10).
    # "judge-audited accuracy" (narrow substring) checked here originally; replaced by a regex in
    # STALE_PATTERNS below because that narrow form silently missed "judge-audited scoring/regime/
    # scores" -- 5 real instances survived a first pass and were only caught by a manual grep.
    "identical prompting and decoding",  # F3: false -- budgets are 200/300/512/1024/2048/unset
    "completion budget shared by every model",  # F3: same false claim, second location
    "exact identifier used for",  # F4: appendix table has no checkpoint strings for 9 of 12 models
    "80\\% threshold",  # F10: uncited "commonly used" benchmark-quality claim
    "share no threshold",  # F9: false -- p>=0.9 and p>0.90 are the same set on a 12-model panel
    # --- 2026-09-02 source/PDF consistency cleanup (post-Phase-8, pre-Phase-2).
    "mildly opposite",  # unsupported at p=0.48 (not significant) -- "show no positive rank correspondence"
    "verbose output style",  # F2 doesn't survive adjustment -- no verbosity mechanism is established
    "Verbose-output scoring artifact",  # same claim, table-footnote location
    "verbosity artifact",  # same claim, IAA-agreement location
    "Verbose answers",  # appendix "Root Causes" bullet asserting verbosity as a named mechanism
    "primary LLM-as-judge audit",  # mislabels the 874-item Gemini pilot as "the primary" audit
    "effective-size analysis",  # retired term; use "discriminative-coverage analysis"
    "discriminative size",  # malformed hybrid of the two competing terms
    # --- IAA rounds 2-3 scored for real (scripts/score_iaa_rounds.py, cleanup item 8).
    # These were premature/assumed numbers written before rounds 2-3 existed on disk.
    "0.645",  # superseded pooled-180 CON kappa; real value is 0.712 (evaluation/iaa_summary.json)
    "59.1\\%",  # superseded NUM IAA agreement (round-1-only estimate); real pooled value is 81.8%
    # --- 2026-09-03: judge-input-truncation fix (cleanup item 7) shifted 52/449 verdicts on
    # 5 of 12 models, which moved the regime correlation numbers. Old values retired below;
    # real ones are regenerated by scripts/regime_table.py and pinned in the check above.
    "-0.224",  # superseded strict-vs-judge-only Spearman rho; real value is -0.273
    "-0.061",  # superseded Kendall tau; real value is -0.168
    "p = 0.48",  # superseded p-value paired with the old rho; real value is p=0.39
    "95.32",  # superseded judge-only max (was Llama 4 Scout 17B); real max is 94.83 (Gemini 2.5 Pro)
    "8.1 points",  # superseded judge-only spread; real value is 7.6 points
    "ranks 9th of twelve",  # superseded DeepSeek judge-only rank; real rank is 7th, tied w/ Qwen3-32B
    "falls by nine places",  # superseded Gemini 2.5 Flash strict-to-judge-only rank fall; real fall is 8 places
    # --- 2026-09-03: author-requested wording fixes (round 2 cleanup)
    "independent quality-checker",  # overclaims independence for a model that's also evaluated; "secondary model-based quality check"
    "Neither regime is wrong",  # implies a binary right/wrong frame; "neither is ground truth... universally preferable"
    "reducing the likelihood that performance is explained by benchmark-specific familiarity",
    # ^ implied the domain choice addresses contamination; softened to "lowers but does not rule out"
    "less likely to be dominated by memorised recall",  # old unhedged claim; see "reasonable candidate" framing now
    # --- 2026-09-03: caught during a page-by-page visual re-check (item 9 continued), not source
    # grep -- the judge-truncation-fix regeneration (52 verdict flips, earlier this pass) updated
    # evaluation/gemini_vs_phi4_agreement.json (703->698, 80.4%->79.9%) but one manuscript sentence
    # was never swept for the new value.
    "703 (\\textbf{80.4\\% raw agreement})",
    "80.4\\% raw agreement",
    # NOTE: "77.2%" (superseded pooled-180 overall IAA agreement) is deliberately NOT
    # blocklisted as a bare string -- Qwen3-32B's real, correct NUM accuracy is coincidentally
    # 77.2% and appears legitimately elsewhere (draft_05 main table, appendix few-shot table).
    # "0.645" above is the reliable regression guard for this retirement.
]
# "semantic scoring" was blocklisted outright, but it has a legitimate descriptive use
# (contrasting it with strict scoring, e.g. in a novelty-framing sentence) -- the actual
# hazard is describing OUR judge-audited pipeline AS semantic/ground-truth, which
# terminology consistency handles elsewhere, not a bare-phrase ban. Removed as
# overly blunt (caught its own author's legitimate sentence on first tightening).
STALE_PATTERNS = [
    # The false claim was "items ... 192" with nothing correcting it in between.
    # "34 of 192" / "34 of these 192" is the correct, fixed phrasing -- exclude it
    # via a negative lookahead rather than trying to match the bad case directly.
    (r"\bitems?\s+.{0,15}?(?:drawn from|span(?:ning)?)\s+(?!34\b)192\b",
     "items drawn from/span 192 without the '34 of' correction (false -- should be 34)"),
    # F1: "judge-audited" is retired -- ambiguous between judge-only (verdict final, both
    # directions) and judge-augmented (strict OR judge, a one-directional composite the paper
    # never previously disclosed as such). The one legitimate remaining use is the terminology
    # rule itself stating the word is avoided, always rendered as the quoted meta-reference
    # ``judge-audited'' (closing quote immediately after, no intervening noun) -- exclude only
    # that exact form. A narrow substring check ("judge-audited accuracy") missed "judge-audited
    # scoring/regime/scores" and let 5 real instances survive a first pass; catch the general form.
    (r"judge-audited(?!'')", "'judge-audited' used outside its own quoted terminology-rule mention "
     "-- use 'judge-only' or 'judge-augmented' explicitly"),
]

main_tex = SUB_DIR / "main.tex"
assembled_text = ""
if main_tex.exists():
    seen = set()
    stack = [main_tex]
    while stack:
        f = stack.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        content = f.read_text(encoding="utf-8", errors="replace")
        assembled_text += content
        for m in re.finditer(r"\\input\{([^}]+)\}", content):
            rel = m.group(1)
            stack.append((f.parent / rel).resolve())

    for s in STALE_STRINGS:
        hit = s in assembled_text
        check(f"no stale string: {s!r}", not hit)
    for pat, desc in STALE_PATTERNS:
        hit = re.search(pat, assembled_text)
        check(f"no stale pattern: {desc}", not hit, hit.group(0) if hit else "")

    # Broad marker match: any [[...]] bracket pair, not just an all-caps single
    # token. The narrow \[\[([A-Z_]+)\]\] version missed the real, prose-style
    # "[[PENDING: human adjudication ...]]" marker and only caught an unrelated
    # stale header comment -- exactly the kind of false-green result this script
    # must not produce.
    pending_markers = re.findall(r"\[\[.*?\]\]", assembled_text, flags=re.DOTALL)
    if pending_markers:
        previews = [m[:60].replace("\n", " ") + ("..." if len(m) > 60 else "") for m in pending_markers]
        check("zero [[PENDING]] markers remain", False,
              f"{len(pending_markers)} found: {previews}")
    else:
        check("zero [[PENDING]] markers remain", True)

    check("LLM-use footnote present", "Use of AI assistance" in assembled_text)
    check("human-subject statement present", "Institutional review status" in assembled_text)
    check("no 'public domain' copyright overclaim", "no copyright restrictions" not in assembled_text)
else:
    check("main.tex exists", False)

print("\n=== TMLR COMPLIANCE ===")

check("official tmlr.sty present", (SUB_DIR / "tmlr.sty").exists())
check("official tmlr.bst present", (SUB_DIR / "tmlr.bst").exists())
if main_tex.exists():
    check("no [accepted] or [preprint] option (anonymous)",
          "accepted" not in re.search(r"usepackage(\[[^\]]*\])?\{tmlr\}", assembled_text).group(0)
          if re.search(r"usepackage(\[[^\]]*\])?\{tmlr\}", assembled_text) else False)
    check("no \\author{} block (anonymous)", r"\author{" not in assembled_text)

print("\n=== COMPILATION ===")

if main_tex.exists():
    try:
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-f", "main.tex"],
            cwd=SUB_DIR, capture_output=True, text=True, timeout=180)
        log_path = SUB_DIR / "main.log"
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        pdf_path = SUB_DIR / "main.pdf"

        check("main.pdf produced", pdf_path.exists())
        latex_errors = re.findall(r"^! .+$", log_text, flags=re.MULTILINE)
        check("zero LaTeX errors in compile log", len(latex_errors) == 0,
              f"{len(latex_errors)} found: {latex_errors[:5]}")
        undefined_refs = re.findall(r"Reference `[^']+' on page \d+ undefined", log_text)
        check("zero undefined references", len(undefined_refs) == 0,
              f"{len(undefined_refs)} found: {undefined_refs[:5]}")
        undefined_cites = re.findall(r"Citation `[^']+' on page \d+ undefined", log_text)
        check("zero undefined citations", len(undefined_cites) == 0,
              f"{len(undefined_cites)} found: {undefined_cites[:5]}")

        page_match = re.search(r"Output written on main\.pdf \((\d+) pages?", log_text)
        total_pages = int(page_match.group(1)) if page_match else None

        # Main-content page count, measured up to \label{sec:mainend} placed
        # immediately before \bibliography in main.tex -- TMLR has no hard page
        # limit, but the 12pp fast-review target is a real strategic goal, so
        # this is a soft target (warn), not a hard fail, unless wildly exceeded.
        aux_path = SUB_DIR / "main.aux"
        main_content_pages = None
        if aux_path.exists():
            aux_text = aux_path.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"\\newlabel\{sec:mainend\}\{\{[^}]*\}\{(\d+)\}", aux_text)
            if m:
                main_content_pages = int(m.group(1))
        if main_content_pages is not None:
            print(f"  [INFO] main content runs to page {main_content_pages} (12pp is the fast-review target, not a hard limit)")
            if main_content_pages > 16:
                warn("main content well over the 12pp fast-review target", f"{main_content_pages} pages")
        else:
            warn("could not measure main-content page count", "add \\label{sec:mainend} before \\bibliography in main.tex")
        if total_pages is not None:
            print(f"  [INFO] total pages (incl. appendix/references): {total_pages}")
    except FileNotFoundError:
        check("latexmk available", False, "install MiKTeX/TeX Live or compile manually before submitting")
    except subprocess.TimeoutExpired:
        check("compile completes within 180s", False)
else:
    check("main.tex exists for compilation", False)

print("\n" + "=" * 60)
if failures:
    print(f"RESULT: FAIL -- {len(failures)} check(s) failed:")
    for f in failures:
        print(f"  - {f}")
    print("\nDo not submit.")
    sys.exit(1)
else:
    print("RESULT: PASS")
    if warnings:
        print(f"({len(warnings)} warning(s) -- review before submitting)")
    sys.exit(0)
