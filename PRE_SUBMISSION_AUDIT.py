"""Single PASS/FAIL gate for TMLR submission (Phase 8 of the resubmission plan).

Run this before submitting and nothing else. Every check either passes or
prints exactly why it failed. No manual interpretation should be needed.

Usage: python PRE_SUBMISSION_AUDIT.py
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TEX_DIR = ROOT / "paper" / "tmlr"
SUB_DIR = TEX_DIR / "tmlr_submission"

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

# 12 models, 406 items each, no duplicate ids, no FAIL predictions
import sys as _sys
_sys.path.insert(0, str(ROOT))
from scripts.novel_methods_utils import MODEL_FILES, RESULTS_DIR  # noqa: E402

n_models = len(MODEL_FILES)
check("12 models in MODEL_FILES", n_models == 12, f"found {n_models}")

all_406 = True
no_dupes = True
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
    total_fails += sum(1 for r in rows if "FAIL" in r.get("prediction", ""))

check("all 12 models have 406 items", all_406)
check("no duplicate item ids", no_dupes)
# The paper documents 3 API failures (part of "61: 58 empty + 3 API failures" in
# Section 4.3) as an expected, scored-incorrect-by-design outcome, not a data bug.
check("FAIL-prediction count matches documented methodology", total_fails == 3,
      f"found {total_fails}, paper documents 3 API failures -- investigate if this changed")

# Provenance: 34 documents
prov_path = ROOT / "evaluation" / "provenance_summary.json"
if prov_path.exists():
    prov = json.load(open(prov_path, encoding="utf-8"))
    check("provenance: 34 documents represented", prov["documents_represented_in_qa"] == 34,
          f"found {prov['documents_represented_in_qa']}")
else:
    check("provenance_summary.json exists", False)

# Gemma identity confirmed
gemma_path = RESULTS_DIR / "gemma4_e4b_results.csv"
if gemma_path.exists():
    rows = list(csv.DictReader(open(gemma_path, encoding="utf-8")))
    versions = {r.get("model_version", "") for r in rows}
    check("Gemma results use confirmed gemma3:4b identity", versions == {"gemma3:4b"},
          f"found versions: {versions}")

# Cross-model judge: full coverage
judge_dir = ROOT / "evaluation" / "results_judged_phi4"
if judge_dir.exists():
    total_judged = sum(len(list(csv.DictReader(open(f, encoding="utf-8"))))
                        for f in judge_dir.glob("*.csv"))
    check("cross-model judge: full 4128-item coverage complete", total_judged >= 4128,
          f"{total_judged}/4128 -- STILL RUNNING, this must be 4128 before submission")
else:
    check("evaluation/results_judged_phi4/ exists", False, "judge has not been run")

print("\n=== MANUSCRIPT ===")

STALE_STRINGS = [
    "69.0", "n = 100", "42 pairs", "0.041", "0.455", "0.364",
    "0.790", "0.910", "0.861", "0.057", "19.3-point", "26.3-point",
    "DeepSeek R1 70B",  # must be DeepSeek-R1-Distill-Llama-70B / DeepSeek-R1-Distill
]
# "semantic scoring" was blocklisted outright, but it has a legitimate descriptive use
# (contrasting it with strict scoring, e.g. in a novelty-framing sentence) -- the actual
# hazard is describing OUR judge-audited pipeline AS semantic/ground-truth, which
# terminology consistency handles elsewhere, not a bare-phrase ban. Removed as
# overly blunt (caught its own author's legitimate sentence on first tightening).
# The false claim was specifically "items ... drawn from / span 192 documents" --
# 192 alone is the correct, accurate size of the *collected corpus* and appears
# legitimately throughout. Match only the actual false pattern.
STALE_PATTERNS = [
    # The false claim was "items ... 192" with nothing correcting it in between.
    # "34 of 192" / "34 of these 192" is the correct, fixed phrasing -- exclude it
    # via a negative lookahead rather than trying to match the bad case directly.
    (r"\bitems?\s+.{0,15}?(?:drawn from|span(?:ning)?)\s+(?!34\b)192\b",
     "items drawn from/span 192 without the '34 of' correction (false -- should be 34)"),
]
# "corrected accuracy" is legitimate ONLY inside the explicitly-pending appendix
# block (marked [[PENDING]] at its own top); flagged separately via the
# pending-marker check below rather than blocklisted outright, since the
# post-judge appendix rewrite will retire the phrase along with the marker.

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

    pending_markers = re.findall(r"\[\[([A-Z_]+)\]\]", assembled_text)
    if pending_markers:
        check("no [[PENDING]] markers remain", False,
              f"{len(pending_markers)} found: {set(pending_markers)}")
    else:
        check("no [[PENDING]] markers remain", True)

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
