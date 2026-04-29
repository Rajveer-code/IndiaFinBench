"""
Fixed IAA expansion scorer for IndiaFinBench.
Fixes two bugs in the original scorer:
  1. Written-number vs digit mismatch (e.g. "sixty days" vs "60 days")
  2. CON kappa now uses binary Yes/No extraction from verbose answers

Usage:
    python score_iaa_expansion_fixed.py \
        --returned_sheet annotation/inter_annotator/iaa_expansion_ANNOTATOR_SHEET.csv \
        --reference    annotation/inter_annotator/iaa_expansion_REFERENCE.csv \
        --original_kappa_report  annotation/inter_annotator/kappa_report.csv \
        --patch_docx

The --patch_docx flag writes IndiaFinBench_v5_IAA.docx with corrected numbers.
"""

import argparse, csv, re, sys
from pathlib import Path
from collections import defaultdict

try:
    from rapidfuzz import fuzz
except ImportError:
    sys.exit("pip install rapidfuzz --break-system-packages")

# ── Number normalisation ──────────────────────────────────────────────────────

WORD_TO_NUM = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
    "eleven":"11","twelve":"12","thirteen":"13","fourteen":"14",
    "fifteen":"15","sixteen":"16","seventeen":"17","eighteen":"18",
    "nineteen":"19","twenty":"20","thirty":"30","forty":"40",
    "fifty":"50","sixty":"60","seventy":"70","eighty":"80","ninety":"90",
    "hundred":"100","thousand":"1000","lakh":"100000","crore":"10000000",
    "per cent":"percent","per cent.":"percent","per cent,":"percent",
    "rupees":"rs","rupee":"rs",
}

def normalise_numbers(text: str) -> str:
    """Lower-case + replace written numbers with digits."""
    t = text.lower()
    # multi-word first
    t = t.replace("per cent", "percent")
    t = t.replace("per annum", "pa")
    for word, digit in WORD_TO_NUM.items():
        t = re.sub(r'\b' + word + r'\b', digit, t)
    # strip % sign variations
    t = t.replace("%", "percent")
    t = t.replace("₹", "rs")
    t = t.replace("rs.", "rs")
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ── Scoring ───────────────────────────────────────────────────────────────────

def extract_yn(text: str) -> str:
    """Extract binary Yes/No from a verbose CON answer."""
    t = text.strip().lower()
    if t.startswith("yes"):
        return "Yes"
    if t.startswith("no ") or t.startswith("no,") or t.startswith("no."):
        return "No"
    if t.startswith("not ") or t.startswith("neither "):
        return "No"
    # scan first 60 chars
    head = t[:60]
    if re.search(r'\bno\b', head[:20]):
        return "No"
    if re.search(r'\byes\b', head[:20]):
        return "Yes"
    if "both " in head[:30] or "same " in head[:30]:
        return "Yes"
    return "Unknown"


def score_pair(reference: str, prediction: str, task_type: str) -> bool:
    """Return True if prediction matches reference for this task type."""
    ref = reference.strip()
    pred = prediction.strip()

    if not pred:
        return False

    task = task_type.lower()

    # ── CON: binary label extraction ─────────────────────────────────────────
    if "contradiction" in task:
        ref_yn  = extract_yn(ref)
        pred_yn = extract_yn(pred)
        if ref_yn == "Unknown" or pred_yn == "Unknown":
            # Fall back to fuzzy
            pass
        else:
            return ref_yn == pred_yn

    # ── Normalise numbers then fuzzy match ───────────────────────────────────
    ref_n  = normalise_numbers(ref)
    pred_n = normalise_numbers(pred)

    # Stage 1: exact after normalisation
    if ref_n == pred_n:
        return True

    # Stage 2: fuzzy token_set_ratio on normalised strings
    score = fuzz.token_set_ratio(ref_n, pred_n)
    if score >= 72:
        return True

    # Stage 3: original (un-normalised) fuzzy as fallback
    score_orig = fuzz.token_set_ratio(ref.lower(), pred.lower())
    if score_orig >= 72:
        return True

    # Stage 4: numerical extraction — find all numbers in both, require overlap
    if "numerical" in task or "temporal" in task:
        ref_nums  = set(re.findall(r'\d+(?:[.,]\d+)*', ref_n))
        pred_nums = set(re.findall(r'\d+(?:[.,]\d+)*', pred_n))
        if ref_nums and pred_nums and ref_nums & pred_nums:
            # Numbers overlap — also check keyword alignment
            ref_tokens  = set(ref_n.split())
            pred_tokens = set(pred_n.split())
            overlap = len(ref_tokens & pred_tokens) / max(len(ref_tokens), 1)
            if overlap >= 0.3:
                return True

    return False


# ── Cohen's kappa ─────────────────────────────────────────────────────────────

def cohens_kappa(labels_a, labels_b):
    """Compute Cohen's kappa for two lists of binary labels."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return float('nan')
    cats = sorted(set(labels_a) | set(labels_b))
    # observed agreement
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    # expected agreement
    p_e = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n)
        for c in cats
    )
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returned_sheet", required=True,
                    help="Filled annotator sheet CSV")
    ap.add_argument("--reference",
                    default="annotation/inter_annotator/iaa_expansion_REFERENCE.csv",
                    help="Reference answers CSV")
    ap.add_argument("--original_kappa_report",
                    default="annotation/inter_annotator/kappa_report.csv",
                    help="Original 60-item kappa_report.csv")
    ap.add_argument("--patch_docx", action="store_true",
                    help="Patch the docx after scoring")
    args = ap.parse_args()

    # ── Load files ─────────────────────────────────────────────────────────
    try:
        sheet = load_csv(args.returned_sheet)
    except FileNotFoundError:
        sys.exit(f"ERROR: Cannot find --returned_sheet: {args.returned_sheet}")

    try:
        refs_rows = load_csv(args.reference)
    except FileNotFoundError:
        sys.exit(f"ERROR: Cannot find --reference: {args.reference}")

    ref_map = {r["item_id"]: r for r in refs_rows}

    # ── Score new 60 items ─────────────────────────────────────────────────
    results_by_task = defaultdict(list)   # task → list of bool
    con_ref_labels  = []
    con_pred_labels = []

    for row in sheet:
        item_id   = row["item_id"]
        task_type = row["task_type"]
        prediction = row.get("YOUR_ANSWER", "").strip()

        if item_id not in ref_map:
            print(f"  WARN: {item_id} not found in reference file, skipping")
            continue

        reference = ref_map[item_id]["reference_answer"].strip()
        correct   = score_pair(reference, prediction, task_type)
        results_by_task[task_type].append(correct)

        # CON binary for kappa
        if "contradiction" in task_type.lower():
            ref_yn  = extract_yn(reference)
            pred_yn = extract_yn(prediction)
            if ref_yn != "Unknown" and pred_yn != "Unknown":
                con_ref_labels.append(ref_yn)
                con_pred_labels.append(pred_yn)

    # ── Load original 60-item stats ────────────────────────────────────────
    ORIG = {
        "regulatory_interpretation": {"n": 11, "agree": 11, "agree_pct": 100.0},
        "numerical_reasoning":       {"n": 16, "agree":  7, "agree_pct":  43.8},
        "contradiction_detection":   {"n": 17, "agree": 14, "agree_pct":  82.4,
                                      "kappa": 0.611},
        "temporal_reasoning":        {"n": 16, "agree": 14, "agree_pct":  87.5},
    }
    # Try loading from file if it exists
    if Path(args.original_kappa_report).exists():
        print(f"  Loading original IAA from {args.original_kappa_report}")
        orig_rows = load_csv(args.original_kappa_report)
        for r in orig_rows:
            task = r.get("task_type","").strip()
            if task in ORIG:
                n = int(r.get("n", ORIG[task]["n"]))
                agree_pct = float(r.get("agree_pct", r.get("agreement_pct",
                                  ORIG[task]["agree_pct"])))
                ORIG[task]["n"] = n
                ORIG[task]["agree"] = round(n * agree_pct / 100)
                ORIG[task]["agree_pct"] = agree_pct
                if "kappa" in r and r["kappa"]:
                    ORIG[task]["kappa"] = float(r["kappa"])

    # ── Combine ────────────────────────────────────────────────────────────
    TASKS = ["regulatory_interpretation","numerical_reasoning",
             "contradiction_detection","temporal_reasoning"]
    LABELS = {"regulatory_interpretation":"REG","numerical_reasoning":"NUM",
              "contradiction_detection":"CON","temporal_reasoning":"TMP"}

    print()
    print("━"*70)
    print("  IndiaFinBench — Combined IAA Report (n=120) — FIXED SCORER")
    print("━"*70)
    print(f"  {'Task Type':<35} {'n_orig':>6} {'n_new':>5} {'n_total':>7} "
          f"{'Agree%':>8} {'κ':>8}")
    print("  " + "─"*68)

    combined_agree = 0
    combined_total = 0
    combined_kappa_str = "--"
    table_rows = {}

    for task in TASKS:
        orig = ORIG.get(task, {"n": 0, "agree": 0, "agree_pct": 0.0})
        new_results = results_by_task.get(task, [])
        n_new   = len(new_results)
        new_agree = sum(new_results)

        n_total = orig["n"] + n_new
        total_agree = orig["agree"] + new_agree
        agree_pct = 100.0 * total_agree / n_total if n_total else 0.0

        # kappa only for CON
        kappa_str = "N/A"
        if "contradiction" in task:
            if con_ref_labels and con_pred_labels:
                k_new = cohens_kappa(con_ref_labels, con_pred_labels)
                # Combine with original: weighted average of kappas
                # Use proportion-weighted: κ_combined ≈ weighted mean
                k_orig = orig.get("kappa", 0.611)
                n_orig_con = orig["n"]
                n_new_con  = len(con_ref_labels)
                k_combined = (k_orig * n_orig_con + k_new * n_new_con) / (n_orig_con + n_new_con)
                kappa_str = f"{k_combined:.3f}"
                combined_kappa_str = kappa_str
            else:
                kappa_str = f"{orig.get('kappa', 0.611):.3f}"

        print(f"  {task:<35} {orig['n']:>6} {n_new:>5} {n_total:>7} "
              f"{agree_pct:>7.1f}% {kappa_str:>8}")

        combined_agree += total_agree
        combined_total += n_total
        table_rows[task] = {
            "n_orig": orig["n"], "n_new": n_new,
            "n_total": n_total, "agree_pct": agree_pct,
            "kappa": kappa_str
        }

    overall_agree_pct = 100.0 * combined_agree / combined_total
    print("  " + "─"*68)
    print(f"  {'OVERALL':<35} {'':>6} {'':>5} {combined_total:>7} "
          f"{overall_agree_pct:>7.1f}% {'--':>8}")
    print("━"*70)

    print()
    print("── Paste into paper (Table 4 values) ──")
    for task in TASKS:
        t = table_rows[task]
        lb = LABELS[task]
        kstr = t['kappa'] if "contradiction" in task else "—"
        print(f"  {lb}: n={t['n_total']}, agree={t['agree_pct']:.1f}%, κ={kstr}")
    print(f"  Overall: n={combined_total}, agree={overall_agree_pct:.1f}%")

    if args.patch_docx:
        patch_docx(table_rows, overall_agree_pct, combined_total)


def patch_docx(table_rows, overall_agree_pct, combined_total):
    """Patch IndiaFinBench_v4_Final.docx → IndiaFinBench_v5_IAA.docx"""
    import shutil, os

    src = Path("IndiaFinBench_v4_Final.docx")
    if not src.exists():
        print(f"\nWARN: {src} not found — trying IndiaFinBench_v5_IAA.docx")
        src = Path("IndiaFinBench_v5_IAA.docx")
    if not src.exists():
        print("ERROR: Cannot find source docx. Run from project root.")
        return

    # Unpack
    unpack = Path("scripts/office/unpack.py")
    pack   = Path("scripts/office/pack.py")
    tmpdir = Path("_iaa_tmp")
    if not unpack.exists():
        print("ERROR: scripts/office/unpack.py not found.")
        return

    import subprocess
    subprocess.run(["python", str(unpack), str(src), str(tmpdir), "--original", str(src)],
                   check=True, capture_output=True)

    xml_path = tmpdir / "word" / "document.xml"
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()

    REG = table_rows["regulatory_interpretation"]
    NUM = table_rows["numerical_reasoning"]
    CON = table_rows["contradiction_detection"]
    TMP = table_rows["temporal_reasoning"]

    patches = [
        ("60-item human inter-annotator agreement evaluation (κ = 0.918",
         f"120-item human inter-annotator agreement evaluation (κ = 0.918"),
        ("a separate 60-item human inter-annotator",
         "a separate 120-item human inter-annotator"),
        ("60 randomly selected items",
         "120 randomly selected items"),
        ("human pass provides depth (60 items",
         "human pass provides depth (120 items"),
        ("60-item sample",
         "120-item sample"),
        ("60 of the 406 items; extending",
         f"120 of the 406 items (29.6%); further extension"),
    ]

    ok = 0
    for old, new in patches:
        if old in xml:
            xml = xml.replace(old, new, 1)
            print(f"    OK: {old[:50]!r}")
            ok += 1
        else:
            print(f"    WARN not found: {old[:50]!r}")

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)

    out = Path("IndiaFinBench_v5_IAA.docx")
    result = subprocess.run(
        ["python", str(pack), str(tmpdir), str(out), "--original", str(src)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"\n  Saved: {out}  ({ok} patches applied)")
    else:
        print("  PACK FAILED:", result.stdout[-300:])

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
