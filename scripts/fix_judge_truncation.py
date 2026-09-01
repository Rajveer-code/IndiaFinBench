"""Fix the judge-truncation bug for the 4 affected models (Gemini 2.5 Flash,
LLaMA-3.3-70B, LLaMA-3-8B, Mistral-7B).

Two independent repairs:

PHASE A (free, no model calls): question/ref_answer in evaluation/results/*.csv
are restored to full text from the dataset JSON, for all 12 models. This alone
fixes every item where only ref_answer was truncated (prediction was already
complete) -- rejudge those with the corrected reference, no requery needed.

PHASE B (requires live model calls): for items where `prediction` itself was
cut at 200 chars, requery the same model/prompt/settings (reused directly from
scripts/evaluate.py, not reimplemented) to get a fresh full response, then
rescore with the exact same score_answer(). If the fresh response scores
IDENTICALLY to the original, the row is updated (full prediction + rejudge).
If it scores DIFFERENTLY, the row is left untouched (truncated prediction,
original correct value -- internally consistent, just incomplete) and logged
separately, rather than silently pairing a new prediction with an old score
that no longer matches it.

PHASE C: rejudge every updated row with phi4-mini (Ollama, free), using the
exact same rubric as the original full-coverage pass.

Usage: python scripts/fix_judge_truncation.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import truststore
truststore.inject_into_ssl()  # Avast HTTPS-scanning injects its own root cert into the
# Windows store but not OpenSSL's bundled trust store; this makes Python verify via the
# OS-native store (which already trusts it) instead of failing with CERTIFICATE_VERIFY_FAILED.

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from evaluate import (MODELS, build_prompt, call_gemini, call_groq, call_ollama,
                       init_client, score_answer)  # noqa: E402


def client_actually_works(provider, model_id, client):
    """A configured client can still hold a dead/revoked key. Confirm with one real
    call before committing to a full requery loop -- failing fast beats burning
    through 100+ calls that were always going to fail identically."""
    try:
        if provider == "gemini_pool":
            resp = call_gemini(model_id, "Say OK.")
        elif provider == "groq":
            resp = call_groq(model_id, "Say OK.", client)
        else:
            return True
        return not resp.startswith("FAIL:")
    except Exception:
        return False

RESULTS_DIR = ROOT.parent / "evaluation" / "results"
JUDGED_PHI4_DIR = ROOT.parent / "evaluation" / "results_judged_phi4"
DATASET_PATH = ROOT.parent / "annotation" / "raw_qa" / "indiafinbench_qa_combined_406.json"

AFFECTED = ["gemini", "groq70b", "llama3", "mistral"]  # keys into MODELS / filenames

PHI4_PROMPT = """You are an expert evaluator for an Indian financial regulatory reasoning benchmark.

Question: {question}

Reference Answer: {reference}

Model Prediction: {prediction}

Determine if the model prediction is CORRECT.

Rules:
- CORRECT if the final value or fact matches, even if:
  * Different formatting (currency symbols, comma separators, units)
  * Extra calculation steps shown before the final answer
  * Paraphrasing of the same regulatory rule
  * Rounding within 1%
- INCORRECT only if: wrong number, wrong threshold, wrong date, wrong entity, contradicts reference

Reply with exactly:
CORRECT
Reason: [one sentence]
"""


def load_dataset():
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items", data.get("data", list(data.values())[0]))
    return {item["id"]: item for item in data}


def call_phi4_judge(question, reference, prediction):
    import urllib.request
    prompt = PHI4_PROMPT.format(question=question[:600], reference=reference[:300],
                                 prediction=prediction[:500])
    body = json.dumps({"model": "phi4-mini", "prompt": prompt, "stream": False,
                        "options": {"temperature": 0.0}}).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = json.loads(r.read().decode("utf-8"))["response"]
    verdict = 1 if text.strip().upper().startswith("CORRECT") else 0
    reason = text.split("Reason:", 1)[-1].strip() if "Reason:" in text else text.strip()[:200]
    return verdict, reason


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames), list(r)


def save_csv(path, fieldnames, rows):
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def phase_a(dataset):
    print("=== PHASE A: restore question/ref_answer to full text (all 12 models) ===")
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        key = path.stem.replace("_results", "")
        fieldnames, rows = load_csv(path)
        n_q = n_r = 0
        for r in rows:
            item = dataset.get(r["id"])
            if not item:
                continue
            if r["question"] != item["question"]:
                r["question"] = item["question"]
                n_q += 1
            if r["ref_answer"] != item["answer"]:
                r["ref_answer"] = item["answer"]
                n_r += 1
        if n_q or n_r:
            save_csv(path, fieldnames, rows)
            print(f"  {path.name}: question fixed={n_q}  ref_answer fixed={n_r}")


def phase_b_and_c(dataset):
    print("\n=== PHASE B+C: requery truncated predictions, rejudge affected rows ===")
    summary = {}
    for key in AFFECTED:
        cfg = MODELS[key]
        provider = cfg["provider"]
        client = init_client(provider)
        if client is None:
            print(f"  {cfg['label']}: provider unavailable, skipping requery for this model")
            summary[key] = {"skipped_no_client": True}
            continue
        if not client_actually_works(provider, cfg["model_id"], client):
            print(f"  {cfg['label']}: client configured but the API key is rejected/dead -- "
                  f"skipping requery for this model, needs a fresh key")
            summary[key] = {"skipped_dead_key": True}
            continue

        results_path = RESULTS_DIR / f"{key}_results.csv"
        fieldnames, rows = load_csv(results_path)
        by_id = {r["id"]: r for r in rows}

        judged_path = JUDGED_PHI4_DIR / f"{key}_results.csv"
        jfieldnames, jrows = load_csv(judged_path)
        jby_id = {r["id"]: r for r in jrows}

        n_requeried = n_matched = n_flipped = n_ref_only_rejudged = n_infra_failed = 0
        results_changed = judged_changed = False

        for iid, r in by_id.items():
            item = dataset.get(iid)
            if not item or item["task_type"] == "contradiction_detection":
                continue
            jr = jby_id.get(iid)
            if jr is None:
                continue

            pred_was_truncated = len(r["prediction"]) == 200
            if pred_was_truncated:
                prompt = build_prompt(item)
                try:
                    if provider == "gemini_pool":
                        fresh = call_gemini(cfg["model_id"], prompt)
                    elif provider == "groq":
                        fresh = call_groq(cfg["model_id"], prompt, client)
                    elif provider == "ollama":
                        fresh = call_ollama(cfg["model_id"], prompt)
                    else:
                        fresh = None
                except Exception as e:
                    fresh = f"FAIL: {str(e)[:100]}"
                n_requeried += 1

                if fresh is not None and not fresh.startswith("FAIL:"):
                    new_correct = score_answer(item["answer"], fresh, item["task_type"])
                    old_correct = int(r["correct"])
                    if new_correct == old_correct:
                        r["prediction"] = fresh
                        results_changed = True
                        n_matched += 1
                        v, reason = call_phi4_judge(item["question"], item["answer"], fresh)
                        jr["question"] = item["question"]
                        jr["ref_answer"] = item["answer"]
                        jr["prediction"] = fresh
                        jr["judge_verdict"] = str(v)
                        jr["judge_reason"] = reason
                        judged_changed = True
                    else:
                        n_flipped += 1
                        print(f"    FLIP (left untouched): {key} {iid} old_correct={old_correct} "
                              f"new_correct={new_correct}")
                else:
                    n_infra_failed += 1
                    print(f"    INFRA FAIL (left untouched): {key} {iid}: {fresh}")
            else:
                # ref_answer may have changed in Phase A; rejudge with corrected ref, same pred.
                if jr["ref_answer"] != item["answer"] or jr["question"] != item["question"]:
                    v, reason = call_phi4_judge(item["question"], item["answer"], r["prediction"])
                    jr["question"] = item["question"]
                    jr["ref_answer"] = item["answer"]
                    jr["judge_verdict"] = str(v)
                    jr["judge_reason"] = reason
                    judged_changed = True
                    n_ref_only_rejudged += 1

        if results_changed:
            save_csv(results_path, fieldnames, rows)
        if judged_changed:
            save_csv(judged_path, jfieldnames, jrows)

        summary[key] = {"requeried": n_requeried, "matched_and_updated": n_matched,
                         "flipped_left_untouched": n_flipped, "infra_failed": n_infra_failed,
                         "ref_only_rejudged": n_ref_only_rejudged}
        print(f"  {cfg['label']}: requeried={n_requeried} matched={n_matched} "
              f"flipped(untouched)={n_flipped} infra_failed(untouched)={n_infra_failed} "
              f"ref-only-rejudged={n_ref_only_rejudged}")

    return summary


if __name__ == "__main__":
    dataset = load_dataset()
    phase_a(dataset)
    summary = phase_b_and_c(dataset)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
