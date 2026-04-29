"""
evaluate_fewshot_v3.py — Fixed version
=======================================
Uses Vertex AI (GCP credits) for Gemini, same auth as scorer_with_judge_gemini.py
Fixes the 0.0% bug: was using GOOGLE_API_KEY which wasn't set, now uses Vertex AI

SETUP (same as before):
    gcloud auth application-default login
    gcloud config set project finindiabench

USAGE:
    python scripts/evaluate_fewshot_v3.py \
        --dataset annotation/raw_qa/indiafinbench_qa_combined_406.json \
        --output evaluation/results_fewshot/

For Groq models (Qwen3, LLaMA):
    export GROQ_API_KEY=your_key
"""

import csv, json, os, re, time
from pathlib import Path
from collections import defaultdict

try:
    from rapidfuzz import fuzz as _rf
    HAVE_RF = True
except ImportError:
    HAVE_RF = False

# ── Few-shot examples (3 per task type) ──────────────────────────────────────
FEW_SHOT_EXAMPLES = {
    "regulatory_interpretation": [
        {
            "context": "SEBI Circular: A stock broker shall maintain a minimum net worth of "
                       "Rs. 3 crores at all times. The net worth shall be calculated as per "
                       "the guidelines issued by SEBI from time to time.",
            "question": "What is the minimum net worth requirement for a stock broker?",
            "answer": "Rs. 3 crores"
        },
        {
            "context": "SEBI LODR Regulations 2015, Regulation 17(1): The board of directors "
                       "of a listed entity shall have not less than fifty percent of the board "
                       "comprising of non-executive directors.",
            "question": "What minimum percentage of the board must be non-executive directors?",
            "answer": "At least fifty percent"
        },
        {
            "context": "RBI Direction: A Small Finance Bank shall maintain a minimum Capital "
                       "to Risk-weighted Assets Ratio (CRAR) of 15% on an ongoing basis.",
            "question": "What is the minimum CRAR requirement for a Small Finance Bank?",
            "answer": "15%"
        },
    ],
    "numerical_reasoning": [
        {
            "context": "As per RBI guidelines, a bank with Tier-1 Capital Ratio of 12% and "
                       "Adjusted Net Profit (ANP) of Rs. 500 crore may declare dividends up "
                       "to 40% of ANP. Banks with ratio above 11% but below 12.5% may declare "
                       "up to 35% of ANP.",
            "question": "A bank has Tier-1 Capital of Rs. 1,200 crore and Risk Weighted Assets "
                        "of Rs. 10,000 crore with ANP of Rs. 800 crore. Maximum dividend?",
            "answer": "Tier-1 Capital Ratio = 1200/10000 = 12%. Maximum dividend = 40% x 800 "
                      "= Rs. 320 crore"
        },
        {
            "context": "SEBI circular: The upfront margin for equity derivatives shall be 20% "
                       "of the contract value. The contract multiplier for index futures is 50.",
            "question": "Calculate the upfront margin for a Nifty 50 futures contract if the "
                        "index is at 18,500 points.",
            "answer": "Contract value = 18,500 x 50 = Rs. 9,25,000. "
                      "Upfront margin = 20% x 9,25,000 = Rs. 1,85,000"
        },
        {
            "context": "RBI: Government of India announces sale of 7.26% Government Stock 2029 "
                       "for notified amount of Rs. 6,000 crore. Non-competitive bids up to 5% "
                       "of notified amount will be accepted separately.",
            "question": "What is the maximum amount of non-competitive bids for this auction?",
            "answer": "5% x Rs. 6,000 crore = Rs. 300 crore"
        },
    ],
    "contradiction_detection": [
        {
            "context": "PASSAGE A:\nSEBI Circular 2019: Listed companies must file quarterly "
                       "results within 45 days of end of quarter.\n\n---\n\nPASSAGE B:\n"
                       "SEBI Circular 2022: Listed companies must submit quarterly results "
                       "within 60 days from end of quarter.",
            "question": "Do the two passages contradict each other on the filing deadline?",
            "answer": "Yes. Passage A says 45 days, Passage B says 60 days — direct contradiction."
        },
        {
            "context": "PASSAGE A:\nRBI: Cash Reserve Ratio for scheduled commercial banks "
                       "is 4% of NDTL.\n\n---\n\nPASSAGE B:\nScheduled commercial banks must "
                       "maintain four percent of net demand and time liabilities as CRR.",
            "question": "Do the passages contradict each other on the CRR requirement?",
            "answer": "No. Both state the same 4% of NDTL requirement in different words."
        },
        {
            "context": "PASSAGE A:\nSEBI LODR Regulation 36: Annual report to shareholders "
                       "at least 21 days before the AGM.\n\n---\n\nPASSAGE B:\nCompanies Act "
                       "2013, Section 136: Financial statements at least 21 days before AGM.",
            "question": "Do the passages contradict each other on the AGM notice period?",
            "answer": "No. Both specify the same 21-day minimum period."
        },
    ],
    "temporal_reasoning": [
        {
            "context": "SEBI notification March 15, 2018: Minimum public shareholding enhanced "
                       "from 25% to 35%, effective October 1, 2018. Companies must comply "
                       "within three years of notification, i.e., by March 15, 2021.",
            "question": "By what date must listed companies comply with the enhanced public "
                        "shareholding requirement?",
            "answer": "March 15, 2021 (three years from March 15, 2018)"
        },
        {
            "context": "RBI Master Circular July 1, 2015: Priority Sector lending targets "
                       "40% of ANBC. This circular supersedes the Master Circular dated "
                       "July 1, 2014 on the same subject.",
            "question": "Which circular is currently operative — 2014 or 2015?",
            "answer": "The 2015 Master Circular, as it supersedes the 2014 circular."
        },
        {
            "context": "SEBI circular November 15, 2018: Business responsibility report shall "
                       "be submitted for financial year ending March 31, 2019 and annually.",
            "question": "For which financial year was the first Business Responsibility Report due?",
            "answer": "Financial year ending March 31, 2019"
        },
    ],
}

SYSTEM_PROMPT = (
    "You are an expert in Indian financial regulation and policy. "
    "Answer questions using ONLY the provided context passage. "
    "Do not use any external knowledge. Be concise and precise. "
    "Give only the answer — no preamble."
)

TOP_4_MODELS = {
    "gemini":      {"label": "Gemini 2.5 Flash", "provider": "vertex",
                   "model_id": "gemini-2.5-flash"},
    "qwen3_32b":   {"label": "Qwen3-32B",         "provider": "groq",
                   "model_id": "qwen/qwen3-32b",
                   "extra": {"reasoning_effort": "none"}},
    "groq70b":     {"label": "LLaMA-3.3-70B",      "provider": "groq",
                   "model_id": "llama-3.3-70b-versatile"},
    "llama4scout": {"label": "Llama 4 Scout 17B",  "provider": "groq",
                   "model_id": "meta-llama/llama-4-scout-17b-16e-instruct"},
}

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(item: dict) -> str:
    task = item.get("task_type", "")
    q    = item.get("question", "")
    
    examples = FEW_SHOT_EXAMPLES.get(task, [])
    shots = ""
    for ex in examples:
        shots += (f"Context:\n{ex['context']}\n\n"
                  f"Question: {ex['question']}\n"
                  f"Answer: {ex['answer']}\n\n---\n\n")
    
    if task == "contradiction_detection":
        ctx_a = item.get("context_a", "")[:1200]
        ctx_b = item.get("context_b", "")[:800]
        q_block = (f"Context:\nPASSAGE A:\n{ctx_a}\n\n"
                   f"PASSAGE B:\n{ctx_b}\n\n"
                   f"Question: {q}\n"
                   f"Answer (Yes or No then one sentence):")
    elif task == "numerical_reasoning":
        ctx = " ".join(item.get("context","").split()[:400])
        q_block = (f"Context:\n{ctx}\n\n"
                   f"Question: {q}\n"
                   f"Answer (show calculation with units):")
    elif task == "temporal_reasoning":
        ctx = " ".join(item.get("context","").split()[:400])
        q_block = (f"Context:\n{ctx}\n\n"
                   f"Question: {q}\n"
                   f"Answer (note relevant dates):")
    else:
        ctx = " ".join(item.get("context","").split()[:400])
        q_block = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
    
    return shots + q_block

# ── Scoring ───────────────────────────────────────────────────────────────────

def normalise(t):
    if not t: return ""
    t = re.sub(r"[₹,]", "", str(t).lower().strip())
    t = re.sub(r"[^\w\s%.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def score(ref, pred, task):
    if not pred or "fail:" in pred.lower(): return 0
    r, p = normalise(ref), normalise(pred)
    if r == p: return 1
    if "contradiction" in task:
        ry = "yes" if r.startswith("yes") else ("no" if r.startswith("no") else "")
        py = "yes" if p.startswith("yes") else ("no" if p.startswith("no") else "")
        if ry and py: return 1 if ry == py else 0
    rn = set(re.findall(r"\d[\d]*\.?\d*", re.sub(r"[₹,]","",r)))
    pn = set(re.findall(r"\d[\d]*\.?\d*", re.sub(r"[₹,]","",p)))
    if rn and pn and rn == pn: return 1
    if rn and pn and rn.issubset(pn): return 1
    if HAVE_RF and _rf.token_set_ratio(r, p)/100 >= 0.72: return 1
    return 0

# ── API callers ───────────────────────────────────────────────────────────────

_vertex_client = None

def get_vertex_client():
    global _vertex_client
    if _vertex_client: return _vertex_client
    from google import genai
    from google.genai.types import HttpOptions
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "finindiabench")
    _vertex_client = genai.Client(
        vertexai=True, project=project, location="us-central1",
        http_options=HttpOptions(api_version="v1"),
    )
    return _vertex_client

def call_gemini_vertex(model_id: str, prompt: str) -> str:
    for attempt in range(5):
        try:
            client = get_vertex_client()
            resp = client.models.generate_content(
                model=model_id,
                contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
            )
            time.sleep(0.4)
            return resp.text.strip() if resp.text else "FAIL: empty response"
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = 30 * (attempt + 1)
                print(f"    Rate limit — waiting {wait}s...")
                time.sleep(wait)
            elif "503" in err:
                time.sleep(15)
            else:
                print(f"    Vertex error: {err[:80]}")
                time.sleep(5)
    return "FAIL: Vertex retries exhausted"

def call_groq(model_id: str, prompt: str, extra: dict = None) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key: return "FAIL: GROQ_API_KEY not set"
    try:
        from groq import Groq
    except ImportError:
        return "FAIL: pip install groq"
    client = Groq(api_key=key)
    params = {
        "model": model_id, "max_tokens": 400, "temperature": 0.0,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user",   "content": prompt}],
    }
    if extra:
        params.update(extra)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(**params)
            time.sleep(0.4)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(min(2**attempt*10, 60))
            else:
                return f"FAIL: {str(e)[:80]}"
    return "FAIL: Groq retries exhausted"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",
                   default="annotation/raw_qa/indiafinbench_qa_combined_406.json")
    p.add_argument("--output", default="evaluation/results_fewshot/")
    p.add_argument("--models", nargs="+", default=list(TOP_4_MODELS.keys()))
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items")

    ABBR = {"regulatory_interpretation":"REG","numerical_reasoning":"NUM",
            "contradiction_detection":"CON","temporal_reasoning":"TMP"}

    summary_rows = []

    for mk in args.models:
        cfg   = TOP_4_MODELS.get(mk)
        if not cfg: print(f"Unknown: {mk}"); continue
        label = cfg["label"]
        out_csv = out_dir / f"{mk}_3shot_results.csv"

        # Resume
        done = {}
        if out_csv.exists():
            with open(out_csv, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if "FAIL" not in row.get("prediction",""):
                        done[row["id"]] = row
            print(f"\n{label}: resuming — {len(done)} done")
        else:
            print(f"\n{label}: starting fresh")

        results = list(done.values())
        errors  = 0

        for i, item in enumerate(data):
            if item.get("id","") in done: continue

            task   = item.get("task_type","")
            prompt = build_prompt(item)

            if cfg["provider"] == "vertex":
                pred = call_gemini_vertex(cfg["model_id"], prompt)
            else:
                pred = call_groq(cfg["model_id"], prompt, cfg.get("extra"))

            if "FAIL" in pred:
                errors += 1
                if errors <= 3:
                    print(f"    API error: {pred[:80]}")
                correct = 0
            else:
                # Use 'answer' field — correct field name in your dataset
                ref     = item.get("answer", "")
                correct = score(ref, pred, task)

            results.append({
                "id":         item.get("id",""),
                "task_type":  task,
                "difficulty": item.get("difficulty",""),
                "question":   item.get("question","")[:80],
                "ref_answer": item.get("answer",""),
                "prediction": pred[:300],
                "correct":    correct,
                "condition":  "3shot",
            })
            done[item.get("id","")] = results[-1]

            if (i+1) % 25 == 0:
                _save(results, out_csv)
                valid = [r for r in results if "FAIL" not in r.get("prediction","")]
                acc   = sum(int(r["correct"]) for r in valid)/len(valid)*100 if valid else 0
                print(f"  [{i+1}/{len(data)}] acc={acc:.1f}%  errors={errors}")

        _save(results, out_csv)

        # Compute accuracy — exclude FAILs
        valid   = [r for r in results if "FAIL" not in r.get("prediction","")]
        ts_dict = defaultdict(list)
        for r in valid:
            ts_dict[ABBR.get(r["task_type"],r["task_type"])].append(int(r["correct"]))
        overall = sum(int(r["correct"]) for r in valid)/len(valid)*100 if valid else 0

        print(f"\n  {label} 3-shot:")
        row = {"model": label, "condition": "3shot", "overall": round(overall,1)}
        for ts in ["REG","NUM","CON","TMP"]:
            s = ts_dict.get(ts,[])
            a = sum(s)/len(s)*100 if s else 0
            row[ts] = round(a,1)
            print(f"    {ts}: {a:.1f}%  (n={len(s)})")
        print(f"    Overall: {overall:.1f}%  ({errors} FAIL errors)")
        summary_rows.append(row)

    if summary_rows:
        sp = out_dir / "fewshot_summary.csv"
        with open(sp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)

        print(f"\n{'='*62}")
        print("  FEW-SHOT vs ZERO-SHOT (compare to your Table 6)")
        print(f"{'='*62}")
        print(f"  {'Model':<25} {'REG':>6} {'NUM':>6} {'CON':>6} {'TMP':>6} {'Overall':>8}")
        for r in summary_rows:
            print(f"  {r['model']:<25} "
                  f"{r.get('REG',0):>5.1f}% {r.get('NUM',0):>5.1f}% "
                  f"{r.get('CON',0):>5.1f}% {r.get('TMP',0):>5.1f}% "
                  f"{r.get('overall',0):>7.1f}%")
        print(f"\n  Saved: {sp}")

def _save(rows, path):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()