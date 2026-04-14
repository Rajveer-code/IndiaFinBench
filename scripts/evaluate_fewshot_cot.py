"""
evaluate_fewshot_cot.py
------------------------
PURPOSE
    Run few-shot (1-shot, 3-shot) and zero-shot CoT evaluations on the
    hardest 60 items in IndiaFinBench, for the top-3 models.

    This addresses the Phase 2 Improvement 1 gap: the paper acknowledges
    zero-shot only. Reviewers expect at least a few-shot ablation.

MODELS
    Runs on: Claude 3.5 Haiku, Gemini 2.5 Flash, LLaMA-3.3-70B (Groq)
    (Edit MODEL_CFG below to change models)

CONDITIONS
    zero_shot     : original baseline (from existing CSVs, reproduced here for completeness)
    zero_shot_cot : adds "Think step by step before answering." to system prompt
    one_shot      : one in-context example per task type
    three_shot    : three in-context examples per task type

INPUTS
    annotation/raw_qa/indiafinbench_qa_combined_150.json
    (Examples for few-shot are drawn from the easiest items NOT in the 60-item hard subset)

OUTPUTS
    evaluation/results/fewshot/  (one CSV per model × condition)
    evaluation/error_analysis/fewshot_summary.csv
    paper/tables/table_fewshot.tex

USAGE
    export ANTHROPIC_API_KEY=...
    export GOOGLE_API_KEY=...
    export GROQ_API_KEY=...
    python scripts/evaluate_fewshot_cot.py
    python scripts/evaluate_fewshot_cot.py --models haiku --conditions zero_shot_cot one_shot
"""
from __future__ import annotations
import json, csv, os, re, time, sys, io, argparse
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    from rapidfuzz import fuzz as _rf
except ImportError:
    raise ImportError("pip install rapidfuzz")

BASE        = Path(__file__).parent.parent
QA_PATH     = BASE / "annotation" / "raw_qa" / "indiafinbench_qa_combined_150.json"
RESULTS_DIR = BASE / "evaluation" / "results" / "fewshot"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model configuration ────────────────────────────────────────────────────────
MODEL_CFG = {
    "haiku": {
        "label":    "Claude 3.5 Haiku",
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",  # update to latest available
    },
    "gemini": {
        "label":    "Gemini 2.5 Flash",
        "provider": "gemini",
        "model_id": "gemini-2.5-flash",
    },
    "groq70b": {
        "label":    "LLaMA-3.3-70B",
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
    },
}

CONDITIONS = ["zero_shot", "zero_shot_cot", "one_shot", "three_shot"]

TASK_TYPES = [
    "regulatory_interpretation",
    "numerical_reasoning",
    "contradiction_detection",
    "temporal_reasoning",
]

# ── System prompts ─────────────────────────────────────────────────────────────
SYSTEM_BASE = (
    "You are an expert in Indian financial regulation and policy. "
    "Answer questions using ONLY the provided context passage. "
    "Do not use any external knowledge. "
    "Be concise and precise. Give only the answer — no preamble."
)
SYSTEM_COT = SYSTEM_BASE + "\n\nThink step by step before answering."


# ── Scoring (identical to evaluate.py) ────────────────────────────────────────
def normalise(text: str) -> str:
    if not text: return ""
    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s%.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def score_answer(ref: str, pred: str, task_type: str) -> int:
    if not pred or "fail:" in pred.lower():
        return 0
    if task_type == "contradiction_detection":
        r, p = normalise(ref), normalise(pred)
        ry = "yes" if r.startswith("yes") else ("no" if r.startswith("no") else r)
        py = "yes" if p.startswith("yes") else ("no" if p.startswith("no") else p)
        return int(ry == py)
    rn, pn = normalise(ref), normalise(pred)
    nr = set(re.findall(r"\d[\d,]*\.?\d*", rn))
    np_ = set(re.findall(r"\d[\d,]*\.?\d*", pn))
    if nr and np_ and nr == np_: return 1
    return int(_rf.token_set_ratio(rn, pn) / 100.0 >= 0.72)


# ── Select 60-item hard subset ─────────────────────────────────────────────────
def select_hard_subset(data: list[dict]) -> list[dict]:
    """Return the 60 hardest items: all 'hard' + lowest-ranked 'medium'.

    Stratify by task type to avoid depleting any task.
    Also return the remaining items as the example pool for few-shot.
    """
    hard_items   = [x for x in data if x.get("difficulty","").lower() == "hard"]
    medium_items = [x for x in data if x.get("difficulty","").lower() == "medium"]
    easy_items   = [x for x in data if x.get("difficulty","").lower() == "easy"]

    n_needed = max(0, 60 - len(hard_items))
    # Take medium items by task, proportional to task size
    medium_per_task: dict[str, list] = defaultdict(list)
    for item in medium_items:
        medium_per_task[item["task_type"]].append(item)

    selected_medium = []
    n_per_task = n_needed // len(TASK_TYPES)
    remainder  = n_needed % len(TASK_TYPES)
    for i, task in enumerate(TASK_TYPES):
        take = n_per_task + (1 if i < remainder else 0)
        selected_medium.extend(medium_per_task[task][:take])

    subset = hard_items + selected_medium
    # Example pool: items NOT in subset (easy + leftover medium)
    subset_ids = {x["id"] for x in subset}
    pool = [x for x in data if x["id"] not in subset_ids]
    print(f"  Hard subset: {len(subset)} items ({len(hard_items)} hard + {len(selected_medium)} medium)")
    print(f"  Example pool: {len(pool)} items")
    return subset, pool


# ── Few-shot example builder ───────────────────────────────────────────────────
def build_examples(pool: list[dict], task_type: str, n: int) -> list[dict]:
    """Select n easy examples of the given task type from the pool."""
    candidates = [x for x in pool if x["task_type"] == task_type
                  and x.get("difficulty","").lower() == "easy"]
    if len(candidates) < n:
        # fall back to any pool items of this task type
        candidates = [x for x in pool if x["task_type"] == task_type]
    return candidates[:n]


def format_example(item: dict) -> str:
    """Format a QA item as a few-shot example."""
    task = item["task_type"]
    if task == "contradiction_detection":
        return (
            f"Passage A:\n{item.get('context_a','')[:800]}\n\n"
            f"Passage B:\n{item.get('context_b','')[:800]}\n\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )
    ctx_words = item.get("context","").split()
    ctx = " ".join(ctx_words[:300])
    return f"Context:\n{ctx}\n\nQuestion: {item['question']}\nAnswer: {item['answer']}"


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(item: dict, examples: list[dict], condition: str) -> str:
    task = item["task_type"]
    ctx_words = item.get("context","").split()
    ctx = " ".join(ctx_words[:450])

    # Build example block
    example_block = ""
    if examples:
        parts = []
        for ex in examples:
            parts.append("--- Example ---\n" + format_example(ex))
        example_block = "\n\n".join(parts) + "\n\n--- Your question ---\n"

    if task == "contradiction_detection":
        q_block = (
            f"Passage A:\n{item.get('context_a','')[:1500]}\n\n"
            f"Passage B:\n{item.get('context_b','')[:1500]}\n\n"
            f"Question: {item['question']}\n\n"
            f"Answer with 'Yes' or 'No' then one sentence of explanation:"
        )
    elif task == "numerical_reasoning":
        q_block = (
            f"Context:\n{ctx}\n\nQuestion: {item['question']}\n\n"
            f"Show your calculation and give the final answer with units:"
        )
    elif task == "temporal_reasoning":
        q_block = (
            f"Context:\n{ctx}\n\nQuestion: {item['question']}\n\n"
            f"Answer precisely, noting relevant dates or sequences:"
        )
    else:
        q_block = f"Context:\n{ctx}\n\nQuestion: {item['question']}\n\nAnswer:"

    return example_block + q_block


# ── API callers (reuse pattern from evaluate.py) ───────────────────────────────
def call_anthropic(model_id: str, system: str, prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=model_id, max_tokens=400, system=system,
                messages=[{"role":"user","content":prompt}])
            return resp.content[0].text.strip()
        except Exception as e:
            if "529" in str(e) or "overloaded" in str(e).lower():
                time.sleep(min(2**attempt*5, 60)); continue
            return f"FAIL: {str(e)[:100]}"
    return "FAIL: Anthropic retries exhausted"


def call_gemini(model_id: str, system: str, prompt: str) -> str:
    from google import genai
    key = os.environ.get("GOOGLE_API_KEY","")
    if not key: return "FAIL: GOOGLE_API_KEY not set"
    for attempt in range(8):
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=model_id, contents=f"{system}\n\n{prompt}",
                config={"temperature":0.0,"max_output_tokens":400,"thinking_config":{"thinking_budget":0}})
            return resp.text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(min(2**attempt*10, 120)); continue
            return f"FAIL: {str(e)[:100]}"
    return "FAIL: Gemini retries exhausted"


def call_groq(model_id: str, system: str, prompt: str) -> str:
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY","")
    if not key: return "FAIL: GROQ_API_KEY not set"
    client = Groq(api_key=key)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model_id, max_tokens=400, temperature=0.0,
                messages=[{"role":"system","content":system},{"role":"user","content":prompt}])
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(min(2**attempt*5,60)); continue
            return f"FAIL: {str(e)[:100]}"
    return "FAIL: Groq retries exhausted"


def call_model(cfg: dict, system: str, prompt: str) -> str:
    provider = cfg["provider"]
    if provider == "anthropic": return call_anthropic(cfg["model_id"], system, prompt)
    if provider == "gemini":    return call_gemini(cfg["model_id"], system, prompt)
    if provider == "groq":      return call_groq(cfg["model_id"], system, prompt)
    return f"FAIL: unknown provider {provider}"


# ── Core evaluation ────────────────────────────────────────────────────────────
def evaluate_condition(model_key: str, condition: str, subset: list[dict],
                       pool: list[dict]) -> list[dict]:
    cfg   = MODEL_CFG[model_key]
    label = cfg["label"]
    out_path = RESULTS_DIR / f"{model_key}_{condition}.csv"

    # Load checkpoint
    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "FAIL" not in row.get("prediction",""):
                    done[row["id"]] = row

    remaining = len(subset) - len(done)
    print(f"\n  {label} | {condition} | {len(done)}/60 done | {remaining} remaining")

    if remaining == 0:
        print("  Already complete — skipping.")
        return list(done.values())

    system = SYSTEM_COT if condition == "zero_shot_cot" else SYSTEM_BASE
    n_examples = {"zero_shot":0,"zero_shot_cot":0,"one_shot":1,"three_shot":3}[condition]

    # Build example pool per task type
    examples_by_task: dict[str, list] = {}
    for task in TASK_TYPES:
        examples_by_task[task] = build_examples(pool, task, n_examples)

    results = list(done.values())
    delay = {"anthropic":0.5, "gemini":1.0, "groq":0.5}.get(cfg["provider"], 1.0)

    for item in subset:
        if item["id"] in done: continue
        examples = examples_by_task[item["task_type"]]
        prompt   = build_prompt(item, examples, condition)
        pred     = call_model(cfg, system, prompt)
        correct  = score_answer(item["answer"], pred, item["task_type"])
        results.append({
            "id": item["id"], "task_type": item["task_type"],
            "difficulty": item.get("difficulty",""), "condition": condition,
            "question": item["question"][:80], "ref_answer": item["answer"],
            "prediction": pred[:300], "correct": correct,
        })
        status = "✓" if correct else "✗"
        print(f"  [{len(results):2d}/60] {status} {item['id']:<12} {pred[:50]}")

        if len(results) % 15 == 0:
            _write(results, out_path)
        time.sleep(delay)

    _write(results, out_path)
    return results


def _write(rows: list[dict], path: Path) -> None:
    fields = ["id","task_type","difficulty","condition","question","ref_answer","prediction","correct"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def summarise(all_results: dict[tuple, list]) -> None:
    """Print and save summary table + LaTeX."""
    print(f"\n{'='*70}")
    print("  Few-shot / CoT Summary (60-item hard subset)")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Condition':<18} {'REG':>6} {'NUM':>6} {'CON':>6} {'TMP':>6} {'Overall':>8}")
    print(f"  {'-'*70}")

    summary_rows = []
    for (mk, cond), rows in sorted(all_results.items(), key=lambda x:(x[0][0],CONDITIONS.index(x[0][1]))):
        label = MODEL_CFG[mk]["label"]
        task_scores: dict[str,list] = defaultdict(list)
        for r in rows:
            task_scores[r["task_type"]].append(int(r["correct"]))
        cells = {}
        for task in TASK_TYPES:
            ts = {"regulatory_interpretation":"REG","numerical_reasoning":"NUM",
                  "contradiction_detection":"CON","temporal_reasoning":"TMP"}[task]
            s = task_scores.get(task,[])
            cells[ts] = sum(s)/len(s)*100 if s else float("nan")
        all_s = [int(r["correct"]) for r in rows]
        overall = sum(all_s)/len(all_s)*100 if all_s else float("nan")
        summary_rows.append({"model":label,"condition":cond,**cells,"overall":overall})
        print(f"  {label:<25} {cond:<18} "
              f"{cells.get('REG',float('nan')):>5.1f}%  {cells.get('NUM',float('nan')):>5.1f}%  "
              f"{cells.get('CON',float('nan')):>5.1f}%  {cells.get('TMP',float('nan')):>5.1f}%  {overall:>7.1f}%")

    # Save CSV
    pd_rows = pd.DataFrame(summary_rows)
    out_csv = BASE/"evaluation"/"error_analysis"/"fewshot_summary.csv"
    pd_rows.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    # LaTeX table
    out_tex = BASE/"paper"/"tables"/"table_fewshot.tex"
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Few-shot and CoT prompting results on the 60-item hard subset.",
        r"  zero\_shot: original condition; zero\_shot\_cot: +step-by-step CoT;",
        r"  one\_shot/three\_shot: in-context examples from the same task type.}",
        r"\label{tab:fewshot}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Condition} & \textbf{REG} & \textbf{NUM} & \textbf{CON} & \textbf{TMP} & \textbf{Overall} \\",
        r"\midrule",
    ]
    prev_model = None
    for row in summary_rows:
        if row["model"] != prev_model and prev_model is not None:
            lines.append(r"\midrule")
        prev_model = row["model"]
        cond_disp = row["condition"].replace("_","\\_")
        cells = " & ".join(
            f"{row.get(t,float('nan')):.1f}" if not (isinstance(row.get(t),float) and row.get(t)!=row.get(t)) else "--"
            for t in ["REG","NUM","CON","TMP","overall"])
        lines.append(f"{row['model']} & {cond_disp} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"  Saved: {out_tex}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     nargs="+", default=list(MODEL_CFG.keys()))
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    args = parser.parse_args()

    if not QA_PATH.exists():
        print(f"ERROR: {QA_PATH} not found"); sys.exit(1)

    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items")

    subset, pool = select_hard_subset(data)

    all_results: dict[tuple, list] = {}
    for mk in args.models:
        if mk not in MODEL_CFG:
            print(f"Unknown model: {mk}. Available: {list(MODEL_CFG)}"); continue
        for cond in args.conditions:
            if cond not in CONDITIONS:
                print(f"Unknown condition: {cond}. Available: {CONDITIONS}"); continue
            rows = evaluate_condition(mk, cond, subset, pool)
            all_results[(mk, cond)] = rows

    summarise(all_results)
    print("\nFew-shot/CoT evaluation complete.")


if __name__ == "__main__":
    main()
