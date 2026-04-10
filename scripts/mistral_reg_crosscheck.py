"""
mistral_reg_crosscheck.py
Runs Mistral-7B on 20 REG items to verify LLaMA validator is unbiased.
Run from IndiaFinBench/ root: python scripts/mistral_reg_crosscheck.py
"""
import json, random, re, requests, sys
from pathlib import Path

try:
    from rapidfuzz import fuzz
except ImportError:
    print("pip install rapidfuzz"); sys.exit(1)

QA_PATH = "annotation/raw_qa/indiafinbench_qa_combined_150.json"
SEED    = 42
N_ITEMS = 20
random.seed(SEED)

with open(QA_PATH, encoding="utf-8") as f:
    data = json.load(f)

reg_items = [item for item in data if item["task_type"] == "regulatory_interpretation"]
sample    = random.sample(reg_items, min(N_ITEMS, len(reg_items)))
print(f"Sampled {len(sample)} REG items for Mistral cross-validation")

SYSTEM_PROMPT = """You are an annotation assistant for a research benchmark dataset.
Your ONLY job is to answer questions using the provided context passage.
Use ONLY information present in the context. No outside knowledge.
Give only the answer. No explanations."""

def build_prompt(item):
    return (f"Context passage:\n{item['context']}\n\n"
            f"Question: {item['question']}\n\nAnswer:")

def query_mistral(prompt, retries=3):
    url     = "http://localhost:11434/api/chat"
    payload = {"model": "mistral", "stream": False,
                "options": {"temperature": 0.0},
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user",   "content": prompt}]}
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            if attempt == retries - 1: return f"API_ERROR: {e}"
            import time; time.sleep(5)
    return "API_ERROR"

def normalise(text):
    if not text: return ""
    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s%.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def score(ref, pred):
    rn, pn = normalise(ref), normalise(pred)
    if rn == pn: return 1
    rn2 = set(re.findall(r"\d+[\d,]*\.?\d*", rn))
    pn2 = set(re.findall(r"\d+[\d,]*\.?\d*", pn))
    if rn2 and pn2 and rn2 == pn2: return 1
    if fuzz.token_set_ratio(rn, pn) / 100.0 >= 0.72: return 1
    return 0

# Check Ollama
try:
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    if not any("mistral" in m for m in models):
        print("ERROR: mistral not in Ollama. Run: ollama pull mistral"); sys.exit(1)
    print(f"Ollama OK. Models: {models}")
except Exception as e:
    print(f"ERROR: Cannot connect to Ollama: {e}"); sys.exit(1)

print(f"\nRunning Mistral on {len(sample)} REG items...\n")
n_correct = 0
for i, item in enumerate(sample, 1):
    pred    = query_mistral(build_prompt(item))
    correct = score(item["answer"], pred)
    n_correct += correct
    status  = "✓" if correct else "✗"
    print(f"  [{i:02d}] {status}  {item['id'][:20]:<20}  pred: {pred[:50]}")

agreement_pct = n_correct / len(sample) * 100
print(f"\n{'='*55}")
print(f"  RESULT: {agreement_pct:.1f}% agreement ({n_correct}/{len(sample)})")
print(f"{'='*55}")
print(f"\n  PASTE INTO PAPER (Section 3.4):")
print(f"  Replace [X]% with: {agreement_pct:.1f}%")