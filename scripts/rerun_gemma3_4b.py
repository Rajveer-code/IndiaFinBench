"""Re-run Gemma on the full 406-item benchmark under a confirmed model identity.

Why this exists: the original evaluate_v7_models.py requested Ollama tag
"gemma4", which is not a real Google release (no surviving logs confirm what
it actually resolved to). Confirmed via /api/show that gemma3:4b is a real,
currently-pulled model (family gemma3, 4,299,915,632 params, Q4_K_M) and
matches the paper's methods text. Re-running under this confirmed identity
replaces the ambiguous result rather than guessing at it.

Reuses evaluate_v7_models.evaluate_model() directly -- same prompts, same
four-stage scorer, same CSV schema as every other model -- so results stay
comparable. Deliberately does NOT call its main()'s
print_full_leaderboard()/update_baselines(), which write stale placeholder
numbers (including a third, different "n=30" human baseline) into the live
HuggingFace demo's demo/data/baselines.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate_v7_models import MODELS, QA_PATH, evaluate_model  # noqa: E402

MODELS["gemma4_e4b"]["model_id"] = "gemma3:4b"  # was the nonexistent tag "gemma4"

with open(QA_PATH, encoding="utf-8") as f:
    data = json.load(f)
assert len(data) == 406, f"expected 406 items, got {len(data)}"

print(f"Re-running Gemma on {len(data)} items as gemma3:4b (confirmed identity).")
acc = evaluate_model("gemma4_e4b", data)
print("\nFinal:", json.dumps(acc, indent=2))
