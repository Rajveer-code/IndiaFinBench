"""
Master runner: executes all experiments in order and produces final summary.
Run this AFTER setting up GEMINI_API_KEY and GROQ_API_KEY.

Usage:
  python scripts/run_all_experiments.py              # all experiments
  python scripts/run_all_experiments.py --skip-api   # offline only
  python scripts/run_all_experiments.py --only exp1  # single experiment
"""
import subprocess, sys, json, time, os, argparse
from pathlib import Path

SCRIPTS = [
    ("exp1_temporal_depth.py",      "Temporal Chain Depth Analysis",      False),
    ("exp3_con_balance.py",         "CON Class Balance",                   False),
    ("exp5_context_length.py",      "Context Length vs Accuracy",          False),
    ("exp6_feature_regression.py",  "Text Feature Difficulty Regression",  False),
    ("exp7_item_discrimination.py", "Item Discrimination (IRT-lite)",      False),
    ("exp8_error_geometry.py",      "Error Geometry + Dissociation Index", False),
    ("exp11_structural_analysis.py","Structural Analysis (Kendall W)",     False),
    ("exp2_rsts_metric.py",         "RSTS Metric (needs Gemini API)",      True),
    ("exp4_scoring_audit.py",       "Scoring Audit (needs Gemini API)",    True),
    ("exp9_rag_evaluation.py",      "RAG Evaluation (needs APIs + GPU)",   True),
    ("exp10_perturbation.py",       "Perturbation Robustness (Gemini)",    True),
]


def run_experiment(script_name, description, needs_api):
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"  MISSING: Script not found: {script_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
        text=True,
        timeout=3600
    )

    if result.returncode == 0:
        print(f"  COMPLETE: {description}")
        return True
    else:
        print(f"  FAILED: {description} (return code {result.returncode})")
        return False


def check_api_keys():
    gemini_ok = bool(os.environ.get("GEMINI_API_KEY"))
    groq_ok = bool(os.environ.get("GROQ_API_KEY"))
    print(f"API Keys: Gemini={'OK' if gemini_ok else 'MISSING'}, Groq={'OK' if groq_ok else 'MISSING'}")
    return gemini_ok, groq_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all IndiaFinBench experiments")
    parser.add_argument("--skip-api", action="store_true", help="Skip experiments requiring API keys")
    parser.add_argument("--only", type=str, help="Run only specific experiment (e.g., 'exp1')")
    args = parser.parse_args()

    gemini_ok, groq_ok = check_api_keys()

    results = {}
    for script_name, description, needs_api in SCRIPTS:
        if args.only and args.only not in script_name:
            continue
        if args.skip_api and needs_api:
            print(f"  SKIPPED (needs API): {description}")
            continue
        if needs_api and not (gemini_ok or groq_ok):
            print(f"  SKIPPED (no API keys): {description}")
            continue

        success = run_experiment(script_name, description, needs_api)
        results[script_name] = success

        if needs_api:
            print("  Waiting 5s before next API experiment...")
            time.sleep(5)

    print(f"\n{'='*60}")
    print("MASTER RUNNER COMPLETE")
    print(f"{'='*60}")
    for script, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {script}")

    print(f"\nAll figures saved to: paper/figures/novel_methods/")
    print(f"All data saved to:    evaluation/novel_methods/")
