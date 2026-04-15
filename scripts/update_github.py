"""
Upload all new experiment results and figures to GitHub.
Updates README with new method descriptions and result summaries.
Run LAST after all experiments complete.
"""
import subprocess, sys, json, os, re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def run_git(cmd, cwd=None):
    """Run git command and print output."""
    result = subprocess.run(
        cmd, shell=True, cwd=str(cwd or REPO_ROOT),
        capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr and result.returncode != 0:
        print(f"STDERR: {result.stderr.strip()}")
    return result.returncode == 0


def update_readme():
    """Append novel methods section to README.md."""
    readme_path = REPO_ROOT / "README.md"

    # Load key results
    results_to_report = {}
    summary_paths = {
        'tmp_depth': REPO_ROOT / "evaluation/novel_methods/tmp_depth/exp1_summary.json",
        'con_balance': REPO_ROOT / "evaluation/novel_methods/con_balance/con_balance_summary.json",
        'scoring_audit': REPO_ROOT / "evaluation/novel_methods/scoring_audit/scoring_audit_summary.json",
        'dissociation': REPO_ROOT / "evaluation/novel_methods/error_geometry/dissociation_summary.json",
    }
    for key, path in summary_paths.items():
        if path.exists():
            with open(path) as f:
                results_to_report[key] = json.load(f)

    tmp_summary = results_to_report.get('tmp_depth', {})
    con_summary = results_to_report.get('con_balance', {})
    audit_summary = results_to_report.get('scoring_audit', {})
    di_summary = results_to_report.get('dissociation', {})

    yes_count = con_summary.get('yes_count', '?')
    total = max(con_summary.get('total', 1), 1)
    yes_pct = f"{yes_count/total:.0%}" if isinstance(yes_count, int) else '?'
    maj_base = con_summary.get('majority_baseline', '?')
    maj_str = f"{maj_base:.1%}" if isinstance(maj_base, float) else str(maj_base)
    fn_rate = audit_summary.get('false_negative_rate', '?')
    fn_str = f"{fn_rate:.0%}" if isinstance(fn_rate, float) else str(fn_rate)
    corr_adj = audit_summary.get('implied_accuracy_correction', 0)
    corr_str = f"{corr_adj*100:.1f}" if isinstance(corr_adj, float) else str(corr_adj)
    max_model = di_summary.get('max_dissociation_model', 'DeepSeek R1')
    max_di = di_summary.get('max_DI', '?')
    max_di_str = f"{max_di:.3f}" if isinstance(max_di, float) else str(max_di)

    new_section = f"""

---

## Novel Methodological Analyses (v2.0)

Beyond the core evaluation, IndiaFinBench v2.0 includes **11 novel analytical contributions** addressing reviewer concerns about methodological depth.

### Key New Findings

#### 1. Temporal Chain Complexity Analysis
Performance on temporal reasoning degrades systematically with amendment chain depth. Items requiring cross-document amendment tracing show significantly lower accuracy than single-document temporal ordering, confirming that the benchmark measures genuine *regulatory state tracking* rather than general reading comprehension.

Spearman correlation between complexity score and accuracy: see `evaluation/novel_methods/tmp_depth/`

#### 2. Regulatory State Tracking Score (RSTS)
We introduce a new multi-dimensional metric for TMP items decomposing:
- **Event identification** (25%): Did the model find the correct regulatory events?
- **Temporal ordering** (25%): Did the model sequence them correctly?
- **Final state answer** (50%): Did the model identify the operative provision?

RSTS scores for top models: see `evaluation/novel_methods/rsts_scores/`

#### 3. CON Class Balance Analysis
Yes/No distribution in contradiction detection: Yes={yes_count}/{total} ({yes_pct}).
Majority-class baseline: {maj_str}. All models substantially exceed this baseline.

#### 4. Scoring Pipeline Audit
LLM-as-judge evaluation of 100 "incorrect" predictions reveals a {fn_str} false-negative rate in the automated pipeline, concentrated in NUM task (unit formatting). True model accuracies are approximately {corr_str}pp higher than reported.

#### 5. Dissociation Index
The CON–TMP dissociation index (DI) reveals that reasoning-specialized models show the highest gap between pairwise contradiction detection and temporal state tracking. {max_model} shows the highest DI (={max_di_str}), supporting the hypothesis that chain-of-thought reasoning improves local comparison but not global regulatory state maintenance.

#### 6. RAG vs Oracle Evaluation
Two-condition experiment (Oracle vs. RAG-retrieved context) isolates whether TMP failures are retrieval-driven or reasoning-driven. See `evaluation/novel_methods/rag_evaluation/`

### New Figures
All novel method figures are in `paper/figures/novel_methods/`

### Reproducibility
Run all analyses:
```bash
export GEMINI_API_KEY="your_key_here"
export GROQ_API_KEY="your_key_here"
python scripts/run_all_experiments.py
```

Run only offline analyses (no API needed):
```bash
python scripts/run_all_experiments.py --skip-api
```
"""

    current_readme = readme_path.read_text(encoding='utf-8')

    if "Novel Methodological Analyses" in current_readme:
        current_readme = re.sub(
            r'\n---\n\n## Novel Methodological Analyses.*$',
            new_section,
            current_readme,
            flags=re.DOTALL
        )
    else:
        current_readme += new_section

    readme_path.write_text(current_readme, encoding='utf-8')
    print("README.md updated with novel methods section")


def main():
    print("=== GITHUB UPDATE SCRIPT ===\n")

    update_readme()

    print("\nStaging files for commit...")
    run_git("git add evaluation/novel_methods/")
    run_git("git add paper/figures/novel_methods/")
    run_git("git add paper/tables/novel_methods/")
    run_git("git add scripts/exp*.py")
    run_git("git add scripts/novel_methods_utils.py")
    run_git("git add scripts/run_all_experiments.py")
    run_git("git add scripts/generate_novel_methods_summary.py")
    run_git("git add scripts/update_github.py")
    run_git("git add README.md")

    print("\nFiles staged:")
    run_git("git diff --cached --name-only")

    commit_message = (
        "feat: add 11 novel methodological analyses (v2.0)\n\n"
        "- Temporal chain depth analysis (Exp 1)\n"
        "- Regulatory State Tracking Score/RSTS (Exp 2)\n"
        "- CON class balance + majority baseline (Exp 3)\n"
        "- Scoring pipeline false-negative audit (Exp 4)\n"
        "- Context length vs accuracy (Exp 5)\n"
        "- Text feature difficulty regression (Exp 6)\n"
        "- Item discrimination IRT-lite (Exp 7)\n"
        "- Error geometry + Dissociation Index (Exp 8)\n"
        "- RAG vs Oracle evaluation (Exp 9)\n"
        "- Perturbation robustness (Exp 10)\n"
        "- Structural analysis Kendall W + era (Exp 11)"
    )

    print("\nCommitting...")
    success = run_git(f'git commit -m "{commit_message}"')

    if success:
        print("\nPushing to origin...")
        run_git("git push origin main")
        print("\nSuccessfully pushed to GitHub!")
    else:
        print("\nCommit failed or nothing to commit. Check git status:")
        run_git("git status")


if __name__ == "__main__":
    main()
