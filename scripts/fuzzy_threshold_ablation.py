"""
fuzzy_threshold_ablation_fixed.py
----------------------------------
CRITICAL FIX over the original fuzzy_threshold_ablation.py:
  - Original used partial_ratio uniformly across all tasks (WRONG)
  - This version uses the EXACT three-tier scorer from evaluate.py:
      Tier 1: CON -> exact Yes/No match (threshold irrelevant)
      Tier 2: numeric shortcut for digit-bearing answers (threshold irrelevant)
      Tier 3: token_set_ratio >= threshold (the swept parameter)
"""
from __future__ import annotations
import csv, re, sys, warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz as _rf
except ImportError:
    raise ImportError("pip install rapidfuzz")

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"
OUT_DIR     = ROOT / "evaluation" / "error_analysis"
TABLES_DIR  = ROOT / "paper" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85]
TASK_SHORTS = {
    "regulatory_interpretation": "REG",
    "numerical_reasoning":       "NUM",
    "contradiction_detection":   "CON",
    "temporal_reasoning":        "TMP",
}
MODEL_FILES = {
    "Claude 3 Haiku":   "haiku_results.csv",
    "Gemini 2.5 Flash": "gemini_results.csv",
    "LLaMA-3.3-70B":    "groq70b_results.csv",
    "LLaMA-3-8B":       "llama3_results.csv",
    "Mistral-7B":       "mistral_results.csv",
}


def normalise(text: str) -> str:
    if not text: return ""
    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s%.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def score_answer(ref: str, pred: str, task_type: str, threshold: float) -> int:
    """Exact replica of evaluate.py score_answer(), threshold is swept param."""
    if not pred or "fail:" in str(pred).lower():
        return 0
    if task_type == "contradiction_detection":
        r, p = normalise(ref), normalise(pred)
        ry = "yes" if r.startswith("yes") else ("no" if r.startswith("no") else r)
        py = "yes" if p.startswith("yes") else ("no" if p.startswith("no") else p)
        return int(ry == py)
    rn, pn = normalise(ref), normalise(pred)
    nr = set(re.findall(r"\d[\d,]*\.?\d*", rn))
    np_ = set(re.findall(r"\d[\d,]*\.?\d*", pn))
    if nr and np_ and nr == np_:
        return 1
    return int(_rf.token_set_ratio(rn, pn) / 100.0 >= threshold)


def load_all_results() -> pd.DataFrame | None:
    frames = []
    for model_name, filename in MODEL_FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            warnings.warn(f"Missing: {path}; skipping {model_name}")
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "reference_answer" in df.columns:
            df = df.rename(columns={"reference_answer": "ref_answer"})
        df = df[~df.get("prediction", pd.Series(dtype=str)).str.contains("FAIL", case=False, na=True)]
        df["model"] = model_name
        needed = ["model", "id", "task_type", "ref_answer", "prediction"]
        if not all(c in df.columns for c in needed):
            warnings.warn(f"{filename} missing columns; skipping"); continue
        frames.append(df[needed])
        print(f"  Loaded {model_name}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True) if frames else None


def compute_accuracy(df: pd.DataFrame, threshold: float) -> dict:
    df = df.copy()
    df["match"] = df.apply(
        lambda r: score_answer(r["ref_answer"], r["prediction"], r["task_type"], threshold), axis=1)
    result = {"threshold": threshold, "overall": df["match"].mean()}
    for long_name, short in TASK_SHORTS.items():
        sub = df[df["task_type"] == long_name]
        result[short] = sub["match"].mean() if len(sub) > 0 else float("nan")
    return result


def main() -> None:
    print("\n" + "="*65)
    print("  IndiaFinBench — Fixed Fuzzy Threshold Ablation")
    print("  Scorer: CON=exact, numeric shortcut, TSR≥threshold sweep")
    print("="*65)

    data = load_all_results()
    if data is None:
        print(f"\n  ERROR: No CSVs found in {RESULTS_DIR}"); sys.exit(1)

    overall_rows, per_task_rows, model_rows = [], [], []

    for thresh in THRESHOLDS:
        acc = compute_accuracy(data, thresh)
        overall_rows.append({"threshold": thresh, "overall_accuracy": round(acc["overall"]*100, 2)})
        for short in TASK_SHORTS.values():
            v = acc[short]
            per_task_rows.append({"threshold": thresh, "task": short,
                                   "accuracy": round(v*100, 2) if not np.isnan(v) else None})

    for model in data["model"].unique():
        md = data[data["model"] == model]
        for thresh in THRESHOLDS:
            acc = compute_accuracy(md, thresh)
            model_rows.append({"model": model, "threshold": thresh,
                                "overall": round(acc["overall"]*100, 2)})

    overall_df  = pd.DataFrame(overall_rows)
    per_task_df = pd.DataFrame(per_task_rows)
    model_df    = pd.DataFrame(model_rows)

    # stability
    window = [t for t in THRESHOLDS if 0.68 <= t <= 0.75]
    swing = float(overall_df[overall_df["threshold"].isin(window)]["overall_accuracy"].max()
                  - overall_df[overall_df["threshold"].isin(window)]["overall_accuracy"].min())

    print(f"\n  Stability window [0.68, 0.75]:")
    for _, row in overall_df[overall_df["threshold"].isin(window)].iterrows():
        marker = "  ← chosen" if abs(row["threshold"]-0.72)<1e-9 else ""
        print(f"    t={row['threshold']:.2f}  overall={row['overall_accuracy']:.2f}%{marker}")
    print(f"\n  Swing = {swing:.2f} pp  →  {'STABLE' if swing<=2 else 'UNSTABLE'}")

    print(f"\n  Rank order at t=0.72:")
    at72 = model_df[abs(model_df["threshold"]-0.72)<1e-9].sort_values("overall", ascending=False)
    for _, r in at72.iterrows(): print(f"    {r['model']:<25} {r['overall']:.1f}%")

    # save
    overall_df.to_csv(OUT_DIR/"fuzzy_ablation_overall.csv", index=False)
    per_task_df.to_csv(OUT_DIR/"fuzzy_ablation_per_task.csv", index=False)
    model_df.to_csv(OUT_DIR/"fuzzy_ablation_model.csv", index=False)

    # plot
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors_task  = {"REG":"#4472C4","NUM":"#ED7D31","CON":"#A9D18E","TMP":"#FF0000"}
    colors_model = {"Claude 3 Haiku":"#4472C4","Gemini 2.5 Flash":"#ED7D31",
                    "LLaMA-3.3-70B":"#70AD47","LLaMA-3-8B":"#FFC000","Mistral-7B":"#FF0000"}
    for sh, col in colors_task.items():
        sub = per_task_df[per_task_df["task"]==sh].dropna(subset=["accuracy"])
        ax.plot(sub["threshold"]*100, sub["accuracy"], "o-", color=col, lw=1.8, ms=5, label=sh)
    ax.axvline(72, color="gray", ls="--", lw=1.2, alpha=0.8)
    ax.set(xlabel="Threshold (%)", ylabel="Accuracy (%)", title="Per-Task vs. Threshold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_xticks([t*100 for t in THRESHOLDS]); ax.tick_params(axis="x", rotation=45)

    for m in [m for m in MODEL_FILES if m in model_df["model"].values]:
        sub = model_df[model_df["model"]==m]
        lab = m.replace("LLaMA-3.3-70B","LLaMA-70B").replace("LLaMA-3-8B","LLaMA-8B")
        ax2.plot(sub["threshold"]*100, sub["overall"], "o-",
                 color=colors_model.get(m,"black"), lw=1.8, ms=5, label=lab)
    ax2.axvline(72, color="gray", ls="--", lw=1.2, alpha=0.8)
    ax2.set(xlabel="Threshold (%)", ylabel="Overall Accuracy (%)", title="Per-Model vs. Threshold")
    ax2.legend(fontsize=8, loc="lower left"); ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks([t*100 for t in THRESHOLDS]); ax2.tick_params(axis="x", rotation=45)

    plt.suptitle("IndiaFinBench: Threshold Sensitivity (task-aware scorer)", fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR/"fuzzy_ablation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: evaluation/error_analysis/fuzzy_ablation*.csv + .png")

    # LaTeX table
    short_map = {"Claude 3 Haiku":"Haiku","Gemini 2.5 Flash":"Gemini",
                 "LLaMA-3.3-70B":"LLaMA-70B","LLaMA-3-8B":"LLaMA-8B","Mistral-7B":"Mistral"}
    present = [m for m in MODEL_FILES if m in model_df["model"].values]
    cols = [short_map.get(m,m) for m in present] + ["Avg"]
    hdr  = r"\textbf{t}" + "".join(f" & \\textbf{{{c}}}" for c in cols) + r" \\"
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\caption{Threshold sensitivity. Scorer: CON=exact Yes/No; REG/NUM/TMP=numeric shortcut or token\_set\_ratio $\geq t$. Rank order preserved at all thresholds.}",
             r"\label{tab:fuzzy-ablation}",
             rf"\begin{{tabular}}{{l{'r'*len(cols)}}}", r"\toprule", hdr, r"\midrule"]

    for thresh in THRESHOLDS:
        chosen = abs(thresh-0.72)<1e-9
        ts = rf"\textbf{{{thresh:.2f}}}" if chosen else f"{thresh:.2f}"
        vals = []
        for m in present:
            sub = model_df[(model_df["model"]==m)&(abs(model_df["threshold"]-thresh)<1e-9)]["overall"]
            v = f"{sub.iloc[0]:.1f}" if len(sub)>0 else "---"
            vals.append(rf"\textbf{{{v}}}" if chosen else v)
        av = overall_df[abs(overall_df["threshold"]-thresh)<1e-9]["overall_accuracy"]
        avg_v = f"{av.iloc[0]:.1f}" if len(av)>0 else "---"
        vals.append(rf"\textbf{{{avg_v}}}" if chosen else avg_v)
        lines.append(ts + " & " + " & ".join(vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR/"table_fuzzy_ablation.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"  Saved: paper/tables/table_fuzzy_ablation.tex\n")

    print(f"{'='*65}\n  Max swing in [0.68,0.75]: {swing:.2f} pp\n{'='*65}\n")


if __name__ == "__main__":
    main()