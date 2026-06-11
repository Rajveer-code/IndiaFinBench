/* ════════════════════════════════════════════════════════════════════
   IndiaFinBench — data.js
   Canonical frontend leaderboard data. Mirrors results/aggregate/
   all_model_results.csv (Table 6) — do not edit numbers by hand.
   ════════════════════════════════════════════════════════════════════ */

window.IFB_MODELS = [
  { rank: 1,  label: "Gemini 2.5 Flash",  hf_id: "google/gemini-2.5-flash",                   params: "—",                   type: "Frontier API",    overall: 89.7, reg: 93.1, num: 84.8, con: 88.7, tmp: 88.5, ci: "[86.3%, 92.3%]", n_items: 406, tier: 1 },
  { rank: 2,  label: "Qwen3-32B",         hf_id: "Qwen/Qwen3-32B",                            params: "32B",                 type: "Open-weight API", overall: 85.5, reg: 85.1, num: 77.2, con: 90.3, tmp: 92.3, ci: "[81.7%, 88.6%]", n_items: 406, tier: 1 },
  { rank: 3,  label: "LLaMA-3.3-70B",     hf_id: "meta-llama/Llama-3.3-70B-Versatile",        params: "70B",                 type: "Open-weight API", overall: 83.7, reg: 86.2, num: 75.0, con: 95.2, tmp: 79.5, ci: "[79.8%, 87.0%]", n_items: 406, tier: 1 },
  { rank: 4,  label: "Llama 4 Scout 17B", hf_id: "meta-llama/Llama-4-Scout-17B",              params: "17B",                 type: "Open-weight API", overall: 83.3, reg: 86.2, num: 66.3, con: 98.4, tmp: 84.6, ci: "[79.3%, 86.6%]", n_items: 406, tier: 1 },
  { rank: 5,  label: "Kimi K2",           hf_id: "moonshotai/Kimi-K2",                        params: "1T (MoE, 32B active)", type: "Frontier API",   overall: 81.5, reg: 89.1, num: 65.2, con: 91.9, tmp: 75.6, ci: "[77.5%, 85.0%]", n_items: 406, tier: 1 },
  { rank: 6,  label: "LLaMA-3-8B",        hf_id: "meta-llama/Meta-Llama-3-8B-Instruct",       params: "8B",                  type: "Local (Ollama)",  overall: 78.1, reg: 79.9, num: 64.1, con: 93.5, tmp: 78.2, ci: "[73.8%, 81.8%]", n_items: 406, tier: 2 },
  { rank: 7,  label: "GPT-OSS 120B",      hf_id: "openai/gpt-oss-120b",                       params: "120B",                type: "Open-weight API", overall: 77.1, reg: 79.9, num: 59.8, con: 95.2, tmp: 76.9, ci: "[72.8%, 80.9%]", n_items: 406, tier: 2 },
  { rank: 8,  label: "GPT-OSS 20B",       hf_id: "openai/gpt-oss-20b",                        params: "20B",                 type: "Open-weight API", overall: 76.8, reg: 79.9, num: 58.7, con: 95.2, tmp: 76.9, ci: "[72.5%, 80.7%]", n_items: 406, tier: 2 },
  { rank: 9,  label: "Gemini 2.5 Pro",    hf_id: "google/gemini-2.5-pro",                     params: "—",                   type: "Frontier API",    overall: 76.1, reg: 89.7, num: 48.9, con: 93.5, tmp: 64.1, ci: "[71.7%, 80.0%]", n_items: 406, tier: 2 },
  { rank: 10, label: "Mistral-7B",        hf_id: "mistralai/Mistral-7B-Instruct-v0.3",        params: "7B",                  type: "Local (Ollama)",  overall: 75.9, reg: 79.9, num: 66.3, con: 80.6, tmp: 74.4, ci: "[71.5%, 79.8%]", n_items: 406, tier: 2 },
  { rank: 11, label: "DeepSeek R1 70B",   hf_id: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", params: "70B",                 type: "Reasoning API",   overall: 75.1, reg: 72.4, num: 69.6, con: 96.8, tmp: 70.5, ci: "[70.7%, 79.1%]", n_items: 406, tier: 2 },
  { rank: 12, label: "Gemma 4 E4B",       hf_id: "google/gemma-4-e4b",                        params: "4B",                  type: "Local (Ollama)",  overall: 70.4, reg: 83.9, num: 50.0, con: 72.6, tmp: 62.8, ci: "[65.8%, 74.7%]", n_items: 406, tier: 3 }
];

window.IFB_HUMAN = {
  rank: "—", label: "Human Expert", hf_id: "— (n=100 sampled items)",
  params: "—", type: "Human Baseline",
  overall: 69.0, reg: 55.6, num: 44.4, con: 83.3, tmp: 66.7,
  ci: "[59.4%, 77.2%]", n_items: 100, tier: 0, is_human: true
};

window.IFB_CLAUDE = {
  rank: "†", label: "†Claude 3 Haiku", hf_id: "anthropic/claude-3-haiku (150-item subset)",
  params: "—", type: "Frontier API†",
  overall: 91.3, reg: 92.5, num: 93.8, con: 86.7, tmp: 91.4,
  ci: "[85.7%, 94.9%]", n_items: 150, is_subset: true
};

window.IFB_DIFF = [
  { label: "Gemini 2.5 Flash",  easy: 92.5, med: 89.0, hard: 84.4 },
  { label: "Qwen3-32B",         easy: 81.9, med: 87.9, hard: 87.5 },
  { label: "LLaMA-3.3-70B",     easy: 79.4, med: 85.2, hard: 90.6 },
  { label: "Llama 4 Scout 17B", easy: 82.5, med: 81.9, hard: 89.1 },
  { label: "Kimi K2",           easy: 81.9, med: 80.8, hard: 82.8 },
  { label: "LLaMA-3-8B",        easy: 76.2, med: 79.7, hard: 78.1 },
  { label: "GPT-OSS 120B",      easy: 79.4, med: 76.4, hard: 73.4 },
  { label: "GPT-OSS 20B",       easy: 75.0, med: 79.7, hard: 73.4 },
  { label: "Gemini 2.5 Pro",    easy: 83.1, med: 72.5, hard: 68.8 },
  { label: "Mistral-7B",        easy: 74.4, med: 76.9, hard: 76.6 },
  { label: "DeepSeek R1 70B",   easy: 72.5, med: 77.5, hard: 75.0 },
  { label: "Gemma 4 E4B",       easy: 82.5, med: 64.8, hard: 56.2 }
];

window.IFB_TASKS = [
  { code: "REG", key: "reg", label: "Regulatory Interpretation", n: 174, color: "#1F5C45",
    desc: "Extract compliance rules, thresholds and deadlines from SEBI and RBI regulatory text." },
  { code: "NUM", key: "num", label: "Numerical Reasoning", n: 92, color: "#A33B20",
    desc: "Arithmetic over capital ratios, dividend limits, and margin requirements in regulatory prose." },
  { code: "CON", key: "con", label: "Contradiction Detection", n: 62, color: "#96752A",
    desc: "Identify whether two regulatory passages contradict each other on a stated issue." },
  { code: "TMP", key: "tmp", label: "Temporal Reasoning", n: 78, color: "#3B3F8C",
    desc: "Sequence regulatory amendments and identify which circular was operative at a given time." }
];
