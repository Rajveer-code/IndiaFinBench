// IndiaFinBench Data Module

// ── Inline SVG logo data-URIs ─────────────────────────────────────────────────

// Google "G" four-colour logo
const _G = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%234285F4' d='M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.4-1.04 2.58-2.22 3.38v2.81h3.59c2.1-1.93 3.27-4.77 3.27-8.2z'/%3E%3Cpath fill='%2334A853' d='M12 23c2.97 0 5.46-1 7.28-2.69l-3.59-2.81c-.98.66-2.23 1.06-3.69 1.06-2.84 0-5.25-1.92-6.11-4.5H2.18v2.9C3.99 20.53 7.7 23 12 23z'/%3E%3Cpath fill='%23FBBC04' d='M5.89 14.06c-.22-.66-.35-1.36-.35-2.06s.13-1.4.35-2.06V7.04H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.96l3.71-2.9z'/%3E%3Cpath fill='%23EA4335' d='M12 5.5c1.6 0 3.04.55 4.17 1.63l3.13-3.13C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.04l3.71 2.9C6.75 7.42 9.16 5.5 12 5.5z'/%3E%3C/svg%3E`;

// Meta infinity-loop mark (blue gradient) — used for all LLaMA variants
const _META = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 20'%3E%3Cdefs%3E%3ClinearGradient id='mg' x1='0' y1='0' x2='1' y2='0'%3E%3Cstop offset='0' stop-color='%230082FB'/%3E%3Cstop offset='1' stop-color='%2300C6FF'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cellipse cx='14' cy='10' rx='6' ry='8' fill='none' stroke='url(%23mg)' stroke-width='2.8'/%3E%3Cellipse cx='26' cy='10' rx='6' ry='8' fill='none' stroke='url(%23mg)' stroke-width='2.8'/%3E%3Cpath d='M8 10h24' stroke='url(%23mg)' stroke-width='2.8'/%3E%3C/svg%3E`;

// OpenAI spinning-swirl logo
const _OPENAI = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%23000' d='M22.282 9.821a5.985 5.985 0 0 0-.516-4.91 6.046 6.046 0 0 0-6.51-2.9A6.065 6.065 0 0 0 4.981 4.18a5.985 5.985 0 0 0-3.998 2.9 6.046 6.046 0 0 0 .743 7.097 5.98 5.98 0 0 0 .51 4.911 6.051 6.051 0 0 0 6.515 2.9A5.985 5.985 0 0 0 13.26 24a6.056 6.056 0 0 0 5.772-4.206 5.99 5.99 0 0 0 3.997-2.9 6.056 6.056 0 0 0-.747-7.073zM13.26 22.43a4.476 4.476 0 0 1-2.876-1.04l.141-.081 4.779-2.758a.795.795 0 0 0 .392-.681v-6.737l2.02 1.168a.071.071 0 0 1 .038.052v5.583a4.504 4.504 0 0 1-4.494 4.494zM3.6 18.304a4.47 4.47 0 0 1-.535-3.014l.142.085 4.783 2.759a.771.771 0 0 0 .78 0l5.843-3.369v2.332a.08.08 0 0 1-.033.062L9.74 19.95a4.5 4.5 0 0 1-6.14-1.646zM2.34 7.896a4.485 4.485 0 0 1 2.366-1.973V11.6a.766.766 0 0 0 .388.676l5.815 3.355-2.02 1.168a.076.076 0 0 1-.071 0l-4.83-2.786A4.504 4.504 0 0 1 2.34 7.896zm16.597 3.855l-5.833-3.387 2.019-1.168a.076.076 0 0 1 .071 0l4.83 2.791a4.494 4.494 0 0 1-.676 8.105v-5.678a.79.79 0 0 0-.411-.663zm2.01-3.023l-.141-.085-4.774-2.782a.776.776 0 0 0-.785 0L9.409 9.23V6.897a.066.066 0 0 1 .028-.061l4.83-2.787a4.5 4.5 0 0 1 6.68 4.66zm-12.64 4.135l-2.02-1.164a.08.08 0 0 1-.038-.057V6.075a4.5 4.5 0 0 1 7.375-3.453l-.142.08L8.704 5.46a.795.795 0 0 0-.393.681zm1.097-2.365l2.602-1.5 2.607 1.5v2.999l-2.597 1.5-2.607-1.5z'/%3E%3C/svg%3E`;

// Mistral AI — orange grid pattern (matches their brand tiles)
const _MISTRAL = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect width='24' height='24' rx='4' fill='%23FF7000'/%3E%3Crect x='4' y='4' width='4' height='4' fill='%23000'/%3E%3Crect x='10' y='4' width='4' height='4' fill='%23000'/%3E%3Crect x='16' y='4' width='4' height='4' fill='%23000'/%3E%3Crect x='4' y='10' width='4' height='4' fill='%23000'/%3E%3Crect x='16' y='10' width='4' height='4' fill='%23000'/%3E%3Crect x='4' y='16' width='4' height='4' fill='%23000'/%3E%3Crect x='10' y='16' width='4' height='4' fill='%23000'/%3E%3Crect x='16' y='16' width='4' height='4' fill='%23000'/%3E%3C/svg%3E`;

// Alibaba / Qwen — purple circle with wordmark
const _QWEN = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E%3Ccircle cx='20' cy='20' r='19' fill='%237C3AED'/%3E%3Ctext x='20' y='25' text-anchor='middle' font-family='Arial,sans-serif' font-size='12' font-weight='900' fill='white'%3EQwen%3C/text%3E%3C/svg%3E`;

// DeepSeek — blue circle with stylised deep-sea swirl
const _DEEPSEEK = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E%3Ccircle cx='20' cy='20' r='19' fill='%231A6EFF'/%3E%3Ccircle cx='20' cy='20' r='8' fill='none' stroke='white' stroke-width='2'/%3E%3Ccircle cx='20' cy='20' r='3' fill='white'/%3E%3Cpath d='M20 8 Q28 14 28 20 Q28 28 20 32 Q12 28 12 20 Q12 14 20 8Z' fill='none' stroke='rgba(255,255,255,0.4)' stroke-width='1.5'/%3E%3C/svg%3E`;

// Moonshot AI / Kimi — crescent moon on deep-navy
const _KIMI = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E%3Ccircle cx='20' cy='20' r='19' fill='%23060917'/%3E%3Cpath fill='white' d='M24 10a12 12 0 0 1 0 20A10 10 0 0 0 24 10z'/%3E%3Ccircle cx='26' cy='12' r='2' fill='%23FFD700'/%3E%3C/svg%3E`;

// ── Model data ─────────────────────────────────────────────────────────────────

window.IFB_MODELS = [
  {rank:1, label:"Gemini 2.5 Flash",          hf_id:"google/gemini-2.5-flash",                         params:"—",    type:"Frontier API",     overall:89.7, reg:93.1, num:84.8, con:88.7, tmp:88.5, ci:"[86.3%, 92.3%]", n_items:406, tier:1, color:"#4285F4", icon:"G",  logo:_G},
  {rank:2, label:"Qwen3-32B",                  hf_id:"Qwen/Qwen3-32B",                                  params:"32B",  type:"Open-weight API",  overall:85.5, reg:85.1, num:77.2, con:90.3, tmp:92.3, ci:"[81.7%, 88.6%]", n_items:406, tier:1, color:"#7C3AED", icon:"Q",  logo:_QWEN},
  {rank:3, label:"LLaMA-3.3-70B",              hf_id:"meta-llama/Llama-3.3-70B-Versatile",              params:"70B",  type:"Open-weight API",  overall:83.7, reg:86.2, num:75.0, con:95.2, tmp:79.5, ci:"[79.8%, 87.0%]", n_items:406, tier:1, color:"#0EA5E9", icon:"L3", logo:_META},
  {rank:4, label:"Llama 4 Scout 17B",          hf_id:"meta-llama/Llama-4-Scout-17B",                    params:"17B",  type:"Open-weight API",  overall:83.3, reg:86.2, num:66.3, con:98.4, tmp:84.6, ci:"[79.3%, 86.6%]", n_items:406, tier:1, color:"#EC4899", icon:"L4", logo:_META},
  {rank:5, label:"Kimi K2",                    hf_id:"moonshotai/Kimi-K2",                              params:"1T (MoE, 32B active)", type:"Frontier API", overall:81.5, reg:89.1, num:65.2, con:91.9, tmp:75.6, ci:"[77.5%, 85.0%]", n_items:406, tier:1, color:"#14B8A6", icon:"K",  logo:_KIMI},
  {rank:6, label:"LLaMA-3-8B",                 hf_id:"meta-llama/Meta-Llama-3-8B-Instruct",             params:"8B",   type:"Local (Ollama)",   overall:78.1, reg:79.9, num:64.1, con:93.5, tmp:78.2, ci:"[73.8%, 81.8%]", n_items:406, tier:2, color:"#8B5CF6", icon:"L",  logo:_META},
  {rank:7, label:"GPT-OSS 120B",               hf_id:"openai/gpt-oss-120b",                             params:"120B", type:"Open-weight API",  overall:77.1, reg:79.9, num:59.8, con:95.2, tmp:76.9, ci:"[72.8%, 80.9%]", n_items:406, tier:2, color:"#10B981", icon:"GP", logo:_OPENAI},
  {rank:8, label:"GPT-OSS 20B",                hf_id:"openai/gpt-oss-20b",                              params:"20B",  type:"Open-weight API",  overall:76.8, reg:79.9, num:58.7, con:95.2, tmp:76.9, ci:"[72.5%, 80.7%]", n_items:406, tier:2, color:"#059669", icon:"GP", logo:_OPENAI},
  {rank:9, label:"Gemini 2.5 Pro",             hf_id:"google/gemini-2.5-pro",                           params:"—",    type:"Frontier API",     overall:76.1, reg:89.7, num:48.9, con:93.5, tmp:64.1, ci:"[71.7%, 80.0%]", n_items:406, tier:2, color:"#34A853", icon:"G",  logo:_G},
  {rank:10,label:"Mistral-7B",                 hf_id:"mistralai/Mistral-7B-Instruct-v0.3",              params:"7B",   type:"Local (Ollama)",   overall:75.9, reg:79.9, num:66.3, con:80.6, tmp:74.4, ci:"[71.5%, 79.8%]", n_items:406, tier:2, color:"#F59E0B", icon:"M",  logo:_MISTRAL},
  {rank:11,label:"DeepSeek R1 70B",            hf_id:"deepseek-ai/DeepSeek-R1-Distill-Llama-70B",       params:"70B",  type:"Reasoning API",    overall:75.1, reg:72.4, num:69.6, con:96.8, tmp:70.5, ci:"[70.7%, 79.1%]", n_items:406, tier:2, color:"#EF4444", icon:"DS", logo:_DEEPSEEK},
  {rank:12,label:"Gemma 4 E4B",                hf_id:"google/gemma-4-e4b",                              params:"4B",   type:"Local (Ollama)",   overall:70.4, reg:83.9, num:50.0, con:72.6, tmp:62.8, ci:"[65.8%, 74.7%]", n_items:406, tier:3, color:"#06B6D4", icon:"Ge", logo:_G},
];

window.IFB_HUMAN = {
  rank:"—", label:"Human Expert", hf_id:"— (n=100 sampled items)",
  params:"—", type:"Human Baseline",
  overall:69.0, reg:55.6, num:44.4, con:83.3, tmp:66.7,
  ci:"[59.4%, 77.2%]", n_items:100, tier:0, is_human:true,
  color:"#94A3B8", icon:"H", logo:null
};

window.IFB_CLAUDE = {
  rank:"†", label:"†Claude 3 Haiku", hf_id:"anthropic/claude-3-haiku (150-item subset)",
  params:"—", type:"Frontier API†",
  overall:91.3, reg:92.5, num:93.8, con:86.7, tmp:91.4,
  ci:"[85.7%, 94.9%]", n_items:150, is_subset:true,
  color:"#C17B42", icon:"C", logo:null
};

window.IFB_DIFF = [
  {label:"Gemini 2.5 Flash",    easy:92.5, med:89.0, hard:84.4, color:"#4285F4"},
  {label:"Qwen3-32B",            easy:81.9, med:87.9, hard:87.5, color:"#7C3AED"},
  {label:"LLaMA-3.3-70B",        easy:79.4, med:85.2, hard:90.6, color:"#0EA5E9"},
  {label:"Llama 4 Scout 17B",    easy:82.5, med:81.9, hard:89.1, color:"#EC4899"},
  {label:"Kimi K2",              easy:81.9, med:80.8, hard:82.8, color:"#14B8A6"},
  {label:"LLaMA-3-8B",           easy:76.2, med:79.7, hard:78.1, color:"#8B5CF6"},
  {label:"GPT-OSS 120B",         easy:79.4, med:76.4, hard:73.4, color:"#10B981"},
  {label:"GPT-OSS 20B",          easy:75.0, med:79.7, hard:73.4, color:"#059669"},
  {label:"Gemini 2.5 Pro",       easy:83.1, med:72.5, hard:68.8, color:"#34A853"},
  {label:"Mistral-7B",           easy:74.4, med:76.9, hard:76.6, color:"#F59E0B"},
  {label:"DeepSeek R1 70B",      easy:72.5, med:77.5, hard:75.0, color:"#EF4444"},
  {label:"Gemma 4 E4B",          easy:82.5, med:64.8, hard:56.2, color:"#06B6D4"},
];

window.IFB_TASKS = [
  {code:"REG", key:"reg", label:"Regulatory Interpretation", n:174,
   desc:"Extract compliance rules, thresholds and deadlines from SEBI and RBI regulatory text.",
   color:"#4285F4"},
  {code:"NUM", key:"num", label:"Numerical Reasoning", n:92,
   desc:"Arithmetic over capital ratios, dividend limits, and margin requirements in regulatory prose.",
   color:"#F59E0B"},
  {code:"CON", key:"con", label:"Contradiction Detection", n:62,
   desc:"Identify whether two regulatory passages contradict each other on a stated issue.",
   color:"#10B981"},
  {code:"TMP", key:"tmp", label:"Temporal Reasoning", n:78,
   desc:"Sequence regulatory amendments and identify which circular was operative at a given time.",
   color:"#8B5CF6"},
];
