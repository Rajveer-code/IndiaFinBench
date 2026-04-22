"""Second patch: fix Unicode-encoded special characters in docx XML."""
path = r"D:\Projects\IndiaFinBench\paper\docx_unpacked\word\document.xml"
with open(path, encoding="utf-8") as f:
    content = f.read()
original = len(content)

def rep(old, new, label):
    global content
    if old not in content:
        print(f"  WARN: {label}")
        return
    content = content.replace(old, new, 1)
    print(f"  OK: {label}")

# Remove Haiku model table row (literal dagger U+2020)
haiku_m = "\u2020Claude 3 Haiku</w:t>"
idx = content.find(haiku_m)
while idx >= 0:
    tr_start = content.rfind("<w:tr>", 0, idx)
    tr_end = content.find("</w:tr>", idx) + len("</w:tr>")
    content = content[:tr_start] + content[tr_end:]
    print("  OK: Removed a Haiku table row")
    idx = content.find(haiku_m)

# Table 5 caption
rep(
    "Table 5. Models evaluated in this study. \u2020Claude 3 Haiku was evaluated on the initial 150-item subset only (see Section 4.1).",
    "Table 5. Models evaluated in this study.",
    "Table 5 caption"
)

# Table 6 caption
rep(
    "Table 6. IndiaFinBench results \u2014 accuracy (%) by task type. All eleven models evaluated on the full 406-item benchmark. \u2020Claude 3 Haiku evaluated on the initial 150-item subset (REG=53, NUM=32, CON=30, TMP=35); not directly comparable. 95% Wilson score confidence intervals.",
    "Table 6. IndiaFinBench results \u2014 accuracy (%) by task type. All twelve models evaluated on the full 406-item benchmark. \u2020Gemini 2.5 Pro evaluated via Vertex AI; lower NUM/TMP scores reflect verbose output style penalised by reference-matching scoring. 95% Wilson score confidence intervals.",
    "Table 6 caption"
)

# Tier 2 description
rep(
    " \u2014 middle performers (75\u201379%): LLaMA-3-8B, GPT-OSS 120B, GPT-OSS 20B, Mistral-7B, and DeepSeek R1 70B. Notably, GPT-OSS 120B (77.1%) and GPT-OSS 20B (76.8%) are statistically indistinguishable (p = 0.91), suggesting that a six-fold increase in parameter count provides no measurable benefit on this task. Similarly, LLaMA-3-8B and Mistral-7B are statistically tied (p = 0.38).",
    " \u2014 middle performers (75\u201379%): LLaMA-3-8B, GPT-OSS 120B, GPT-OSS 20B, Gemini 2.5 Pro, Mistral-7B, and DeepSeek R1 70B. Notably, GPT-OSS 120B (77.1%) and GPT-OSS 20B (76.8%) are statistically indistinguishable (p = 0.91). Gemini 2.5 Pro (76.1%) falls in this tier despite being a frontier model, an artefact of its verbose output style under reference-matching scoring. LLaMA-3-8B and Mistral-7B are statistically tied (p = 0.38).",
    "Tier 2: add Gemini 2.5 Pro"
)

# NUM analysis
rep(
    "Numerical Reasoning (NUM) is the most discriminative task, with a 34.8 percentage-point spread (Gemini: 84.8% vs Gemma 4 E4B: 50.0%). Gemma 4 E4B\u2019s 50% score is at or near chance level for binary classification, indicating near-complete failure. The GPT-OSS models also underperform on NUM (59.8% and 58.7%), suggesting this model family struggles with the multi-step arithmetic embedded in Indian regulatory text.",
    "Numerical Reasoning (NUM) is the most discriminative task, with a 35.9 percentage-point spread (Gemini 2.5 Flash: 84.8% vs Gemini 2.5 Pro: 48.9%). As discussed, Gemini 2.5 Pro\u2019s low NUM score reflects a scoring artefact of its verbose output style. Among non-reasoning models, Gemma 4 E4B (50.0%) is at near-chance level. The GPT-OSS models also underperform on NUM (59.8% and 58.7%), suggesting this family struggles with multi-step arithmetic over Indian regulatory text.",
    "NUM analysis 34.8->35.9"
)

# Conclusion
rep(
    "Evaluating eleven contemporary models on the full benchmark reveals a clear tier structure: a top group clustering around 81\u201390% overall accuracy, a middle group around 75\u201379%, and one clear underperformer at 70%. Paired bootstrap significance testing establishes which differences are statistically robust.",
    "Evaluating twelve contemporary models on the full benchmark reveals a clear tier structure: a top group clustering around 81\u201390% overall accuracy, a middle group around 75\u201379%, and one clear underperformer at 70%. Paired bootstrap significance testing establishes which differences are statistically robust.",
    "Conclusion eleven->twelve"
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"\nDone. {original} -> {len(content)} chars (delta={len(content)-original:+d})")
