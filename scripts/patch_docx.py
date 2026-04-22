"""Patch all docx XML edits for the IndiaFinBench paper v11 → v12."""
import re

path = r"D:\Projects\IndiaFinBench\paper\docx_unpacked\word\document.xml"
with open(path, encoding="utf-8") as f:
    content = f.read()
original_len = len(content)

def replace_one(old, new, label):
    global content
    if old not in content:
        print(f"  WARN: not found — {label}")
        return
    content = content.replace(old, new, 1)
    print(f"  OK: {label}")

# ══════════════════════════════════════════════════════════════════
# 1. SIMPLE TEXT REPLACEMENTS
# ══════════════════════════════════════════════════════════════════

replace_one(
    "We evaluate eleven models under zero-shot conditions on the full benchmark, with accuracy ranging from 70.4% (Gemma 4 E4B) to 89.7% (Gemini 2.5 Flash). All models substantially outperform a human expert baseline of 60.0%. Numerical reasoning is the most discriminative task, with a 34.8 percentage-point spread across models.",
    "We evaluate twelve models under zero-shot conditions on the full benchmark, with accuracy ranging from 70.4% (Gemma 4 E4B) to 89.7% (Gemini 2.5 Flash). All models substantially outperform a human expert baseline of 60.0%. Numerical reasoning is the most discriminative task, with a 35.9 percentage-point spread across models.",
    "Abstract: eleven->twelve + 34.8->35.9"
)

replace_one(
    "A comprehensive zero-shot evaluation of eleven contemporary LLMs on the full 406-item benchmark, revealing three performance tiers and substantial inter-task variation.",
    "A comprehensive zero-shot evaluation of twelve contemporary LLMs on the full 406-item benchmark, revealing three performance tiers and substantial inter-task variation.",
    "Contributions: eleven->twelve"
)

replace_one(
    "We evaluate eleven models spanning a wide range of sizes, providers, and access modes on the full 406-item benchmark:",
    "We evaluate twelve models spanning a wide range of sizes, providers, and access modes on the full 406-item benchmark:",
    "Section 4.1: eleven->twelve"
)

replace_one(
    "<w:t>Google (API)</w:t>",
    "<w:t>Google (AI Studio API)</w:t>",
    "Gemini 2.5 Flash provider name"
)

replace_one(
    "Table 6 presents overall and per-task accuracy for all eleven models evaluated on the full 406-item benchmark, together with Wilson 95% confidence intervals for overall accuracy.",
    "Table 6 presents overall and per-task accuracy for all twelve models evaluated on the full 406-item benchmark, together with Wilson 95% confidence intervals for overall accuracy.",
    "Section 5.1 intro: eleven->twelve"
)

replace_one(
    "Figure 1. Performance heatmap across all eleven models and four task types. Darker cells indicate higher accuracy.",
    "Figure 1. Performance heatmap across all twelve models and four task types. Darker cells indicate higher accuracy.",
    "Figure 1 caption"
)

replace_one(
    "All eleven models substantially outperform the human expert baseline of 60.0% (n = 30 items). The human baseline reflects non-expert annotators under time constraints, and is provided as a lower-bound reference for task difficulty.",
    "All twelve models substantially outperform the human expert baseline of 60.0% (n = 30 items). A notable finding is that Gemini 2.5 Pro (76.1%) falls below Gemini 2.5 Flash (89.7%) despite being the larger model, driven by its verbose reasoning outputs being penalised under concise reference-matching scoring, particularly on NUM (48.9%) and TMP (64.1%). The human baseline reflects non-expert annotators under time constraints, and is provided as a lower-bound reference for task difficulty.",
    "Section 5.1 narrative: eleven->twelve + Gemini Pro note"
)

replace_one(
    "Paired bootstrap significance testing (10,000 resamples) across all 55 model pairs reveals clear tier structure: 35 of 55 pairs are statistically significantly different at p &lt; 0.05, while 20 pairs are not.",
    "Paired bootstrap significance testing (10,000 resamples) across all 66 model pairs reveals clear tier structure, with the majority of cross-tier pairs statistically significantly different at p &lt; 0.05.",
    "Section 5.2: 55->66 pairs"
)

replace_one(
    " &#x2014; middle performers (75&#x2013;79%): LLaMA-3-8B, GPT-OSS 120B, GPT-OSS 20B, Mistral-7B, and DeepSeek R1 70B. Notably, GPT-OSS 120B (77.1%) and GPT-OSS 20B (76.8%) are statistically indistinguishable (p = 0.91), suggesting that a six-fold increase in parameter count provides no measurable benefit on this task. Similarly, LLaMA-3-8B and Mistral-7B are statistically tied (p = 0.38).",
    " &#x2014; middle performers (75&#x2013;79%): LLaMA-3-8B, GPT-OSS 120B, GPT-OSS 20B, Gemini 2.5 Pro, Mistral-7B, and DeepSeek R1 70B. Notably, GPT-OSS 120B (77.1%) and GPT-OSS 20B (76.8%) are statistically indistinguishable (p = 0.91), suggesting that a six-fold increase in parameter count provides no measurable benefit on this task. Gemini 2.5 Pro (76.1%) falls in this tier despite being a frontier model, an artefact of its verbose output style under reference-matching scoring. LLaMA-3-8B and Mistral-7B are statistically tied (p = 0.38).",
    "Tier 2: add Gemini 2.5 Pro"
)

replace_one(
    "Numerical Reasoning (NUM) is the most discriminative task, with a 34.8 percentage-point spread (Gemini: 84.8% vs Gemma 4 E4B: 50.0%). Gemma 4 E4B&#x2019;s 50% score is at or near chance level for binary classification, indicating near-complete failure. The GPT-OSS models also underperform on NUM (59.8% and 58.7%), suggesting this model family struggles with the multi-step arithmetic embedded in Indian regulatory text.",
    "Numerical Reasoning (NUM) is the most discriminative task, with a 35.9 percentage-point spread (Gemini 2.5 Flash: 84.8% vs Gemini 2.5 Pro: 48.9%). As discussed, Gemini 2.5 Pro&#x2019;s low NUM score reflects a scoring artefact of its verbose output style. Among non-reasoning models, Gemma 4 E4B (50.0%) is at or near chance level, indicating near-complete failure. The GPT-OSS models also underperform on NUM (59.8% and 58.7%), suggesting this family struggles with multi-step arithmetic over Indian regulatory text.",
    "NUM task analysis: 34.8->35.9"
)

replace_one(
    " highlights an important limitation of reasoning-specialised architectures: despite being purpose-built for complex reasoning, DeepSeek R1 70B ranks 10th out of 11 models.",
    " highlights an important limitation of reasoning-specialised architectures: despite being purpose-built for complex reasoning, DeepSeek R1 70B ranks 11th out of 12 models.",
    "DeepSeek R1 paradox rank"
)

replace_one(
    "All eleven models substantially outperform the human expert baseline of 60.0% (n = 30 items). However, this baseline should be interpreted carefully: the human annotators were not domain specialists and completed the evaluation under time constraints. The baseline primarily establishes that IndiaFinBench items are genuinely challenging.",
    "All twelve models substantially outperform the human expert baseline of 60.0% (n = 30 items). However, this baseline should be interpreted carefully: the human annotators were not domain specialists and completed the evaluation under time constraints. The baseline primarily establishes that IndiaFinBench items are genuinely challenging.",
    "Discussion all eleven->twelve"
)

replace_one(
    "This task is also the most discriminative for models (34.8 percentage-point spread), confirming that multi-step numerical inference over domain-specific text is a meaningful differentiator for both humans and LLMs.",
    "This task is also the most discriminative for models (35.9 percentage-point spread), confirming that multi-step numerical inference over domain-specific text is a meaningful differentiator for both humans and LLMs.",
    "Human vs model NUM spread 34.8->35.9"
)

replace_one(
    "Evaluating eleven contemporary models on the full benchmark reveals a clear tier structure: a top group clustering around 81&#x2013;90% overall accuracy, a middle group around 75&#x2013;79%, and one clear underperformer at 70%. Paired bootstrap significance testing establishes which differences are statistically robust.",
    "Evaluating twelve contemporary models on the full benchmark reveals a clear tier structure: a top group clustering around 81&#x2013;90% overall accuracy, a middle group around 75&#x2013;79%, and one clear underperformer at 70%. Paired bootstrap significance testing establishes which differences are statistically robust.",
    "Conclusion eleven->twelve"
)

# ══════════════════════════════════════════════════════════════════
# 2. MODEL TABLE — add Gemini 2.5 Pro row, remove Haiku row
# ══════════════════════════════════════════════════════════════════

GEMINI_PRO_MODEL_ROW = """
      <w:tr>
        <w:tc>
          <w:tcPr>
            <w:tcW w:type="dxa" w:w="2880"/>
            <w:tcBorders>
              <w:top w:val="single" w:sz="4" w:color="000000"/>
              <w:left w:val="single" w:sz="4" w:color="000000"/>
              <w:bottom w:val="single" w:sz="4" w:color="000000"/>
              <w:right w:val="single" w:sz="4" w:color="000000"/>
              <w:insideH w:val="none"/>
              <w:insideV w:val="none"/>
            </w:tcBorders>
          </w:tcPr>
          <w:p>
            <w:r>
              <w:rPr>
                <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
                <w:sz w:val="22"/>
              </w:rPr>
              <w:t>Gemini 2.5 Pro</w:t>
            </w:r>
          </w:p>
        </w:tc>
        <w:tc>
          <w:tcPr>
            <w:tcW w:type="dxa" w:w="3600"/>
            <w:tcBorders>
              <w:top w:val="single" w:sz="4" w:color="000000"/>
              <w:left w:val="single" w:sz="4" w:color="000000"/>
              <w:bottom w:val="single" w:sz="4" w:color="000000"/>
              <w:right w:val="single" w:sz="4" w:color="000000"/>
              <w:insideH w:val="none"/>
              <w:insideV w:val="none"/>
            </w:tcBorders>
          </w:tcPr>
          <w:p>
            <w:r>
              <w:rPr>
                <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
                <w:sz w:val="22"/>
              </w:rPr>
              <w:t>Google (Vertex AI)</w:t>
            </w:r>
          </w:p>
        </w:tc>
        <w:tc>
          <w:tcPr>
            <w:tcW w:type="dxa" w:w="2160"/>
            <w:tcBorders>
              <w:top w:val="single" w:sz="4" w:color="000000"/>
              <w:left w:val="single" w:sz="4" w:color="000000"/>
              <w:bottom w:val="single" w:sz="4" w:color="000000"/>
              <w:right w:val="single" w:sz="4" w:color="000000"/>
              <w:insideH w:val="none"/>
              <w:insideV w:val="none"/>
            </w:tcBorders>
          </w:tcPr>
          <w:p>
            <w:r>
              <w:rPr>
                <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>
                <w:sz w:val="22"/>
              </w:rPr>
              <w:t>&#x2014;</w:t>
            </w:r>
          </w:p>
        </w:tc>
      </w:tr>"""

# Find after Gemini 2.5 Flash row (uniquely identified by its provider cell)
flash_anchor = "<w:t>Google (AI Studio API)</w:t>"
idx = content.find(flash_anchor)
if idx >= 0:
    tr_end = content.find("</w:tr>", idx) + len("</w:tr>")
    content = content[:tr_end] + GEMINI_PRO_MODEL_ROW + content[tr_end:]
    print("  OK: Insert Gemini 2.5 Pro row in model table")
else:
    print("  WARN: Gemini Flash anchor not found for model table insert")

# Remove Haiku row from model table
haiku_m = "&#x2020;Claude 3 Haiku</w:t>"
idx = content.find(haiku_m)
if idx >= 0:
    tr_start = content.rfind("<w:tr>", 0, idx)
    tr_end = content.find("</w:tr>", idx) + len("</w:tr>")
    content = content[:tr_start] + content[tr_end:]
    print("  OK: Remove Haiku from model table")
else:
    print("  WARN: Haiku model table row not found")

# Update Table 5 caption
replace_one(
    "Table 5. Models evaluated in this study. &#x2020;Claude 3 Haiku was evaluated on the initial 150-item subset only (see Section 4.1).",
    "Table 5. Models evaluated in this study.",
    "Table 5 caption"
)

# Remove Haiku paragraph in Section 4.1
haiku_para = "In addition, Claude 3 Haiku (Anthropic) was evaluated on the initial 150-item subset"
idx = content.find(haiku_para)
if idx >= 0:
    para_start = content.rfind("<w:p>", 0, idx)
    para_end = content.find("</w:p>", idx) + len("</w:p>")
    content = content[:para_start] + content[para_end:]
    print("  OK: Remove Haiku paragraph")
else:
    print("  WARN: Haiku paragraph not found")

# ══════════════════════════════════════════════════════════════════
# 3. RESULTS TABLE — insert Gemini 2.5 Pro row, update averages, remove Haiku
# ══════════════════════════════════════════════════════════════════

def make_cell(width, text, bold=False):
    b = "\n                <w:b/>" if bold else ""
    return (f"        <w:tc>\n"
            f"          <w:tcPr>\n"
            f"            <w:tcW w:type=\"dxa\" w:w=\"{width}\"/>\n"
            f"            <w:tcBorders>\n"
            f"              <w:top w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>\n"
            f"              <w:left w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>\n"
            f"              <w:bottom w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>\n"
            f"              <w:right w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>\n"
            f"              <w:insideH w:val=\"none\"/>\n"
            f"              <w:insideV w:val=\"none\"/>\n"
            f"            </w:tcBorders>\n"
            f"          </w:tcPr>\n"
            f"          <w:p>\n"
            f"            <w:r>\n"
            f"              <w:rPr>\n"
            f"                <w:rFonts w:ascii=\"Times New Roman\" w:hAnsi=\"Times New Roman\"/>{b}\n"
            f"                <w:sz w:val=\"20\"/>\n"
            f"              </w:rPr>\n"
            f"              <w:t>{text}</w:t>\n"
            f"            </w:r>\n"
            f"          </w:p>\n"
            f"        </w:tc>\n")

GEMINI_PRO_RESULTS_ROW = (
    "      <w:tr>\n" +
    make_cell(2160, "Gemini 2.5 Pro&#x2020;") +
    make_cell(720, "89.7") +
    make_cell(720, "48.9") +
    make_cell(720, "93.5") +
    make_cell(720, "64.1") +
    make_cell(1080, "76.1") +
    make_cell(1800, "[71.7%, 80.0%]") +
    "      </w:tr>"
)

# Insert after GPT-OSS 20B row (unique: "[72.5%, 80.7%]" CI)
gpt20b_ci = "[72.5%, 80.7%]</w:t>"
idx = content.find(gpt20b_ci)
if idx >= 0:
    tr_end = content.find("</w:tr>", idx) + len("</w:tr>")
    content = content[:tr_end] + "\n" + GEMINI_PRO_RESULTS_ROW + content[tr_end:]
    print("  OK: Insert Gemini 2.5 Pro row in results table")
else:
    print("  WARN: GPT-OSS 20B CI anchor not found")

# Update Average row values
avg_anchor = "<w:t>Average</w:t>"
pos = content.find(avg_anchor)
if pos < 0:
    print("  WARN: Average row not found")
else:
    # REG average
    i = content.find("<w:t>83.2</w:t>", pos)
    if i >= 0:
        content = content[:i] + "<w:t>83.8</w:t>" + content[i + len("<w:t>83.2</w:t>"):]
        print("  OK: Avg REG 83.2->83.8")
    # NUM average
    i = content.find("<w:t>69.7</w:t>", pos)
    if i >= 0:
        content = content[:i] + "<w:t>65.5</w:t>" + content[i + len("<w:t>69.7</w:t>"):]
        print("  OK: Avg NUM 69.7->65.5")
    # CON average
    i = content.find("<w:t>91.1</w:t>", pos)
    if i >= 0:
        content = content[:i] + "<w:t>91.0</w:t>" + content[i + len("<w:t>91.1</w:t>"):]
        print("  OK: Avg CON 91.1->91.0")
    # TMP average
    i = content.find("<w:t>79.1</w:t>", pos)
    if i >= 0:
        content = content[:i] + "<w:t>77.0</w:t>" + content[i + len("<w:t>79.1</w:t>"):]
        print("  OK: Avg TMP 79.1->77.0")
    # Overall average
    i = content.find("<w:t>80.8</w:t>", pos)
    if i >= 0:
        content = content[:i] + "<w:t>79.4</w:t>" + content[i + len("<w:t>80.8</w:t>"):]
        print("  OK: Avg Overall 80.8->79.4")

# Remove Haiku results row (second occurrence of &#x2020;Claude 3 Haiku after the model table)
haiku_m = "&#x2020;Claude 3 Haiku</w:t>"
idx = content.find(haiku_m)  # first occurrence is now gone; check if any remain
if idx >= 0:
    tr_start = content.rfind("<w:tr>", 0, idx)
    tr_end = content.find("</w:tr>", idx) + len("</w:tr>")
    content = content[:tr_start] + content[tr_end:]
    print("  OK: Remove Haiku from results table")
else:
    print("  INFO: No remaining Haiku rows (already removed)")

# Update Table 6 caption
replace_one(
    "Table 6. IndiaFinBench results &#x2014; accuracy (%) by task type. All eleven models evaluated on the full 406-item benchmark. &#x2020;Claude 3 Haiku evaluated on the initial 150-item subset (REG=53, NUM=32, CON=30, TMP=35); not directly comparable. 95% Wilson score confidence intervals.",
    "Table 6. IndiaFinBench results &#x2014; accuracy (%) by task type. All twelve models evaluated on the full 406-item benchmark. &#x2020;Gemini 2.5 Pro evaluated via Vertex AI; lower NUM/TMP scores reflect verbose output style penalised by reference-matching scoring. 95% Wilson score confidence intervals.",
    "Table 6 caption"
)

# ── Save ───────────────────────────────────────────────────────────
with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nDone. {original_len} -> {len(content)} chars (delta={len(content)-original_len:+d})")
