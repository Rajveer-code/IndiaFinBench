# IndiaFinBench Submission Checklist

## Paper (paper/indiafinbench_paper_v3_submission.md)
- [x] FIX 1: Abstract verb consistency ("introduce" throughout)
- [x] FIX 2: MMLU/HELM added to Section 2.3 with references
- [x] FIX 3: Difficulty assignment authorship stated explicitly
- [x] FIX 4: Validator name corrected (LLaMA-3.3-70B-Versatile) + conflict-of-interest explanation
- [x] FIX 5: Zero-shot justification sentence added to Section 4.2
- [x] FIX 6: 0.72 threshold justified with three-way threshold comparison
- [x] FIX 7: Table 1 header has full task names inline
- [x] FIX 8: Confidence interval paragraph added to Section 5.1
- [x] FIX 9: Error taxonomy disclosure (rule-based assignment) added to Section 6.1
- [x] FIX 10: Limitations strengthened with CI reference
- [x] FIX 11: Conclusion verb fixed ("present" → "introduce")
- [x] FIX 12: Appendix B has verbatim prompt templates (not just path reference)
- [x] FIX 13: Appendix C.2 has Wilson CI table with computed values
- [x] FIX 14: Ethics Statement added
- [x] FIX 15: Acknowledgements added

### FIX 16 Consistency Scan Results
- [x] "Gemma" appears 0 times in paper body ✅
- [x] "We presented" → changed to "We introduce" in Conclusion ✅
- [x] "future work" confirmed in Section 7.2 and Conclusion ✅
- [x] All \cite references have corresponding entries in References ✅
- [x] Table 1 header has full task names (REG (Reg. Interp.) etc.) ✅
- [x] Appendix B has actual verbatim prompt text from evaluation/prompts/ ✅
- [x] Appendix C.2 has Wilson CI table with computed numbers ✅

## LaTeX (paper/indiafinbench_v3.tex + paper/references.bib)
- [x] All tables use booktabs style (\toprule \midrule \bottomrule)
- [x] Table 1 has \textbf{} for best-per-column values
- [x] All 13 references in references.bib with correct BibTeX keys
- [x] Appendix uses \appendix \section{}
- [x] Ethics and Acknowledgements as \section*{}

## README (README.md)
- [x] Item count corrected (150 actual, 750 target)
- [x] Task distribution table shows actual counts (53/32/30/35)
- [x] Project status table is current (Phases 3 & 4 complete, v3 complete)
- [x] Model list matches Table 1 exactly (Claude/Gemini/LLaMA-70B/LLaMA-8B/Mistral)
- [x] Results summary table added
- [x] Citation BibTeX block added

## HuggingFace Upload (scripts/upload_to_huggingface.py)
- [x] Loads actual benchmark data from annotation/raw_qa/indiafinbench_qa_combined_150.json
- [x] Stratified train/test split (80/20, seed=42)
- [x] Dataset card (README_HF.md) written by write_dataset_card()
- [x] push_to_hub() call present

## Gradio Demo (demo/app.py + demo/requirements.txt)
- [x] Loads dataset from HuggingFace (Rajveer-code/IndiaFinBench, split="test")
- [x] Task type + difficulty filtering works via load_random_question()
- [x] Reference answer hidden until "Reveal Reference Answer" clicked
- [x] About tab with results table and links
- [x] requirements.txt present (gradio>=4.0.0, datasets>=2.0.0)

---

## Output File Summary

| Phase | File | Status |
|---|---|---|
| Phase 1 | `paper/indiafinbench_paper_v3_submission.md` | ✅ Written |
| Phase 2 | `paper/indiafinbench_v3.tex` | ✅ Written |
| Phase 2 | `paper/references.bib` | ✅ Written |
| Phase 3 | `README.md` | ✅ Updated |
| Phase 4 | `scripts/upload_to_huggingface.py` | ✅ Written |
| Phase 4 | `README_HF.md` | ✅ Generated at runtime |
| Phase 5 | `demo/app.py` | ✅ Written |
| Phase 5 | `demo/requirements.txt` | ✅ Written |
| Phase 6 | `SUBMISSION_CHECKLIST.md` | ✅ Written |

---

## Known Manual Steps Before Submission (cannot be automated)

1. **HuggingFace upload:**
   Run `huggingface-cli login` then `python scripts/upload_to_huggingface.py`

2. **LaTeX compilation:**
   Download ACL style file (`acl.sty`) from the ACL Anthology,
   place it in `paper/`, then compile:
   ```
   cd paper
   pdflatex indiafinbench_v3.tex
   bibtex   indiafinbench_v3
   pdflatex indiafinbench_v3.tex
   pdflatex indiafinbench_v3.tex
   ```

3. **arXiv submission:**
   After upload, update the BibTeX `journal` field in:
   - `paper/references.bib` (self-citation entry if added)
   - `README.md` (Citation section)
   - `demo/app.py` (About tab)
   with the actual arXiv ID (e.g., `arXiv:2506.XXXXX`).

4. **Gradio Spaces deployment:**
   Create a new HuggingFace Space, select Gradio SDK, and upload
   `demo/app.py` and `demo/requirements.txt`.
   The Space will auto-launch after `upload_to_huggingface.py` has run
   (so the dataset is available to load).

5. **SOP paragraph update:**
   Update P3_LLM_Benchmark_Blueprint.docx with the final accuracy numbers:
   - Overall range: 72.7%–91.3%
   - Best model: Claude 3 Haiku (91.3%)
   - Most discriminative task: Numerical Reasoning (~31 point spread)

---

## Phase 0 Findings (for author reference)

- **Validator model in `run_ai_annotator.py`**: `llama-3.3-70b-versatile` (line 33) — paper v2 incorrectly said "Gemma-2-9B-IT" — **FIXED in v3**
- **Paper v2 location**: `indiafinbench_paper_v2 (1).md` (repo root, not `paper/`)
- **Dataset file**: `annotation/raw_qa/indiafinbench_qa_combined_150.json` (1,820 lines, 150 items confirmed)
- **README discrepancy**: README said 750 items, paper said 150 — **README now corrected**
- **All results match**: Numbers in `evaluation/error_analysis/error_report.md` match Table 1 exactly ✅
- **Prompt files**: All 4 read verbatim from `evaluation/prompts/` — inserted into Appendix B ✅
