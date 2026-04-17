# AI Annotator Prompt — IndiaFinBench Validation Study

**HOW TO USE THIS:**
1. Copy everything between the ═══ lines below
2. Paste it into Claude / ChatGPT / Gemini as the first message
3. Then paste the CSV rows (or attach the file)
4. The AI will output the filled column

---

═══════════════════════════════════════════════════════════════
SYSTEM CONTEXT (paste this first):

You are an independent annotator for an academic research project validating
a dataset of Indian financial regulatory question-answer pairs.

YOUR TASK:
For each row I give you, you will see:
  - row_number: the row index
  - task_type: the category (regulatory_interpretation / numerical_reasoning /
               contradiction_detection / temporal_reasoning)
  - difficulty: easy / medium / hard
  - context: a passage from an official Indian financial regulation document
             (SEBI or RBI circulars, regulations, master directions)
  - question: a question about that context
  - reference_answer: the answer being validated

You must decide: Is the reference_answer CORRECT given ONLY the context provided?

STRICT RULES — READ CAREFULLY:
1. DO NOT use any external knowledge, internet search, or training data about
   Indian finance. Base your judgment ONLY on the text in the "context" field.
2. DO NOT assume knowledge of SEBI regulations, RBI rules, Companies Act, etc.
   beyond what is explicitly stated in the provided context.
3. Mark YES if the reference_answer is a correct and complete answer to the
   question, derivable solely from the context.
4. Mark NO if the reference_answer is wrong, incomplete, or requires information
   not present in the context.
5. For numerical questions: minor formatting differences are NOT errors.
   "₹5,500 crore" and "5500 crore" mean the same thing → mark YES if the
   numeric value is correct.
6. For contradiction_detection questions (Yes/No questions about whether two
   passages contradict): mark YES if the reference_answer (Yes or No) correctly
   identifies whether the passages contradict each other per the context.
7. When in doubt, prefer YES — mark NO only when you are confident the
   reference_answer is factually wrong based on what the context states.

OUTPUT FORMAT:
For each row, output exactly one line in this format:
  Row [row_number]: [YES/NO] | [one-sentence reason if NO, else leave blank]

Example output:
  Row 1: YES |
  Row 2: NO | Context says 3 years but reference_answer says 5 years
  Row 3: YES |
  Row 7: NO | Reference answer gives wrong percentage; context states 25% not 30%

After all rows, output a summary line:
  SUMMARY: [total rows] rows | [count YES] YES | [count NO] NO

═══════════════════════════════════════════════════════════════

---

## Then paste the CSV rows like this:

```
--- BEGIN ANNOTATION TASK ---

Row 1:
task_type: regulatory_interpretation
difficulty: medium
context: [paste context here]
question: [paste question here]
reference_answer: [paste reference answer here]

Row 2:
...
--- END ANNOTATION TASK ---
```

---

## Tips for getting the best results from AI annotators

**For Claude:** Works well with this prompt as-is. Claude will carefully read
each context and give reliable YES/NO judgments.

**For ChatGPT (GPT-4o):** Add at the top: "Do not use any information from
your training data about Indian finance law. Only use the text I provide."

**For Gemini:** Same prompt works. Add "Do not search the internet."

**Batch size:** For best reliability, give 20-30 rows at a time rather than
all 90 at once. Paste rows 1-30, collect output, then rows 31-60, etc.

**If the AI starts adding financial knowledge:** Remind it:
"Stop. You are only allowed to use the text in the context field. Do not
add any knowledge from outside the provided text."

---

## What to do with the output

Once you have YES/NO for all 90 rows, save as:
  annotation/multi_annotator/annotator2_completed.csv   (BTech CSE annotator)
  annotation/multi_annotator/annotator3_completed.csv   (MBA annotator)

Format of the completed CSV:
  id, task_type, difficulty, context, question, reference_answer,
  is_correct__YES_or_NO, notes_if_NO

Then run:
  python scripts/compute_fleiss_kappa.py
