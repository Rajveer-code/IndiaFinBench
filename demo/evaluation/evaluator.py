"""
evaluation/evaluator.py
-----------------------
Purpose:  Run any HuggingFace text-generation model against all 150 IndiaFinBench
          questions and return a predictions dict for score_submission().
Inputs:   hf_model_id (str), questions (list[dict])
Outputs:  predictions dict {id: answer_str}
Usage:
    from evaluation.evaluator import IndiaFinBenchEvaluator
    evaluator = IndiaFinBenchEvaluator("mistralai/Mistral-7B-Instruct-v0.3")
    preds = evaluator.run(questions)
"""

import sys
import time
from typing import Any

from .tasks import SYSTEM_PROMPT, build_prompt, extract_answer

# Timeout per item (seconds) — avoids hanging on large models
ITEM_TIMEOUT_S = 30
# Maximum new tokens to generate
MAX_NEW_TOKENS = 200


class IndiaFinBenchEvaluator:
    """Load a HuggingFace model and evaluate it on IndiaFinBench questions.

    Supports any model accessible via `transformers.pipeline` with
    task="text-generation". The pipeline is loaded lazily on first call to run().

    Args:
        hf_model_id: HuggingFace model ID, e.g. "mistralai/Mistral-7B-Instruct-v0.3".
        device:      "cpu", "cuda", or "auto" (default).
        progress_cb: Optional callback(done: int, total: int) for progress tracking.
    """

    def __init__(
        self,
        hf_model_id: str,
        device: str = "auto",
        progress_cb: Any = None,
    ) -> None:
        self.hf_model_id = hf_model_id
        self.device      = device
        self.progress_cb = progress_cb
        self._pipeline   = None

    def _load(self) -> None:
        """Lazily load the HuggingFace pipeline."""
        try:
            from transformers import pipeline, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers required for model evaluation.\n"
                "Run: pip install transformers torch"
            )

        print(f"  Loading model: {self.hf_model_id} ...", file=sys.stderr)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
            self._pipeline = pipeline(
                "text-generation",
                model=self.hf_model_id,
                tokenizer=tokenizer,
                device_map=self.device if self.device != "auto" else "auto",
                torch_dtype="auto",
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            print(f"  Model loaded: {self.hf_model_id}", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{self.hf_model_id}': {e}"
            )

    def _call(self, prompt: str) -> str:
        """Run a single prompt through the pipeline.

        Args:
            prompt: Formatted prompt string.

        Returns:
            Model output string (generated text only, preamble stripped).
        """
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        try:
            outputs = self._pipeline(
                full_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                return_full_text=False,
            )
            if outputs and isinstance(outputs, list):
                return outputs[0].get("generated_text", "").strip()
        except Exception as e:
            return f"FAIL: {str(e)[:100]}"
        return ""

    def run(
        self,
        questions: list[dict],
        max_items: int | None = None,
    ) -> dict[str, str]:
        """Evaluate the model on all benchmark questions.

        Args:
            questions:  List of question dicts from questions.json.
            max_items:  If set, evaluate only the first max_items questions
                        (useful for quick smoke-tests).

        Returns:
            Dict mapping question id -> extracted answer string.
        """
        if self._pipeline is None:
            self._load()

        subset    = questions[:max_items] if max_items else questions
        n         = len(subset)
        preds: dict[str, str] = {}

        for i, item in enumerate(subset, 1):
            qid     = item["id"]
            task    = item["task_type"]
            prompt  = build_prompt(item)

            try:
                raw  = self._call(prompt)
                pred = extract_answer(raw, task)
            except Exception as e:
                pred = f"FAIL: {str(e)[:80]}"

            preds[qid] = pred

            if self.progress_cb:
                self.progress_cb(i, n)

            if i % 25 == 0:
                print(f"  [{i:03d}/{n}] {qid}", file=sys.stderr)

        return preds
