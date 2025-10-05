import os
import csv
import json
import re
from typing import Dict

from prompt import build_prompt
from retriever import retrieve_contexts


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FEWSHOT_PATH = os.path.join(DATA_DIR, "fewshot_cases.json")


def parse_scores(output: str) -> Dict[str, int]:
    scores = {"nature": None, "purpose": None, "overall": None}
    if not output:
        return scores

    patterns = {
        "nature": r"Nature\s*Score:\s*(\d)",
        "purpose": r"Purpose\s*Score:\s*(\d)",
        "overall": r"Overall\s*(?:Similarity|Score):\s*(\d)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            scores[key] = int(m.group(1))
    return scores


def main():
    if not os.path.exists(FEWSHOT_PATH):
        raise FileNotFoundError(f"Missing fewshot cases at: {FEWSHOT_PATH}")

    with open(FEWSHOT_PATH, "r", encoding="utf-8") as f:
        fewshot_cases = json.load(f)

    test_cases = [
        ("Chemicals for industrial use", "chemical additives for detergents"),
        (
            "chemical products used in the manufacture of plastics and in the photocopying industry",
            "plastics in the form of granules, powders, masses, resins, gels, emulsions, dispersions, pastes, flakes, chips",
        ),
        ("Paints", "construction materials"),
    ]

    log_path = os.path.join(os.path.dirname(BASE_DIR), "results_log_1.csv")

    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "p1", "p2", "nature_score", "purpose_score", "overall_score"])

        for idx, (p1, p2) in enumerate(test_cases, start=1):
            contexts = retrieve_contexts(p1, p2, top_k=3)
            prompt = build_prompt(fewshot_cases, p1, p2, contexts, max_fewshot=2)

            # Try local model if available
            output_text = ""
            try:
                from model import LLMWrapper

                llm = LLMWrapper(model_name="google/flan-t5-base", device=-1, max_new_tokens=256)
                output_text = llm.run(prompt, temperature=0.0, top_p=1.0)
            except Exception:
                # Fallback: empty output, scores stay None
                output_text = ""

            scores = parse_scores(output_text)
            writer.writerow([idx, p1, p2, scores["nature"], scores["purpose"], scores["overall"]])

    print(f"✅ Log saved to {log_path}")


if __name__ == "__main__":
    main()


