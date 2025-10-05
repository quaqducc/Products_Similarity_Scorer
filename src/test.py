import json
import os
import re
from prompt import build_prompt

# Load few-shot cases
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FEWSHOT_PATH = os.path.join(DATA_DIR, "fewshot_cases.json")
NICE_PATH = os.path.join(DATA_DIR, "nice_chunks.json")

with open(FEWSHOT_PATH, "r", encoding="utf-8") as f:
    fewshot_cases = json.load(f)

# Load NICE chunks once (large file)
with open(NICE_PATH, "r", encoding="utf-8") as f:
    nice_chunks = json.load(f)


def retrieve_contexts(product_1: str, product_2: str, top_k: int = 3):
    """
    Very simple keyword-based retriever over NICE data.
    Returns top_k short context strings.
    """
    # Extract basic keywords (length > 3) from product strings
    terms = set(
        [t for t in re.findall(r"[A-Za-z]+", (product_1 + " " + product_2).lower()) if len(t) > 3]
    )

    scored = []
    for entry in nice_chunks:
        heading = entry.get("heading", "")
        note = entry.get("explanatory_note", "")
        items = entry.get("items", [])
        class_no = entry.get("class_number", "?")

        blob = (heading + "\n" + note + "\n" + "\n".join([it.get("Goods and Service", "") for it in items])).lower()
        score = sum(blob.count(term) for term in terms)

        if score > 0:
            # Build a short summary context
            matched_items = [it.get("Goods and Service", "") for it in items if any(term in it.get("Goods and Service", "").lower() for term in terms)]
            snippet_items = "; ".join(matched_items[:3])
            context = f"Class {class_no}: {heading}"
            if snippet_items:
                context += f"\nExamples: {snippet_items}"
            scored.append((score, context))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ctx for _, ctx in scored[:top_k]]

# Example input
p1 = "Make-up preparations"
p2 = "Tissues of paper for removing make-up"

# Retrieve contexts from NICE data
retrieved_contexts = retrieve_contexts(p1, p2, top_k=3)

# Build prompt
prompt = build_prompt(fewshot_cases, p1, p2, retrieved_contexts, max_fewshot=2)

print("=== PROMPT (truncated preview) ===")
print(prompt[:1200] + ("..." if len(prompt) > 1200 else ""))
print()

# Try local inference if available
try:
    from model import LLMWrapper

    llm = LLMWrapper(model_name="google/flan-t5-base", device=-1, max_new_tokens=256)
    print("=== MODEL OUTPUT ===")
    output_text = llm.run(prompt, temperature=0.0, top_p=1.0)
    print(output_text)
except Exception as e:
    print("[Info] Local transformers pipeline not available or failed to run.")
    print("Reason:", str(e))
    print("You can still run the full pipeline on Kaggle using the provided notebook with remote inference.")
