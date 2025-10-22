import json
from typing import List, Dict, Optional


def format_fewshot(example: Dict) -> str:
	"""
	Format one few-shot example into a structured text block.
	"""
	case_id = example.get("case_id", "UNKNOWN")
	input_data = example.get("input", {})
	reasoning = example.get("reasoning", {})
	output = example.get("output", {})

	return (
		f"### Example {case_id}\n"
		f"**Input:**\n"
		f"- Product 1: {input_data.get('product_1')}\n"
		f"- Product 2: {input_data.get('product_2')}\n"
		f"- Class Info: {json.dumps(input_data.get('class_info', {}), ensure_ascii=False, indent=2)}\n\n"
		f"**Reasoning:**\n"
		+ "\n".join([f"- {k.capitalize()}: {v}" for k, v in reasoning.items()])
		+ "\n\n"
		f"**Output:**\n"
		f"- Nature Score: {output.get('nature_score')}\n"
		f"- Purpose Score: {output.get('purpose_score')}\n"
		f"- Overall Similarity: {output.get('overall_similarity')}\n"
	)

def build_prompt(
    fewshot_examples: List[Dict],
    product_1: str,
    product_2: str,
    retrieved_contexts: List[str],
    max_fewshot: Optional[int] = None,
) -> str:
    """
    Build the full prompt for LLM evaluation (revised version).
    Keeps original output format (Reasoning + Output) but simplifies and strengthens instruction.
    """

    # ==== 1. Task description ====
    instruction = (
        "You are an expert examiner in trademark classification (NICE system).\n"
        "Your task is to assess the similarity between two goods or services.\n"
        "Base your reasoning primarily on these factors:\n"
        "- Nature (what the product is, its composition or type)\n"
        "- Intended Purpose (what the product is used for)\n"
        "- Optionally: Channels of trade, usual origin, or complementarity if relevant.\n\n"

        "=== EVALUATION PROCESS ===\n"
        "Step 1. Analyze NATURE:\n"
        "  • Identify what each product essentially is (chemical, device, material, etc.).\n"
        "  • Are they in the same technical category or made of similar materials?\n\n"
        "Step 2. Analyze INTENDED PURPOSE:\n"
        "  • What is each product used for?\n"
        "  • Do they serve the same user need or are they complementary?\n\n"
        "Step 3. Combine your findings and give an OVERALL SIMILARITY score.\n"
        "  • Consider the strength of overlap in purpose and nature.\n"
        "  • If Intended Purpose is strongly similar, it can raise the Overall score even if Nature differs.\n\n"

        "=== SCORING SCALE ===\n"
        "Use whole numbers only:\n"
        "0 = Not similar\n"
        "1 = Slightly similar\n"
        "2 = Moderately similar\n"
        "3 = Similar\n"
        "4 = Highly similar / identical\n\n"
        "Keep your reasoning concise and factual (max 3–5 sentences per case).\n"
        "Your final answer must include both 'Reasoning' and 'Output' sections, in the same format as examples.\n\n"
    )

    # ==== 2. Few-shot examples ====
    limited_examples = (
        fewshot_examples[:max_fewshot]
        if isinstance(max_fewshot, int) and max_fewshot > 0
        else fewshot_examples
    )
    fewshot_text = "\n\n".join([format_fewshot(ex) for ex in limited_examples])

    # ==== 3. NICE / guideline context ====
    context_text = ""
    if retrieved_contexts:
        context_text = (
            "### Reference Context (from NICE classification & Guidelines):\n"
            + "\n".join(retrieved_contexts)
            + "\n\n"
        )

    # ==== 4. Query ====
    query_text = (
        "### New Case\n"
        f"- Product 1: {product_1}\n"
        f"- Product 2: {product_2}\n\n"
        "Please reason step-by-step based on the factors above.\n"
        "Then provide your conclusion following this format:\n\n"
        "Reasoning:\n"
        "- Nature: (brief reasoning)\n"
        "- Intended Purpose: (brief reasoning)\n"
        "- Optional factors: (if any)\n\n"
        "Output:\n"
        "- Nature Score: [0–4]\n"
        "- Purpose Score: [0–4]\n"
        "- Overall Similarity: [0–4]\n"
        "Your final line must clearly show the Overall Similarity score.\n"
    )

    # ==== 5. Combine ====
    return instruction + fewshot_text + "\n\n" + context_text + query_text

