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
	Build the full prompt for LLM evaluation.
	"""

	# Header with task description
	instruction = (
		"You are an expert in trademark law and NICE classification.\n"
		"Your task is to evaluate the similarity between two products "
		"according to the official NICE classification and guideline factors: Nature, Intended Purpose and Channels of Trade(optionally)\n\n"
		"Scoring system (for each factor and overall):\n"
		"0 = Not similar, 1 = Slightly similar, 2 = Somewhat similar, "
		"3 = Similar, 4 = Highly similar.\n\n"
	)

	# Format few-shot examples (optionally limit the number used)
	limited_examples = (
		fewshot_examples[:max_fewshot]
		if isinstance(max_fewshot, int) and max_fewshot > 0
		else fewshot_examples
	)
	fewshot_text = "\n\n".join([format_fewshot(ex) for ex in limited_examples])

	# Context from retriever
	context_text = (
		"### Reference Context (NICE classification & guidelines):\n"
		+ "\n".join(retrieved_contexts)
		+ "\n\n"
	)

	# New query
	query_text = (
		"### New Case\n"
		f"- Product 1: {product_1}\n"
		f"- Product 2: {product_2}\n\n"
		"Please provide your reasoning step by step for each relevant factor "
		"(nature, intended purpose, usual origin, channels of trade, etc.), "
		"and then give a structured output in the following format:\n\n"
		"Reasoning:\n"
		"- Factor 1: ...\n"
		"- Factor 2: ...\n\n"
		"Output:\n"
		"- Nature Score: [0–4]\n"
		"- Purpose Score: [0–4]\n"
		"- Overall Similarity: [0–4]\n"
	)

	# Final combined prompt
	return instruction + fewshot_text + "\n\n" + context_text + query_text


