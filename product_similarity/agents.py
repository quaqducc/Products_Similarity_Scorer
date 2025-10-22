from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def _build_agent_prompt(
	factor_name: str,
	product_1: str,
	product_2: str,
	context: Optional[str] = None,
) -> str:
	parts = [
		"You are a domain expert agent specialized in one factor of product similarity.",
		f"Your factor: {factor_name}.",
		"Assess the two products for this factor only. Provide:",
		"- Reasoning: 2-4 concise sentences grounded in the provided context if relevant.",
		"- Score: an integer in [0,4] where 0=Not similar, 4=Highly similar.",
		"Return as plain text with headings 'Reasoning:' and 'Score:'.",
		"\nProducts:",
		f"- Product 1: {product_1}",
		f"- Product 2: {product_2}",
	]
	if context:
		parts.append("\nContext:\n" + str(context))
	parts.append(
		"\nOutput format:\nReasoning: <brief analysis>\nScore: <0-4>"
	)
	return "\n".join(parts)


@dataclass
class FactorAgentConfig:
	model_name: str = DEFAULT_MODEL
	device: int = -1
	max_new_tokens: int = 256
	temperature: float = 0.0
	top_p: float = 1.0


class FactorAgent:
	"""
	Evaluate a single factor using a dedicated HF/NVIDIA pipeline.

	By default uses a Hugging Face text-generation pipeline for instruction models.
	Set per-factor model overrides via constructor map.
	"""

	def __init__(
		self,
		*,
		default: Optional[FactorAgentConfig] = None,
		per_factor: Optional[Dict[str, FactorAgentConfig]] = None,
		use_chat_api: bool = False,
		chat_api_base_url: Optional[str] = None,
		chat_api_key: Optional[str] = None,
		chat_api_model: Optional[str] = None,
	):
		self._default = default or FactorAgentConfig()
		self._per_factor = per_factor or {}
		self._use_chat_api = use_chat_api
		self._chat_base = chat_api_base_url
		self._chat_key = chat_api_key
		self._chat_model = chat_api_model

		self._pipeline_cache: Dict[str, object] = {}

	def _get_config(self, factor_name: str) -> FactorAgentConfig:
		return self._per_factor.get(factor_name, self._default)

	def _get_pipeline(self, model_name: str, device: int) -> object:
		if model_name in self._pipeline_cache:
			return self._pipeline_cache[model_name]
		try:
			from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore
		except Exception as exc:
			raise RuntimeError(
				"Transformers is required for FactorAgent. Install with: pip install transformers accelerate"
			) from exc
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		model = AutoModelForCausalLM.from_pretrained(model_name)
		pipe = pipeline(
			"text-generation",
			model=model,
			tokenizer=tokenizer,
			device=device,
		)
		self._pipeline_cache[model_name] = pipe
		return pipe

	def _run_hf(self, cfg: FactorAgentConfig, prompt: str) -> str:
		pipe = self._get_pipeline(cfg.model_name, cfg.device)
		out = pipe(
			prompt,
			max_new_tokens=cfg.max_new_tokens,
			temperature=cfg.temperature,
			top_p=cfg.top_p,
			do_sample=cfg.temperature > 0.0,
		)[0]["generated_text"]
		return str(out).strip()

	def _run_chat(self, prompt: str, cfg: FactorAgentConfig) -> str:
		if not (self._chat_base and self._chat_key and self._chat_model):
			raise RuntimeError("Chat API configuration is incomplete for FactorAgent.")
		try:
			from openai import OpenAI  # type: ignore
		except Exception as exc:
			raise RuntimeError("Install openai for chat API: pip install openai") from exc
		client = OpenAI(base_url=str(self._chat_base), api_key=str(self._chat_key))
		resp = client.chat.completions.create(
			model=str(self._chat_model),
			messages=[{"role": "user", "content": prompt}],
			max_tokens=cfg.max_new_tokens,
			temperature=max(cfg.temperature, 0.0),
			top_p=cfg.top_p,
			stream=False,
		)
		msg = resp.choices[0].message
		reasoning = getattr(msg, "reasoning_content", None)
		content = (getattr(msg, "content", None) or "").strip()
		if reasoning:
			return (str(reasoning).strip() + "\n" + content).strip()
		return content

	@staticmethod
	def _parse_score(output_text: str) -> Optional[int]:
		import re
		m = re.search(r"Score\s*[:\-]\s*(\d)", output_text, flags=re.IGNORECASE)
		return int(m.group(1)) if m else None

	def evaluate(
		self,
		factor_name: str,
		product_1: str,
		product_2: str,
		context: Optional[str] = None,
	) -> Dict[str, Optional[object]]:
		"""
		Run the agent for one factor and return a dict with text and score.
		Keys: factor, reasoning_text, raw_output, score
		"""
		cfg = self._get_config(factor_name)
		prompt = _build_agent_prompt(factor_name, product_1, product_2, context)
		if self._use_chat_api:
			generated = self._run_chat(prompt, cfg)
		else:
			generated = self._run_hf(cfg, prompt)
		score = self._parse_score(generated)
		return {
			"factor": factor_name,
			"reasoning_text": generated,
			"raw_output": generated,
			"score": score,
		}


def evaluate_multiple_factors(
	agent: FactorAgent,
	product_1: str,
	product_2: str,
	factors: list[str],
	contexts: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Optional[object]]]:
	results: Dict[str, Dict[str, Optional[object]]] = {}
	for f in factors:
		ctx = (contexts or {}).get(f)
		results[f] = agent.evaluate(f, product_1, product_2, ctx)
	return results


