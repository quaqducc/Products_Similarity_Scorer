from typing import Optional, Any, Dict


class LLMWrapper:
	"""
	Wrapper for loading and running a HuggingFace model.
	This class imports Transformers lazily so that installing heavy
	dependencies is optional if you only need the prompt and retriever.
	"""

	def __init__(self, model_name: str = "google/flan-t5-base", device: int = -1, max_new_tokens: int = 512):
		self.model_name = model_name
		self.device = device
		self.max_new_tokens = max_new_tokens

		try:
			from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline  # type: ignore
		except Exception as exc:  # pragma: no cover - import error path
			raise RuntimeError(
				"Transformers is required for local inference. Install with: pip install transformers sentencepiece accelerate"
			) from exc

		# Load model + tokenizer
		self._tokenizer = AutoTokenizer.from_pretrained(model_name)
		self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

		# Build text generation pipeline
		self._generator = pipeline(
			"text2text-generation",
			model=self._model,
			tokenizer=self._tokenizer,
			device=device,
		)

	def run(self, prompt: str, temperature: float = 0.0, top_p: float = 1.0) -> str:
		"""
		Run the model on a given prompt and return generated text.
		"""
		output = self._generator(
			prompt,
			max_new_tokens=self.max_new_tokens,
			temperature=temperature,
			top_p=top_p,
			do_sample=temperature > 0.0,
		)
		return str(output[0]["generated_text"]).strip()



class ChatAPIWrapper:
	"""
	Wrapper for OpenAI-compatible Chat Completions APIs (e.g., NVIDIA integrate.api.nvidia.com).
	Imports the OpenAI client lazily.
	"""

	def __init__(
		self,
		*,
		base_url: str,
		api_key: str,
		model: str,
		max_tokens: int = 512,
	):
		try:
			from openai import OpenAI  # type: ignore
		except Exception as exc:  # pragma: no cover - import error path
			raise RuntimeError(
				"OpenAI client is required for chat API. Install with: pip install openai"
			) from exc

		self._client = OpenAI(base_url=base_url, api_key=api_key)
		self._model = model
		self._max_tokens = max_tokens
		self._base_url = base_url

	def run(
		self,
		prompt: str,
		*,
		temperature: float = 0.6,
		top_p: float = 0.95,
		extra_body: Optional[Dict[str, Any]] = None,
	) -> str:
		messages = [{"role": "user", "content": prompt}]
		# NVIDIA-specific default thinking tokens if not provided
		if extra_body is None and isinstance(self._base_url, str) and "integrate.api.nvidia.com" in self._base_url:
			extra_body = {"min_thinking_tokens": 1024, "max_thinking_tokens": 2048}

		resp = self._client.chat.completions.create(
			model=self._model,
			messages=messages,
			temperature=temperature,
			top_p=top_p,
			max_tokens=self._max_tokens,
			frequency_penalty=0,
			presence_penalty=0,
			stream=False,
			extra_body=extra_body or {},
		)
		msg = resp.choices[0].message
		reasoning = getattr(msg, "reasoning_content", None)
		content = (getattr(msg, "content", None) or "").strip()
		if reasoning:
			return (str(reasoning).strip() + "\n" + content).strip()
		return content

