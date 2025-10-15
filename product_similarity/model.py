from typing import Optional


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


