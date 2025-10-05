# src/model.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import Optional


class LLMWrapper:
    """
    Wrapper for loading and running a HuggingFace model.
    """

    def __init__(self, model_name: str = "google/flan-t5-base", device: int = -1, max_new_tokens: int = 512):
        """
        Initialize tokenizer, model, and pipeline.
        :param model_name: HuggingFace model ID
        :param device: -1 = CPU, 0 = GPU
        :param max_new_tokens: max length of generated output
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        # Load model + tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Build text generation pipeline
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )

    def run(self, prompt: str, temperature: float = 0.0, top_p: float = 1.0) -> str:
        """
        Run the model on a given prompt.
        :param prompt: full input prompt
        :param temperature: sampling temperature
        :param top_p: nucleus sampling
        :return: raw generated text (string)
        """
        output = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0,
        )

        return output[0]["generated_text"].strip()
