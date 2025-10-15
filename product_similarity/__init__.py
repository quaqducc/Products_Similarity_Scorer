"""Product Similarity utilities.

Includes:
- Prompt builders
- NICE keyword retriever
- HF Transformers wrapper (optional)
- End-to-end pipeline helpers
"""

from .prompt import build_prompt, format_fewshot
from .retriever import retrieve_contexts
from .model import LLMWrapper
from .pipeline import run_similarity, parse_scores

__all__ = [
	"build_prompt",
	"format_fewshot",
	"retrieve_contexts",
	"LLMWrapper",
	"run_similarity",
	"parse_scores",
]

__version__ = "0.1.0"


