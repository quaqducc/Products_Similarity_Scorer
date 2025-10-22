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
from .agents import FactorAgent, FactorAgentConfig, evaluate_multiple_factors
from .judge import LLMJudge, JudgeConfig

__all__ = [
	"build_prompt",
	"format_fewshot",
	"retrieve_contexts",
	"LLMWrapper",
	"run_similarity",
	"parse_scores",
    "FactorAgent",
    "FactorAgentConfig",
    "evaluate_multiple_factors",
    "LLMJudge",
    "JudgeConfig",
]

__version__ = "0.1.0"


