from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class JudgeConfig:
	weights: Optional[Dict[str, float]] = None  # per-factor weights; defaults applied if None


class LLMJudge:
	"""
	Combine factor-level agent outputs into final overall similarity.

	Current implementation: deterministic weighted average with simple tie-breaking.
	The interface allows future replacement with an LLM-based judge if needed.
	"""

	def __init__(self, config: Optional[JudgeConfig] = None) -> None:
		cfg = config or JudgeConfig()
		self._weights = cfg.weights or {
			"Nature": 0.5,
			"Intended Purpose": 0.5,
			# Other factors may be added, default to 0 if missing
		}

	def _normalize_weights(self, factors: List[str]) -> Dict[str, float]:
		weights: Dict[str, float] = {}
		total = 0.0
		for f in factors:
			w = float(self._weights.get(f, 0.0))
			weights[f] = w
			total += w
		if total <= 0.0:
			# fallback uniform over provided factors
			n = max(len(factors), 1)
			return {f: 1.0 / n for f in factors}
		return {f: (weights[f] / total) for f in factors}

	def combine_factor_scores(self, factor_outputs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
		"""
		factor_outputs: mapping factor -> { score: int|None, reasoning_text: str, ... }
		Returns a dict containing final_overall (int), details per factor, and weighted breakdown.
		"""
		factors = list(factor_outputs.keys())
		weights = self._normalize_weights(factors)

		weighted_sum = 0.0
		sum_weights = 0.0
		details: Dict[str, Dict[str, object]] = {}

		for f in factors:
			entry = factor_outputs.get(f, {})
			raw_score = entry.get("score")
			score_val = float(raw_score) if isinstance(raw_score, int) else None
			w = float(weights.get(f, 0.0))
			details[f] = {
				"weight": w,
				"score": raw_score,
				"text": entry.get("reasoning_text", ""),
			}
			if score_val is not None:
				weighted_sum += score_val * w
				sum_weights += w

		final_score = round(weighted_sum / sum_weights) if sum_weights > 0 else 0

		return {
			"overall_similarity": int(final_score),
			"weights": weights,
			"details": details,
		}


