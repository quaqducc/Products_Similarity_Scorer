import json
import os
import re
from typing import Dict, List, Optional, Tuple


# Resolve project-relative paths
PACKAGE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
SPSC_PATH = os.path.join(PROJECT_ROOT, "spsc_data", "spsc_data", "spsc_tree.json")


def _load_spsc_tree() -> Dict:
	"""
	Load SPSC tree JSON. Raises FileNotFoundError if missing.
	"""
	if not os.path.exists(SPSC_PATH):
		raise FileNotFoundError(f"Missing SPSC tree at: {SPSC_PATH}")
	with open(SPSC_PATH, "r", encoding="utf-8") as f:
		return json.load(f)


_SPSC_FLAT_CACHE: Optional[List[Dict[str, str]]] = None


def _flatten_nodes(node: Dict, path_titles: List[str], path_codes: List[str], out: List[Dict[str, str]]) -> None:
	title = str(node.get("title", "")).strip()
	code = str(node.get("code", "")).strip()

	cur_titles = path_titles + ([title] if title else [])
	cur_codes = path_codes + ([code] if code else [])

	full_path_title = " > ".join([t for t in cur_titles if t])
	full_path_code = " > ".join([c for c in cur_codes if c])

	if title or code:
		out.append({
			"title": title,
			"code": code,
			"path_title": full_path_title,
			"path_code": full_path_code,
		})

	for child in node.get("children", []) or []:
		_flatten_nodes(child, cur_titles, cur_codes, out)


def _get_spsc_flat_cached() -> List[Dict[str, str]]:
	global _SPSC_FLAT_CACHE
	if _SPSC_FLAT_CACHE is not None:
		return _SPSC_FLAT_CACHE
	data = _load_spsc_tree()
	flat: List[Dict[str, str]] = []
	for root in data.get("roots", []) or []:
		_flatten_nodes(root, [], [], flat)
	_SPSC_FLAT_CACHE = flat
	return flat


def retrieve_spsc_contexts(product_1: str, product_2: str, *, top_k: int = 2) -> List[str]:
	"""
	Lightweight keyword-based matching from product descriptions to SPSC nodes.
	Returns short context strings to be appended to the LLM prompt.
	"""
	text = f"{product_1} {product_2}".lower()
	terms = set([t for t in re.findall(r"[a-z]+", text) if len(t) > 3])
	if not terms:
		return []

	scored: List[Tuple[int, Dict[str, str]]] = []
	for n in _get_spsc_flat_cached():
		blob = (n.get("title", "") + " " + n.get("path_title", "")).lower()
		score = sum(blob.count(term) for term in terms)
		if score > 0:
			scored.append((score, n))

	if not scored:
		return []

	scored.sort(key=lambda x: x[0], reverse=True)
	top = [n for _, n in scored[:max(int(top_k), 0)]]

	contexts: List[str] = []
	for n in top:
		title = n.get("title", "")
		code = n.get("code", "")
		path_title = n.get("path_title", "")
		parts = []
		if code:
			parts.append(f"SPSC {code}: {title}".strip())
		else:
			parts.append(f"SPSC: {title}".strip())
		if path_title and path_title != title:
			parts.append(f"Path: {path_title}")
		contexts.append("\n".join(parts))
	return contexts


