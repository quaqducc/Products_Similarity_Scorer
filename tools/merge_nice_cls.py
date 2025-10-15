import json
import os
import re
from glob import glob


def split_heading_and_note(meta_goods_and_services: str) -> tuple[str, str]:
	match = re.search(r"\bExplanatory Note\b", meta_goods_and_services)
	if match:
		heading = meta_goods_and_services[: match.start()].strip()
		note = meta_goods_and_services[match.end():].strip()
	else:
		heading = meta_goods_and_services.strip()
		note = ""
	if heading.lower().startswith("class heading"):
		heading = heading[len("Class Heading"):].strip()
	return heading, note


def load_group(file_path: str) -> dict:
	with open(file_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	class_number = str(data.get("group_number") or "").strip()
	meta = data.get("meta") or {}
	goods_and_services_blob = str(meta.get("Goods and Services") or "")
	heading, note = split_heading_and_note(goods_and_services_blob)

	items_in = data.get("items") or []
	items_out = []
	for it in items_in:
		no_val = it.get("No.") if isinstance(it, dict) else None
		gs_val = it.get("Goods and Services") if isinstance(it, dict) else None
		if not no_val and not gs_val:
			continue
		items_out.append({
			"No": str(no_val).strip() if no_val is not None else "",
			"Goods and Service": str(gs_val).strip() if gs_val is not None else "",
		})

	return {
		"class_number": class_number,
		"heading": heading,
		"explanatory_note": note,
		"items": items_out,
	}


def main() -> None:
	project_root = os.path.dirname(os.path.dirname(__file__))
	input_dir = os.path.join(project_root, "data_nice_cls")
	output_file = os.path.join(project_root, "data", "nice_chunks.json")

	paths = glob(os.path.join(input_dir, "group_*.json"))

	def extract_num(p: str) -> int:
		m = re.search(r"group_(\d+)\.json$", p)
		return int(m.group(1)) if m else 0

	paths.sort(key=extract_num)

	merged = []
	for p in paths:
		merged.append(load_group(p))

	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	with open(output_file, "w", encoding="utf-8") as f:
		json.dump(merged, f, ensure_ascii=False, indent=2)
	print(f"Written {len(merged)} classes to {output_file}")


if __name__ == "__main__":
	main()


