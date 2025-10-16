import os
from pathlib import Path
import pandas as pd


def split_first_token(text: object) -> tuple[str, str]:
	"""Return (first_token, remainder_without_first) split by space. Empty-safe."""
	s = str(text or "").strip()
	if not s:
		return "", ""
	parts = s.split()
	first = parts[0]
	rest = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
	return first, rest


def main() -> None:
	project_root = Path(__file__).resolve().parents[1]
	data_dir = project_root / "data"
	input_csv = data_dir / "100_samples.csv"
	output_csv = data_dir / "75_samples.csv"
	if not input_csv.exists():
		raise FileNotFoundError(f"Input not found: {input_csv}")

	df = pd.read_csv(input_csv)
	# Take first 75 rows
	df75 = df.head(75).copy()

	col0 = 'Item 1'
	col1 = 'Item 2'

	# Extract class ids as first token, and remove them from product names
	cls1_rest = df75[col0].apply(split_first_token)
	cls2_rest = df75[col1].apply(split_first_token)
	df75["class1"] = cls1_rest.map(lambda t: t[0])
	df75[col0] = cls1_rest.map(lambda t: t[1])
	df75["class2"] = cls2_rest.map(lambda t: t[0])
	df75[col1] = cls2_rest.map(lambda t: t[1])

	data_dir.mkdir(parents=True, exist_ok=True)
	df75.to_csv(output_csv, index=False)
	print(f"Wrote {len(df75)} rows to {output_csv}")


if __name__ == "__main__":
	main()


