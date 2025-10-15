import argparse
import json
import os
import subprocess
import sys

from product_similarity.pipeline import run_similarity


def cmd_run(args: argparse.Namespace) -> int:
	result = run_similarity(
		product_1=args.p1,
		product_2=args.p2,
		max_fewshot=args.max_fewshot,
		top_k=args.top_k,
		model_name=args.model,
		device=args.device,
		max_new_tokens=args.max_new_tokens,
		temperature=args.temperature,
		top_p=args.top_p,
	)
	print(json.dumps(result, ensure_ascii=False, indent=2))
	return 0


def cmd_build_nice(args: argparse.Namespace) -> int:
	tools_path = os.path.join(os.path.dirname(__file__), "tools", "merge_nice_cls.py")
	proc = subprocess.run([sys.executable, tools_path], check=False)
	return proc.returncode


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Product Similarity CLI")
	sub = parser.add_subparsers(dest="command", required=True)

	run_p = sub.add_parser("run", help="Run similarity for two product descriptions")
	run_p.add_argument("--p1", required=True, help="Product 1 description")
	run_p.add_argument("--p2", required=True, help="Product 2 description")
	run_p.add_argument("--max-fewshot", type=int, default=2)
	run_p.add_argument("--top-k", type=int, default=3)
	run_p.add_argument("--model", default=None, help="HF model id (e.g. google/flan-t5-base)")
	run_p.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 GPU")
	run_p.add_argument("--max-new-tokens", type=int, default=256)
	run_p.add_argument("--temperature", type=float, default=0.0)
	run_p.add_argument("--top-p", type=float, default=1.0)
	run_p.set_defaults(func=cmd_run)

	bn_p = sub.add_parser("build-nice", help="Build data/nice_chunks.json from data_nice_cls")
	bn_p.set_defaults(func=cmd_build_nice)

	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	return int(args.func(args))


if __name__ == "__main__":
	sys.exit(main())


