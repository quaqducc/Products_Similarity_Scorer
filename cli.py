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
		class_1=args.class1,
		class_2=args.class2,
		max_fewshot=args.max_fewshot,
		top_k=args.top_k,
		include_spsc=(not args.no_spsc),
		spsc_top_k=args.spsc_top_k,
		model_name=args.model,
		chat_api_base_url=args.chat_api_base_url,
		chat_api_key=args.chat_api_key,
		chat_api_model=args.chat_api_model,
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


def cmd_build_tree(args: argparse.Namespace) -> int:
	tools_path = os.path.join(os.path.dirname(__file__), "tools", "build_tree_from_excel.py")
	cmd = [sys.executable, tools_path]
	if args.input:
		cmd += ["--input", args.input]
	if args.sheet_name:
		cmd += ["--sheet-name", args.sheet_name]
	if args.output:
		cmd += ["--output", args.output]
	if args.key_col:
		cmd += ["--key-col", args.key_col]
	if args.parent_col:
		cmd += ["--parent-col", args.parent_col]
	if args.code_col:
		cmd += ["--code-col", args.code_col]
	if args.title_col:
		cmd += ["--title-col", args.title_col]
	proc = subprocess.run(cmd, check=False)
	return proc.returncode


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Product Similarity CLI")
	sub = parser.add_subparsers(dest="command", required=True)

	run_p = sub.add_parser("run", help="Run similarity for two product descriptions")
	run_p.add_argument("--p1", required=True, help="Product 1 description")
	run_p.add_argument("--p2", required=True, help="Product 2 description")
	run_p.add_argument("--class1", default=None, help="NICE class number for product 1 (optional)")
	run_p.add_argument("--class2", default=None, help="NICE class number for product 2 (optional)")
	run_p.add_argument("--max-fewshot", type=int, default=2)
	run_p.add_argument("--top-k", type=int, default=3)
	run_p.add_argument("--spsc-top-k", type=int, default=2, help="Top SPSC contexts to include")
	run_p.add_argument("--no-spsc", action="store_true", help="Disable adding SPSC context")
	run_p.add_argument("--model", default=None, help="HF model id (e.g. google/flan-t5-base)")
	run_p.add_argument("--chat-api-base-url", default=None, help="OpenAI-compatible chat API base URL")
	run_p.add_argument("--chat-api-key", default=None, help="OpenAI-compatible chat API key")
	run_p.add_argument("--chat-api-model", default=None, help="OpenAI-compatible chat API model id")
	run_p.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 GPU")
	run_p.add_argument("--max-new-tokens", type=int, default=256)
	run_p.add_argument("--temperature", type=float, default=0.0)
	run_p.add_argument("--top-p", type=float, default=1.0)
	run_p.set_defaults(func=cmd_run)

	bn_p = sub.add_parser("build-nice", help="Build data/nice_chunks.json from data_nice_cls")
	bn_p.set_defaults(func=cmd_build_nice)

	bt_p = sub.add_parser("build-tree", help="Build hierarchy tree (JSON) from an Excel file")
	bt_p.add_argument("--input", help="Path to Excel file")
	bt_p.add_argument("--sheet-name", help="Excel sheet name (default: first sheet)")
	bt_p.add_argument("--output", help="Output JSON path")
	bt_p.add_argument("--key-col", help="Column name for Key (default: 'Key')")
	bt_p.add_argument("--parent-col", help="Column name for Parent key (default: 'Parent key')")
	bt_p.add_argument("--code-col", help="Column name for Code (default: 'Code')")
	bt_p.add_argument("--title-col", help="Column name for Title (default: 'Title')")
	bt_p.set_defaults(func=cmd_build_tree)

	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	return int(args.func(args))


if __name__ == "__main__":
	sys.exit(main())


