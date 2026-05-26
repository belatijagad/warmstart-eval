import argparse
import os

from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Merge a LoRA adapter into a base model and optionally push to the Hub."
  )
  parser.add_argument(
    "--base-model-id",
    default="Qwen/Qwen2.5-3B-Instruct",
    help="Base model id on the Hugging Face Hub.",
  )
  parser.add_argument(
    "--adapter-id",
    default="belati/Qwen2.5-3B-Instruct_multireasoner-u_sft1a",
    help="LoRA adapter id to merge.",
  )
  parser.add_argument(
    "--base-revision",
    default=None,
    help="Optional base model revision (commit SHA or tag).",
  )
  parser.add_argument(
    "--adapter-revision",
    default=None,
    help="Optional adapter revision (commit SHA or tag).",
  )
  parser.add_argument(
    "--merged-repo-id",
    default="belati/Qwen2.5-3B-Instruct_multireasoner-u_sft1a_merged",
    help="Target repo id for the merged model.",
  )
  parser.add_argument(
    "--output-dir",
    default=None,
    help="Local output directory to save the merged model.",
  )
  parser.add_argument(
    "--hf-token",
    default=None,
    help="Hugging Face token. Defaults to HF_TOKEN env var.",
  )
  parser.add_argument(
    "--no-push",
    action="store_true",
    help="Skip pushing the merged model to the Hub.",
  )
  return parser.parse_args()


def main() -> None:
  load_dotenv()
  args = parse_args()
  hf_token = args.hf_token or os.getenv("HF_TOKEN")

  base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_id, revision=args.base_revision
  )

  tokenizer = AutoTokenizer.from_pretrained(
    args.base_model_id, revision=args.base_revision
  )

  model = PeftModel.from_pretrained(
    base_model, args.adapter_id, revision=args.adapter_revision
  )

  print("Merging weights (this might take a minute)...")
  merged_model = model.merge_and_unload()

  if args.output_dir:
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)  # ty:ignore[unresolved-attribute]
    print(f"Saved merged model to {args.output_dir}.")

  if args.no_push:
    print("Skipping push to Hub.")
    return

  if not hf_token:
    raise ValueError("HF token not provided. Set HF_TOKEN or pass --hf-token.")

  print(f"Pushing full merged model to {args.merged_repo_id}...")
  merged_model.push_to_hub(args.merged_repo_id, token=hf_token)
  tokenizer.push_to_hub(args.merged_repo_id, token=hf_token)  # ty:ignore[unresolved-attribute]

  print("Merging complete!")


if __name__ == "__main__":
  main()