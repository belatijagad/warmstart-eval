import argparse
import os
import shutil
import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
  parser = argparse.ArgumentParser(description="Merge LoRA weights into base model.")
  parser.add_argument("--repo-path", required=True, help="Path to the base model weights")
  parser.add_argument("--adapter-path", required=True, help="Path to the LoRA adapter")
  parser.add_argument("--merged-path", required=True, help="Output directory for merged model")
  args = parser.parse_args()

  if os.path.exists(args.merged_path):
    shutil.rmtree(args.merged_path)
  os.makedirs(args.merged_path, exist_ok=True)

  print(f"Loading base model from {args.repo_path} in bfloat16...")
  base_model = AutoModelForCausalLM.from_pretrained(
    args.repo_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
  )

  print(f"Loading tokenizer from {args.adapter_path}...")
  tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)

  print(f"Applying LoRA adapter from {args.adapter_path}...")
  model = PeftModel.from_pretrained(base_model, args.adapter_path)
  
  print("Merging weights (merge_and_unload)...")
  merged_model = model.merge_and_unload()

  print(f"Saving merged model to {args.merged_path}...")
  merged_model.save_pretrained(
    args.merged_path,
    safe_serialization=True,
    max_shard_size="5GB",
  )
  tokenizer.save_pretrained(args.merged_path)
  print("Merge complete!")

if __name__ == "__main__":
  main()
