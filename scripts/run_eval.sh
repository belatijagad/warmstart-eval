#!/usr/bin/env bash
set -euo pipefail

HF_USER="${HF_USER:-belati}"
# Restricted to gsm8k and aime24_avg only
TASK="${TASK:-gsm8k|0,aime24_avg|0}"
RESULTS_DIR="${RESULTS_DIR:-results}"

BASE_MODELS=(
  "Qwen/Qwen2.5-3B-Instruct"
)

SPLITS=(1a 2a full)

mkdir -p "$RESULTS_DIR"

for base_model in "${BASE_MODELS[@]}"; do
  model_slug="${base_model##*/}"

  # 1. Evaluate Base Model Baseline
  echo "=== Evaluating base model: $base_model ==="
  uv run lighteval vllm \
    "model_name=${base_model},dtype=bfloat16,generation_parameters={temperature:0.7,top_p:0.95,max_new_tokens:8192}" \
    "$TASK" --compute-generation-entropy --save-generations --output-dir "$RESULTS_DIR"

  for split in "${SPLITS[@]}"; do
    output_dir="./saved_models/${model_slug}_multireasoner_sft-${split}"
    
    # 2. Evaluate intermediate checkpoints (LoRA adapters)
    if [[ -d "$output_dir" ]]; then
      # Find all subdirectories starting with 'checkpoint-'
      for ckpt_dir in "$output_dir"/checkpoint-*/; do
        # Ensure the directory actually exists and isn't an empty glob match
        [[ -d "$ckpt_dir" ]] || continue
        
        ckpt_name=$(basename "$ckpt_dir")
        echo "=== Evaluating Checkpoint: ${model_slug} (${split}) - ${ckpt_name} ==="
        
        # We pass the base model to vllm, and instruct lighteval to overlay the local adapter
        uv run lighteval vllm \
          "model_name=${base_model},adapter_name=${ckpt_dir},dtype=bfloat16,generation_parameters={temperature:0.7,top_p:0.95,max_new_tokens:8192}" \
          "$TASK" --compute-generation-entropy --save-generations --output-dir "$RESULTS_DIR"
      done
    else
      echo "Warning: Local output directory $output_dir not found. Skipping intermediate checkpoints."
    fi

    # 3. Evaluate the final Trained & Merged Model
    merged_repo="${HF_USER}/${model_slug}_multireasoner_sft-${split}_merged"
    echo "=== Evaluating final merged model: $merged_repo ==="
    
    uv run lighteval vllm \
      "model_name=${merged_repo},dtype=bfloat16,generation_parameters={temperature:0.7,top_p:0.95,max_new_tokens:8192}" \
      "$TASK" --compute-generation-entropy --save-generations --output-dir "$RESULTS_DIR"
  done
done