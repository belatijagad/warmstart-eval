#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN in your environment}"

HF_USER="${HF_USER:-belati}"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-qwen25-3b}"
HF_PATH_PREFIX="${HF_PATH_PREFIX:-runs}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
TASK="${TASK:-aime24_avg|0,aime25_avg|0,gsm8k|0}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MERGED_DIR="${MERGED_DIR:-${RESULTS_DIR}/merged}"

SPLITS=(1a 2a full)
CHECKPOINT_STEPS=(15 30 45)

mkdir -p "$RESULTS_DIR"
mkdir -p "$MERGED_DIR"

resolve_model_path() {
  local repo_path="$1"
  local subfolder="$2"

  if [[ -f "${repo_path}/${subfolder}/config.json" ]]; then
    echo "${repo_path}/${subfolder}"
  else
    echo "$repo_path"
  fi
}

resolve_adapter_path() {
  local repo_path="$1"
  local subfolder="$2"

  if [[ -d "${repo_path}/${subfolder}/lora_adapter" ]]; then
    echo "${repo_path}/${subfolder}/lora_adapter"
  else
    echo "${repo_path}/lora_adapter"
  fi
}

for split in "${SPLITS[@]}"; do
  repo_id="${HF_USER}/${HF_REPO_PREFIX}-${split}"
  repo_slug="${HF_REPO_PREFIX}-${split}"

  for step in "${CHECKPOINT_STEPS[@]}"; do
    subfolder="${HF_PATH_PREFIX}/step-${step}"
    run_name="${repo_slug}-step-${step}"
    output_dir="${RESULTS_DIR}/${run_name}"
    merged_path="${MERGED_DIR}/${run_name}"
    mkdir -p "$output_dir"

    echo "=== Processing ${run_name} ==="

    repo_path=$(uv run huggingface-cli download "$repo_id" --repo-type model)
    
    model_path="$(resolve_model_path "$repo_path" "$subfolder")"
    lora_path="$(resolve_adapter_path "$repo_path" "$subfolder")"

    uv run python merge_lora.py \
      --repo-path "$model_path" \
      --adapter-path "$lora_path" \
      --merged-path "$merged_path"

    echo "Running lighteval on ${merged_path}..."
    uv run lighteval vllm \
      "model_name=${merged_path},dtype=bfloat16,generation_parameters={temperature:0.7,top_p:0.95,max_new_tokens:8192}" \
      "$TASK" --compute-generation-entropy --save-generations --output-dir "$output_dir"
  done
done