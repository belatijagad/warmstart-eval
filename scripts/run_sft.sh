#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN in your environment}"

HF_USER="${HF_USER:-belati}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-$SCRIPT_DIR/sft_config.yaml}"

BASE_MODELS=(
  "Qwen/Qwen2.5-3B-Instruct"
)

SPLITS=(1a 2a full)

render_config() {
  local output_path="$1"

  python - "$CONFIG_TEMPLATE" "$output_path" <<'PY'
import os
import sys
from pathlib import Path

template_path, output_path = sys.argv[1], sys.argv[2]
template = Path(template_path).read_text()

required = ["BASE_MODEL", "DATASET_SPLIT", "OUTPUT_DIR", "HUB_MODEL_ID"]
missing = [name for name in required if not os.environ.get(name)]
if missing:
  raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

replacements = {
  "{{BASE_MODEL}}": os.environ["BASE_MODEL"],
  "{{DATASET_SPLIT}}": os.environ["DATASET_SPLIT"],
  "{{OUTPUT_DIR}}": os.environ["OUTPUT_DIR"],
  "{{HUB_MODEL_ID}}": os.environ["HUB_MODEL_ID"],
}

for key, value in replacements.items():
  template = template.replace(key, value)

Path(output_path).write_text(template)
PY
}

for base_model in "${BASE_MODELS[@]}"; do
  model_slug="${base_model##*/}"

  for split in "${SPLITS[@]}"; do
    run_name="${model_slug}-sft-${split}"
    adapter_repo="${HF_USER}/${model_slug}_multireasoner_sft-${split}"
    merged_repo="${adapter_repo}_merged"
    output_dir="./saved_models/${model_slug}_multireasoner_sft-${split}"

    export WANDB_PROJECT="${WANDB_PROJECT:-multireasoner-openr1}"
    export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-sft-phase}"
    export WANDB_NAME="${run_name}"

    if [[ -d "$output_dir" && -f "$output_dir/adapter_model.safetensors" ]]; then
      echo "Skipping SFT; found existing adapter in $output_dir"
    else
      echo "Starting SFT for ${model_slug} (${split})..."
      run_config="$(mktemp "/tmp/sft_config_${model_slug}_${split}_XXXX.yaml")"
      export BASE_MODEL="$base_model"
      export DATASET_SPLIT="$split"
      export OUTPUT_DIR="$output_dir"
      export HUB_MODEL_ID="$adapter_repo"

      render_config "$run_config"

      uv run trl sft --config "$run_config"

      echo "Merging adapter back into base model..."
      uv run scripts/merge.py \
        --base-model-id "$base_model" \
        --adapter-id "$adapter_repo" \
        --merged-repo-id "$merged_repo" \
        --hf-token "$HF_TOKEN"

      rm -f "$run_config"
    fi
  done
done