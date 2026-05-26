#!/usr/bin/env bash
set -euo pipefail

HF_USER="${HF_USER:-belati}"
TASK="${TASK:-gsm8k|0,aime24_avg|0,aime25_avg|0}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MERGED_DIR="${MERGED_DIR:-merged}"

BASE_MODELS=(
  "Qwen/Qwen2.5-3B-Instruct"
)

SPLITS=(1a 2a full)

STEPS=(50 100 150 200 250 300)

declare -A REVISIONS_1A=(
  [50]=7a4ab2520df7f959a254d9151e2474ff612d86f9
  [100]=3d773647989f288dc477575dde434a735d55e0ad
  [150]=6aa8dd48e07d124f72a66e680775e7c4372ae5f6
  [200]=b8ccd32024b00a1a4876a5215e4b99acfddf0e78
  [250]=60884ef7fe33100f2105456dfeae94a8197a4af8
  [300]=29fcd985e759348ae785e759a259703e2fe43d7e
)
declare -A REVISIONS_2A=(
  [50]=12392defae1629e666123daabc6eb162f9edf8d6
  [100]=79cbbc21ef8b714b6bbe8d7c0513ff9da7aa6399
  [150]=9346a9a169273640a7184689370fdac7f783789d
  [200]=dba921c6028c84c002799be5734656b674e0119a
  [250]=056f9de5fd56b35196ee646569cac59a779ea75e
  [300]=7dea8f78b48369e3541ac00940feda748389349c
)
declare -A REVISIONS_FULL=(
  [50]=7f319bfe4cc30e7afb37410e3fd93a642f0a9bb1
  [100]=47ebadf668d8290f7a4958ea2bf462023a97291e
  [150]=606582dece61b84c14237ebce62391fcd52dba76
  [200]=a2595c70c2f021948a00a1b9b0d649f70f49e781
  [250]=77dce99cd54760f75b4d5538b30e303f1d3a9360
  [300]=2ac52e1b4cf14547ff024decba0c913e646f960e
)

get_revision() {
  local split="$1"
  local step="$2"
  case "$split" in
    1a) printf '%s' "${REVISIONS_1A[$step]-}" ;;
    2a) printf '%s' "${REVISIONS_2A[$step]-}" ;;
    full) printf '%s' "${REVISIONS_FULL[$step]-}" ;;
  esac
}

mkdir -p "$RESULTS_DIR" "$MERGED_DIR"

for base_model in "${BASE_MODELS[@]}"; do
  model_slug="${base_model##*/}"

  # 1. Evaluate Base Model Baseline
  # echo "=== Evaluating base model: $base_model ==="
  # uv run lighteval vllm \
  #   "model_name=${base_model},dtype=bfloat16,generation_parameters={temperature:0.7,top_p:0.95,max_new_tokens:8192}" \
  #   "$TASK" --compute-generation-entropy --save-generations --output-dir "$RESULTS_DIR"

  for split in "${SPLITS[@]}"; do
    adapter_repo="${HF_USER}/${model_slug}_multireasoner_sft-${split}"

    # 2. Evaluate specific adapter revisions from the Hub.
    has_revision=false
    for step in "${STEPS[@]}"; do
      revision="$(get_revision "$split" "$step")"
      [[ -n "$revision" ]] || continue
      has_revision=true

      merged_dir="${MERGED_DIR}/${model_slug}_${split}_s${step}"
      if [[ ! -d "$merged_dir" ]]; then
        echo "=== Merging adapter: ${adapter_repo}@${revision} (step ${step}) ==="
        uv run scripts/merge.py \
          --base-model-id "$base_model" \
          --adapter-id "$adapter_repo" \
          --adapter-revision "$revision" \
          --output-dir "$merged_dir" \
          --no-push
      else
        echo "=== Using cached merged model: $merged_dir ==="
      fi

      echo "=== Evaluating merged adapter: ${model_slug} (${split}) - step ${step} ==="
      uv run lighteval vllm \
        "model_name=${merged_dir},dtype=bfloat16,generation_parameters={temperature:0.7,top_p:0.95,max_new_tokens:8192}" \
        "$TASK" --compute-generation-entropy --save-generations --output-dir "$RESULTS_DIR"
    done

    if [[ "$has_revision" == "false" ]]; then
      echo "Warning: No revisions configured for split ${split}. Skipping adapter evaluations."
    fi

    # 3. Evaluate the final Trained & Merged Model
    merged_repo="${HF_USER}/${model_slug}_multireasoner_sft-${split}_merged"
    echo "=== Evaluating final merged model: $merged_repo ==="
    
    uv run lighteval vllm \
      "model_name=${merged_repo},dtype=bfloat16,generation_parameters={temperature:0.6,top_p:0.95,max_new_tokens:8192}" \
      "$TASK" --compute-generation-entropy --save-generations --output-dir "$RESULTS_DIR"
  done
done