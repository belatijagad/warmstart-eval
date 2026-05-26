#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-belati/warmstart-results}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PRIVATE="${PRIVATE:-false}"

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Results directory not found: $RESULTS_DIR" >&2
  exit 1
fi

uv run - <<'PY'
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from huggingface_hub import HfApi
except Exception as exc:
    raise SystemExit(
        "huggingface_hub is required. Install with: python -m pip install huggingface_hub"
    ) from exc

repo_id = os.environ.get("REPO_ID", "belati/warmstart-results")
results_dir = Path(os.environ.get("RESULTS_DIR", "results")).resolve()
commit_msg = f"Upload results {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"

api = HfApi()
api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path=str(results_dir),
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo=".",
    commit_message=commit_msg,
)
PY

echo "Uploaded $RESULTS_DIR to https://huggingface.co/datasets/$REPO_ID"
