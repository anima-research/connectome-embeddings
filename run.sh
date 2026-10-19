#!/usr/bin/env bash
# launchd entry point: source the token, run the server (system python3 has torch/MPS + sentence-transformers + peft).
set -euo pipefail; cd "$(dirname "$0")"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
export HF_HOME="$HOME/embed-service/hf-home" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
set -a; [ -f .env ] && . ./.env; set +a
exec /usr/bin/python3 server.py --port 8804 --modality text "$@"
