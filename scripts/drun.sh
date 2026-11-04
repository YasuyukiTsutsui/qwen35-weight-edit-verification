#!/usr/bin/env bash
# Run a command inside the GPU container with the project mounted at /work.
#   bash scripts/drun.sh python3 scripts/eval_math.py --model models/Qwen3.5-0.8B --tag qwen_base
# Safety rails: container RAM is capped (no swap) so that an overshoot kills the container, not the host;
# WSP_MEM overrides the cap (default 12g); a preflight check refuses to start when host RAM / GPU are short.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -W 2>/dev/null || pwd)"
MEM="${WSP_MEM:-12g}"
python "$(dirname "$0")/monitor.py" --check "${WSP_MIN_FREE_GB:-8}" "${WSP_MAX_GPU_MB:-2500}" || { echo "preflight failed; not starting"; exit 1; }
MSYS_NO_PATHCONV=1 docker run --rm --gpus all --ipc=host --shm-size=2g \
  --memory="$MEM" --memory-swap="$MEM" --pids-limit=2048 \
  -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=WARNING -e PYTHONUNBUFFERED=1 \
  -v "$ROOT:/work" -v wsp-hfcache:/root/.cache \
  wsp-vllm:latest "$@"
