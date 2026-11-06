#!/usr/bin/env bash
# CPU-only test of scripts/train_lora_sft.py (no GPU, container RAM capped at 4 GB). See test_lora_trainer_cpu.py.
cd "$(dirname "$0")/.."
ROOT="$(pwd -W 2>/dev/null || pwd)"
MSYS_NO_PATHCONV=1 docker run --rm --ipc=host --shm-size=1g --memory=4g --memory-swap=4g --pids-limit=1024 \
  -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e CUDA_VISIBLE_DEVICES= \
  -v "$ROOT:/work" -w /work wsp-vllm:latest python3 scripts/test_lora_trainer_cpu.py
