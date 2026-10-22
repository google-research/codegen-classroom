#!/usr/bin/env bash
# Starts a vLLM OpenAI-compatible server for codegen classroom students.
#
# Run it in its own conda env (vLLM pins its own torch, which conflicts with
# PyHealth), e.g.:
#   conda create -n vllm python=3.12 -y && conda activate vllm && pip install vllm
#   bash scripts/serve_vllm.sh
#
# Then point the classroom at it (default in configs/*_vllm.yaml):
#   llm.backend: vllm, llm.base_url: http://localhost:8000/v1
# and check it with:  curl http://localhost:8000/v1/models
#
# Settings (environment variables):
#   MODEL           HF model id or local path to serve.
#   GPUS            CUDA_VISIBLE_DEVICES for the server (keep student GPUs free).
#   TP              Tensor-parallel size (number of GPUs in GPUS).
#   PORT            Server port.
#   MAX_MODEL_LEN   Context length; must exceed prompt + llm.max_output_tokens.
#   GPU_MEM_UTIL    Fraction of each GPU's memory vLLM may use.
#   EXTRA_ARGS      Any extra `vllm serve` arguments.
#
# Pre-quantized bitsandbytes checkpoints (e.g. unsloth/*-bnb-4bit) are detected
# from their config. Text-only use of a multimodal checkpoint can skip the
# vision encoder's memory with EXTRA_ARGS='--limit-mm-per-prompt {"image":0,"video":0}'.
# Example: the 4-bit Qwen3.8-27B checkpoint is ~21 GB (vision tower not
# quantized), so it needs roughly 25-30 GB with KV cache, e.g. one free A6000:
#   MODEL=unsloth/Qwen3.8-27B-unsloth-bnb-4bit GPUS=0 MAX_MODEL_LEN=32768 \
#     bash scripts/serve_vllm.sh
# GPU_MEM_UTIL is a fraction of TOTAL memory and must fit in what is free.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
GPUS="${GPUS:-0}"
TP="${TP:-1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Serving ${MODEL} on GPU(s) ${GPUS} (tp=${TP}) at http://localhost:${PORT}/v1"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${GPUS}" exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  ${EXTRA_ARGS}
