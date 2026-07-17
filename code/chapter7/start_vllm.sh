#!/usr/bin/env bash
# Working vLLM OpenAI API server for macOS Apple Silicon (CPU).
# Root cause of previous hang: OpenMP deadlock during warmup.
# Fix: force single OpenMP thread + bind to one CPU core.

set -euo pipefail

export VLLM_CPU_KVCACHE_SPACE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export VLLM_CPU_OMP_THREADS_BIND=0

exec python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen1.5-0.5B-Chat \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 512 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 128 \
    --gpu-memory-utilization 0.12 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --disable-log-stats
