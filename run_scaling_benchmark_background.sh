#!/usr/bin/env bash
set -euo pipefail

RUN_GROUP="${1:-sedrah-ddp-scale-$(date -u +%Y%m%d-%H%M%S)}"
LOG_DIR="logs/distributed_scaling"

mkdir -p "${LOG_DIR}"

nohup .venv/bin/python benchmark_distributed_scaling.py \
  --gpu-counts 1,2,4,8 \
  --target-effective-batch 8 \
  --epochs 1 \
  --run-group "${RUN_GROUP}" \
  > "${LOG_DIR}/${RUN_GROUP}.launcher.log" 2>&1 &

echo "$!" > "${LOG_DIR}/${RUN_GROUP}.pid"
echo "started run_group=${RUN_GROUP}"
echo "pid=$(cat "${LOG_DIR}/${RUN_GROUP}.pid")"
echo "launcher_log=${LOG_DIR}/${RUN_GROUP}.launcher.log"
