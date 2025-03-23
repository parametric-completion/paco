#!/usr/bin/env bash

if [ "$DEBUG" = "1" ]; then
  set -x
fi

# Get the number of GPUs from CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # If CUDA_VISIBLE_DEVICES is not set, assume all GPUs are available
    NGPUS=$(nvidia-smi --list-gpus | wc -l)
else
    # Count GPUs in CUDA_VISIBLE_DEVICES by counting commas and adding 1
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c)
    NGPUS=$((NGPUS + 1))
fi

PORT=${1:-29500}  # Use provided port or default to 29500

echo "Using $NGPUS GPUs"

# Pass all remaining arguments to train.py via torchrun
shift 1  # Remove the PORT argument
torchrun --master_port=${PORT} --nproc_per_node=${NGPUS} train.py distributed=true "$@"

# Usage:
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh 29500
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_ddp.sh 29500 exp_name=my_exp seed=1117
# ./scripts/train_ddp.sh  # Uses all GPUs and default port 29500
