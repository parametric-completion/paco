#!/usr/bin/env bash

if [ "$DEBUG" = "1" ]; then
  set -x
fi

export LOCAL_RANK=0

python train.py "$@"

# Usage:
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_dp.sh
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_dp.sh exp_name=my_exp seed=1117
