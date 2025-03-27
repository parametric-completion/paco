#!/usr/bin/env bash

if [ "$DEBUG" = "1" ]; then
  set -x
fi

export LOCAL_RANK=0

python test.py "$@"

# Usage:
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/test.sh
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/test.sh dataset.mode=easy seed=1117
