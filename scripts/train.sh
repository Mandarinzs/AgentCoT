#!/usr/bin/env bash
set -euo pipefail

python -m agentcot.cli.train \
  --model-config configs/model.yaml \
  --data-config configs/data.yaml \
  --train-config configs/train.yaml
