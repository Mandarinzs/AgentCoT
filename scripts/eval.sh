#!/usr/bin/env bash
set -euo pipefail

python -m agentcot.cli.eval \
  --model-config configs/model.yaml \
  --data-config configs/data.yaml \
  --eval-config configs/eval.yaml
