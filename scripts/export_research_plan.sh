#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:-artifacts/research/research_bundle.json}"
export PYTHONPATH="${PYTHONPATH:-}:src"
python -m agentcot.cli.research_plan --repo-root . --output "$OUTPUT_PATH"
