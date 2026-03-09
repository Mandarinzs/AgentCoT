# AgentCoT

AgentCoT is a baseline framework for fairness-aware recommendation Q&A training/evaluation, built on top of `transformers + accelerate`.

## Project Structure

```text
.
├── configs/
├── data/
├── scripts/
├── src/agentcot/
│   ├── data/
│   ├── models/
│   ├── fairness/
│   ├── trainer/
│   └── eval/
└── tests/
```

## Environment & Dependency Pins

Core dependencies are pinned in `pyproject.toml`.

### CUDA / PyTorch Compatibility Matrix

| PyTorch | CUDA runtime | Install hint |
|---|---|---|
| 2.2.2 | cu118 | `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118` |
| 2.2.2 | cu121 | `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121` |
| 2.2.2 | cpu | `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu` |

> Keep the rest of dependencies from `pyproject.toml` unchanged to ensure reproducibility.

## Quick Start

### 1) Data Preparation

Prepare JSONL data:

```json
{"query": "推荐一款适合夜跑的耳机", "answer": "...", "group": "female", "fairness_label": 1}
```

A minimal runnable example dataset is provided at `data/train.jsonl`.

### 2) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 3) Configure data/model addresses

- Dataset path: `configs/data.yaml` -> `train_file`
- Model path or model id: `configs/model.yaml` -> `model_name_or_path`
- Tokenizer path or model id: `configs/model.yaml` -> `tokenizer_name_or_path`

You can start from `configs/model.example.yaml` (uses a tiny local tokenizer/config for smoke tests). If no weight file exists in that local directory, AgentCoT will initialize model weights from config automatically.

### 4) Training (real training + checkpoint saving)

```bash
bash scripts/train.sh
```

Artifacts:
- `outputs/agentcot/train_metrics.json`
- `outputs/agentcot/checkpoint-final/`

### 5) Evaluation (real model eval + metrics)

```bash
bash scripts/eval.sh
```

Artifacts:
- `outputs/agentcot/eval_metrics.json`
- `outputs/agentcot/predictions.jsonl`

### 6) Reproducibility Commands (Example)

```bash
python -m agentcot.cli.train --model-config configs/model.yaml --data-config configs/data.yaml --train-config configs/train.yaml
python -m agentcot.cli.eval --model-config configs/model.yaml --data-config configs/data.yaml --eval-config configs/eval.yaml
pytest -q
```

## Server Deployment / Running

1. Clone repository and enter project root.
2. Create venv and install deps.
3. Edit config files (`configs/data.yaml`, `configs/model.yaml`).
4. Run `bash scripts/train.sh`, then `bash scripts/eval.sh`.
5. Check files under `outputs/agentcot/` for final metrics and predictions.

## Notes

- `src/agentcot/data`: reading, splitting, caching.
- `src/agentcot/models`: HuggingFace/ModelScope-compatible local/remote model loading.
- `src/agentcot/fairness`: fairness metrics, regularization, and group reweighting.
- `src/agentcot/trainer`: minimal `accelerate`-based training/validation loop.
- `src/agentcot/eval`: recommendation QA quality + fairness joint metrics.
