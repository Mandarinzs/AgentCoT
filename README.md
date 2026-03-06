# AgentCoT

AgentCoT is a baseline framework for fairness-aware recommendation Q&A training/evaluation, built on top of `transformers + accelerate`.

## Project Structure

```text
.
├── configs/
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
{"query": "推荐一款适合夜跑的耳机", "answer": "...", "group": "female"}
```

Place your dataset under `data/` and update `configs/data.yaml`.

### 2) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 3) Training

```bash
bash scripts/train.sh
```

### 4) Evaluation

```bash
bash scripts/eval.sh
```

### 5) Reproducibility Commands (Example)

```bash
python -m agentcot.cli.train --model-config configs/model.yaml --data-config configs/data.yaml --train-config configs/train.yaml
python -m agentcot.cli.eval --model-config configs/model.yaml --data-config configs/data.yaml --eval-config configs/eval.yaml
pytest -q
```

## Notes

- `src/agentcot/data`: reading, splitting, caching.
- `src/agentcot/models`: HuggingFace/ModelScope-compatible local/remote model loading.
- `src/agentcot/fairness`: fairness metrics, regularization, and group reweighting.
- `src/agentcot/trainer`: minimal `accelerate`-based training/validation loop.
- `src/agentcot/eval`: recommendation QA quality + fairness joint metrics.
