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

---

## 详细部署与数据全流程指南（中文）

下面这部分面向“从 0 到 1 落地”的场景，覆盖：环境部署、数据集新增、数据预处理/切分/缓存、训练与评估联调、常见问题排查。

### 1. 部署前准备（机器、系统、目录）

#### 1.1 硬件与系统建议

- **GPU 训练（推荐）**：NVIDIA GPU + 对应 CUDA 驱动。
- **CPU 验证（可行）**：可用于流程打通和单元测试，但训练速度较慢。
- **Python 版本**：建议使用 `python3.10+`（与当前项目依赖更稳妥）。

#### 1.2 获取代码并进入项目

```bash
git clone <your-repo-url> AgentCoT
cd AgentCoT
```

#### 1.3 推荐目录布局

```text
AgentCoT/
├── configs/
├── data/                  # 你的数据集目录（建议）
│   ├── train.jsonl        # 默认配置中的训练入口文件
│   └── ...
├── outputs/               # 训练输出（checkpoint、日志）
├── .cache/agentcot/       # 数据缓存目录（由代码自动创建）
└── src/
```

---

### 2. 环境部署（一步一步）

#### 2.1 创建虚拟环境并安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

说明：
- `-e .[dev]` 会以可编辑模式安装项目，同时安装开发依赖（例如测试工具）。

#### 2.2 安装匹配的 PyTorch 版本

请先确认你的 CUDA 版本，再选下面一条：

```bash
# CUDA 11.8
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 纯 CPU
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```

建议：先安装 torch，再执行一次 `pip install -e .[dev]`，确保依赖一致。

#### 2.3 快速自检

```bash
python -c "import torch; print(torch.__version__)"
python -c "import agentcot; print('agentcot ok')"
pytest -q
```

---

### 3. 如何添加数据集（最关键）

项目默认读取 **JSONL（每行一个 JSON 对象）**，并在 `configs/data.yaml` 中指定入口文件。

#### 3.1 数据格式要求

最低建议字段：

```json
{"query": "推荐一款适合夜跑的耳机", "answer": "...", "group": "female"}
```

字段建议：
- `query`：用户问题（推荐问答场景中的输入）。
- `answer`：参考答案或目标输出。
- `group`：公平性分组标签（如性别、年龄段、地区等），用于公平性统计。

> 当前 `load_jsonl_dataset` 仅做逐行 JSON 解析，不会自动校验字段完整性；建议你在入库前自行校验。 

#### 3.2 新增一个数据集文件

1. 在 `data/` 下新建文件，例如：`data/my_domain_v1.jsonl`。
2. 确保 UTF-8 编码、每行合法 JSON、无多余逗号。
3. 用下面命令抽检：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('data/my_domain_v1.jsonl')
for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
    if not line.strip():
        continue
    json.loads(line)
print('jsonl format ok')
PY
```

#### 3.3 更新数据配置

编辑 `configs/data.yaml`：

```yaml
train_file: data/my_domain_v1.jsonl
cache_dir: .cache/agentcot
train_ratio: 0.8
val_ratio: 0.1
seed: 42
```

参数解释：
- `train_file`：数据入口（当前训练/评估都会依赖该配置）。
- `cache_dir`：缓存目录，`DatasetCache` 会将处理结果序列化为 `pkl`。
- `train_ratio` / `val_ratio`：随机切分比例；测试集比例 = `1 - train_ratio - val_ratio`。
- `seed`：随机种子，确保切分可复现。

#### 3.4 多数据集管理建议

- 推荐按版本命名：`data/reco_qa_2026q1.jsonl`。
- 为每个数据集保存一份独立配置：
  - `configs/data.reco_qa_2026q1.yaml`
  - `configs/data.reco_qa_2026q2.yaml`
- 使用命令行切换配置，避免互相覆盖。

---

### 4. 如何处理数据集（读取、切分、缓存、质量控制）

#### 4.1 读取逻辑

`src/agentcot/data/dataset.py` 的 `load_jsonl_dataset(path)` 会：
- 逐行读取文件。
- 跳过空行。
- 对每行调用 `json.loads`。
- 返回 `RecommendationQADataset(examples=list[dict])`。

#### 4.2 切分逻辑

`src/agentcot/data/splitter.py` 的 `split_dataset(...)` 会：
- 先用固定随机种子打乱数据。
- 按比例切成 `train / val / test`。
- 对非法比例做参数检查：
  - `train_ratio` 必须在 `(0,1)`。
  - `val_ratio` 必须在 `[0,1)`。
  - `train_ratio + val_ratio < 1`。

#### 4.3 缓存逻辑

`src/agentcot/data/cache.py` 的 `DatasetCache`：
- `save(key, payload)`：保存到 `<cache_dir>/<key>.pkl`。
- `load(key)`：读取缓存，不存在会抛 `FileNotFoundError`。

适用场景：
- 文本清洗后中间结果缓存。
- tokenizer 后的样本缓存。
- 多次试验复用同一处理结果，减少重复耗时。

#### 4.4 建议的数据处理流水线（实践版）

建议你按以下顺序构建自己的 preprocessing 脚本：

1. **格式检查**：JSONL 可解析、字段完整。
2. **内容清洗**：去除无意义文本、重复样本、异常超长样本。
3. **标签规范**：`group` 值统一命名（例如都小写）。
4. **分布检查**：统计各组样本量，观察是否极端不平衡。
5. **切分数据**：使用固定 seed，保持可复现。
6. **写入缓存**：将清洗后的结构缓存到 `.cache/agentcot`。
7. **记录版本**：保存脚本版本、数据哈希、配置快照。

---

### 5. 训练部署流程（本地/服务器通用）

#### 5.1 配置模型

编辑 `configs/model.yaml`：

```yaml
provider: hf
model_name_or_path: /path/to/local-or-hf-model
tokenizer_name_or_path: /path/to/local-or-hf-model
trust_remote_code: true
```

说明：
- `provider` 当前支持 `hf` 与 `modelscope`（底层走 HuggingFace 兼容加载）。
- `model_name_or_path` 可填本地路径或远程模型名。
- 若模型包含自定义代码，保留 `trust_remote_code: true`。

#### 5.2 配置训练参数

编辑 `configs/train.yaml`：

```yaml
learning_rate: 2.0e-5
num_epochs: 3
batch_size: 4
max_grad_norm: 1.0
output_dir: outputs/agentcot
```

调参建议：
- 显存不足：先降 `batch_size`。
- 收敛慢：可小幅提高 `num_epochs` 或调整学习率。
- 梯度不稳定：适当降低 `learning_rate`，保持 `max_grad_norm`。

#### 5.3 启动训练

```bash
bash scripts/train.sh
```

或显式指定：

```bash
python -m agentcot.cli.train \
  --model-config configs/model.yaml \
  --data-config configs/data.yaml \
  --train-config configs/train.yaml
```

---

### 6. 评估与公平性处理

#### 6.1 评估配置

`configs/eval.yaml` 示例：

```yaml
k: 5
compute_fairness: true
metrics:
  - hit_rate@5
  - demographic_parity_gap
  - equal_opportunity_gap
```

#### 6.2 指标说明

- `hit_rate@5`：推荐候选前 5 命中率。
- `demographic_parity_gap`：不同组正例预测率差距（越小越公平）。
- `equal_opportunity_gap`：不同组真阳性率差距（越小越公平）。

#### 6.3 公平性处理模块怎么用

项目提供了三类常见能力：

1. **评估指标**：`src/agentcot/fairness/metrics.py`
2. **重加权**：`src/agentcot/fairness/reweighting.py` 的 `compute_group_weights`
3. **正则项**：`src/agentcot/fairness/regularization.py` 的 `fairness_regularization_loss`

你可以在训练循环中把这些模块接入 loss 计算：
- 基础 loss + 公平性正则 loss。
- 或者用 group weight 对样本 loss 加权。

---

### 7. 常见问题与排查

#### 7.1 JSONL 报错（`json.decoder.JSONDecodeError`）

- 常见原因：某一行不是合法 JSON。
- 解决：写一个逐行校验脚本定位坏行号。

#### 7.2 切分比例报错（`train_ratio + val_ratio must be < 1`）

- 说明训练集+验证集占比不能达到 1。
- 例如可设为 `0.8 + 0.1`，留出 0.1 测试集。

#### 7.3 缓存读取失败（`Cache entry not found`）

- 说明对应 key 的缓存文件不存在。
- 先执行一次 `save`，或检查 `cache_dir` 是否变更。

#### 7.4 模型加载失败

- 检查 `model_name_or_path` 路径是否存在。
- 若使用远程模型名，确认网络和鉴权。
- 若模型有自定义代码，保持 `trust_remote_code: true`。

---

### 8. 推荐的“生产可复现”操作清单

每次实验建议固定以下信息：

1. Git commit hash。
2. `configs/model.yaml`、`configs/data.yaml`、`configs/train.yaml`、`configs/eval.yaml` 快照。
3. 数据文件版本号（或哈希）。
4. 训练命令与评估命令。
5. 关键指标（质量 + 公平性）与运行环境（GPU 型号、torch 版本）。

这能显著降低“同代码不同结果”的排查成本。
