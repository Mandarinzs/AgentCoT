from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from agentcot.cli.common import dump_json, load_yaml
from agentcot.data.dataset import load_jsonl_dataset
from agentcot.data.splitter import split_dataset
from agentcot.eval.metrics import evaluate_joint_metrics
from agentcot.models.loader import load_tokenizer_and_model


class EvalDataset:
    def __init__(self, records: list[dict[str, str]], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, object]:
        item = self.records[idx]
        prompt = f"问题: {item.get('query', '')}\n答案: {item.get('answer', '')}"
        encoded = self.tokenizer(prompt, truncation=True, max_length=self.max_length)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.copy(),
            "query": item.get("query", ""),
            "answer": item.get("answer", ""),
            "group": item.get("group", "unknown"),
            "fairness_label": int(item.get("fairness_label", 1)),
        }


def collate_eval(batch: list[dict[str, object]], tokenizer) -> dict[str, object]:
    features = [
        {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
        }
        for item in batch
    ]
    labels = [item["labels"] for item in batch]

    model_batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    max_len = model_batch["input_ids"].shape[1]
    padded_labels = []
    for seq in labels:
        pad_len = max_len - len(seq)
        padded_labels.append(seq + ([-100] * pad_len))

    model_batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    model_batch["meta"] = {
        "query": [item["query"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "group": [item["group"] for item in batch],
        "fairness_label": [item["fairness_label"] for item in batch],
    }
    return model_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentCoT eval entrypoint")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--eval-config", required=True)
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)
    eval_cfg = load_yaml(args.eval_config)

    dataset = load_jsonl_dataset(data_cfg["train_file"])
    _, val_records, test_records = split_dataset(
        dataset.examples,
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=int(data_cfg.get("seed", 42)),
    )
    eval_records = test_records or val_records

    output_dir = Path(eval_cfg.get("output_dir", "outputs/agentcot"))
    checkpoint_dir = output_dir / "checkpoint-final"
    model_path = str(checkpoint_dir) if checkpoint_dir.exists() else model_cfg["model_name_or_path"]

    tokenizer_source = str(checkpoint_dir) if checkpoint_dir.exists() else model_cfg.get("tokenizer_name_or_path")
    tokenizer, model = load_tokenizer_and_model(
        model_name_or_path=model_path,
        tokenizer_name_or_path=tokenizer_source,
        provider=model_cfg.get("provider", "hf"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(eval_cfg.get("max_length", 256))
    batch_size = int(eval_cfg.get("batch_size", 4))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 32))

    eval_ds = EvalDataset(eval_records, tokenizer, max_length=max_length)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_eval(b, tokenizer),
    )

    model.eval()
    losses: list[float] = []
    qa_labels: list[list[str]] = []
    qa_preds: list[list[str]] = []
    fairness_preds: list[int] = []
    fairness_labels: list[int] = []
    groups: list[str] = []
    predictions: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in eval_loader:
            meta = batch.pop("meta")
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu().item()))

            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_only = generated[:, batch["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            for query, answer, pred_text, group, fair_label in zip(
                meta["query"],
                meta["answer"],
                decoded,
                meta["group"],
                meta["fairness_label"],
                strict=True,
            ):
                pred_text = pred_text.strip()
                qa_labels.append([answer])
                qa_preds.append([pred_text])
                fairness_pred = int(bool(pred_text))
                fairness_preds.append(fairness_pred)
                fairness_labels.append(int(fair_label))
                groups.append(group)
                predictions.append(
                    {
                        "query": query,
                        "answer": answer,
                        "prediction": pred_text,
                        "group": group,
                        "fairness_label": int(fair_label),
                        "fairness_pred": fairness_pred,
                    }
                )

    metrics = evaluate_joint_metrics(
        qa_labels=qa_labels,
        qa_preds=qa_preds,
        fairness_preds=fairness_preds,
        fairness_labels=fairness_labels,
        groups=groups,
    )
    metrics["eval_loss"] = float(sum(losses) / max(1, len(losses)))

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "eval_metrics.json", metrics)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[eval] finished. metrics saved to {output_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
