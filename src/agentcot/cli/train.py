from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from torch.utils.data import DataLoader

from agentcot.cli.common import dump_json, ensure_dir, load_yaml
from agentcot.data.cache import DatasetCache
from agentcot.data.dataset import load_jsonl_dataset
from agentcot.data.splitter import split_dataset
from agentcot.models.loader import load_tokenizer_and_model
from agentcot.trainer.loop import TrainerConfig, run_training


class CausalLMDataset:
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
        }


def collate_causal_lm(batch: list[dict[str, object]], tokenizer) -> dict[str, object]:
    input_features = [
        {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
        }
        for item in batch
    ]
    labels = [item["labels"] for item in batch]

    model_batch = tokenizer.pad(input_features, padding=True, return_tensors="pt")
    max_len = model_batch["input_ids"].shape[1]
    padded_labels = []
    for seq in labels:
        pad_len = max_len - len(seq)
        padded_labels.append(seq + ([-100] * pad_len))

    import torch

    model_batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    return model_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentCoT train entrypoint")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--train-config", required=True)
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(args.train_config)

    output_dir = ensure_dir(train_cfg.get("output_dir", "outputs/agentcot"))

    dataset = load_jsonl_dataset(data_cfg["train_file"])
    train_records, val_records, test_records = split_dataset(
        dataset.examples,
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=int(data_cfg.get("seed", 42)),
    )

    cache = DatasetCache(data_cfg.get("cache_dir", ".cache/agentcot"))
    split_key = hashlib.md5(str(Path(data_cfg["train_file"]).resolve()).encode("utf-8")).hexdigest()[:12]
    split_cache_path = cache.save(
        f"split_{split_key}",
        {"train": train_records, "val": val_records, "test": test_records},
    )

    tokenizer, model = load_tokenizer_and_model(
        model_name_or_path=model_cfg["model_name_or_path"],
        tokenizer_name_or_path=model_cfg.get("tokenizer_name_or_path"),
        provider=model_cfg.get("provider", "hf"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(train_cfg.get("max_length", 256))
    batch_size = int(train_cfg.get("batch_size", 4))

    train_ds = CausalLMDataset(train_records, tokenizer, max_length=max_length)
    val_ds = CausalLMDataset(val_records or test_records, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_causal_lm(b, tokenizer),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_causal_lm(b, tokenizer),
    )

    metrics = run_training(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=TrainerConfig(
            learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
            num_epochs=int(train_cfg.get("num_epochs", 1)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        ),
    )

    checkpoint_dir = output_dir / "checkpoint-final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    payload = {
        "metrics": metrics,
        "num_train_examples": len(train_ds),
        "num_val_examples": len(val_ds),
        "split_cache": str(split_cache_path),
        "checkpoint": str(checkpoint_dir),
    }
    dump_json(output_dir / "train_metrics.json", payload)
    print(f"[train] finished. metrics saved to {output_dir / 'train_metrics.json'}")


if __name__ == "__main__":
    main()
