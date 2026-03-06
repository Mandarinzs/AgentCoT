from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentCoT train entrypoint")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--train-config", required=True)
    args = parser.parse_args()
    print(f"[train] model={args.model_config} data={args.data_config} train={args.train_config}")


if __name__ == "__main__":
    main()
