from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentCoT eval entrypoint")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--eval-config", required=True)
    args = parser.parse_args()
    print(f"[eval] model={args.model_config} data={args.data_config} eval={args.eval_config}")


if __name__ == "__main__":
    main()
