from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentcot.planning.loader import load_research_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Research planning bundle entrypoint")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root where configs/research is located",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save merged JSON bundle",
    )
    args = parser.parse_args()

    bundle = load_research_bundle(args.repo_root)
    serialized = json.dumps(bundle, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        print(f"[research-plan] saved bundle to {output_path}")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
