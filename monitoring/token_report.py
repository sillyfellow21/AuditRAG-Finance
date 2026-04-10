from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from core.config import get_settings


def main() -> None:
    settings = get_settings()
    usage_path = settings.data_dir / "token_usage.jsonl"
    if not usage_path.exists():
        print(f"No usage log found at {usage_path}")
        return

    totals = defaultdict(lambda: {"requests": 0, "prompt": 0, "completion": 0, "total": 0})

    with usage_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            model = row.get("model", "unknown")
            totals[model]["requests"] += 1
            totals[model]["prompt"] += int(row.get("prompt_tokens", 0))
            totals[model]["completion"] += int(row.get("completion_tokens", 0))
            totals[model]["total"] += int(row.get("total_tokens", 0))

    print("Token usage summary by model")
    print("=" * 48)
    for model, stats in totals.items():
        print(f"Model: {model}")
        print(f"  Requests: {stats['requests']}")
        print(f"  Prompt tokens: {stats['prompt']}")
        print(f"  Completion tokens: {stats['completion']}")
        print(f"  Total tokens: {stats['total']}")
        print("-" * 48)


if __name__ == "__main__":
    main()
