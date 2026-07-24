from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def allocate_token_budget(
    *, total_tokens: int, weights: Mapping[str, float]
) -> dict[str, int]:
    """Allocate an integer token budget from fractional source weights."""
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if not weights:
        raise ValueError("weights must not be empty")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("weights must be nonnegative")

    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("at least one weight must be positive")

    raw = {
        source_id: total_tokens * (weight / weight_sum)
        for source_id, weight in weights.items()
    }
    allocations = {source_id: int(value) for source_id, value in raw.items()}
    remainder = total_tokens - sum(allocations.values())

    fractional_order = sorted(
        raw,
        key=lambda source_id: (raw[source_id] - allocations[source_id], source_id),
        reverse=True,
    )
    for source_id in fractional_order[:remainder]:
        allocations[source_id] += 1

    return allocations


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_blend_plan(manifest: dict) -> dict:
    enabled_sources = [
        source for source in manifest["sources"] if source.get("enabled", True)
    ]
    weights = {source["id"]: float(source["weight"]) for source in enabled_sources}
    allocations = allocate_token_budget(
        total_tokens=int(manifest["target_tokens"]),
        weights=weights,
    )
    return {
        "manifest_name": manifest["name"],
        "target_tokens": int(manifest["target_tokens"]),
        "tokenizer": manifest.get("tokenizer"),
        "sources": [
            {
                "id": source["id"],
                "target_tokens": allocations[source["id"]],
                "weight": float(source["weight"]),
                "hf_path": source.get("hf_path"),
                "hf_name": source.get("hf_name"),
                "hf_data_dir": source.get("hf_data_dir"),
                "hf_data_files": source.get("hf_data_files"),
                "hf_revision": source.get("hf_revision"),
                "hf_trust_remote_code": source.get("hf_trust_remote_code"),
                "local_path": source.get("local_path"),
                "split": source.get("split", "train"),
                "text_field": source.get("text_field", "text"),
                "download_shards": source.get("download_shards"),
                "notes": source.get("notes", ""),
            }
            for source in enabled_sources
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic blend plan.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    plan = build_blend_plan(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{manifest['name']}.blend-plan.json"
    output_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
