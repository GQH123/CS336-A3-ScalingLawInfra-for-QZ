from __future__ import annotations

import argparse
import json
from pathlib import Path

from scaling_data.blend import build_blend_plan, load_manifest


def write_pipeline_plan(
    *,
    manifest_path: Path,
    output_dir: Path,
    work_dir: Path | None = None,
) -> Path:
    manifest = load_manifest(manifest_path)
    work_root = Path("data_work") if work_dir is None else work_dir
    raw_dir = work_root / "raw_jsonl"
    processed_dir = work_root / "processed_jsonl"
    tokenized_dir = work_root / "tokenized"
    blend_plan = build_blend_plan(manifest)
    plan = {
        **blend_plan,
        "stages": [
            {
                "name": "blend",
                "output": str(output_dir / f"{manifest['name']}.blend-plan.json"),
            },
            {"name": "download", "output_dir": str(raw_dir)},
            {"name": "postprocess", "output_dir": str(processed_dir)},
            {"name": "tokenize", "output_dir": str(tokenized_dir)},
        ],
        "expected_outputs": {
            "raw_dir": str(raw_dir),
            "processed_train_jsonl": str(processed_dir / "train.jsonl.gz"),
            "processed_validation_jsonl": str(processed_dir / "validation.jsonl.gz"),
            "postprocess_stats": str(processed_dir / "postprocess-stats.json"),
            "tokenized_train_index": str(tokenized_dir / "train" / "index.json"),
            "tokenized_validation_index": str(
                tokenized_dir / "validation" / "index.json"
            ),
            "prepared_data_manifest": str(work_root / "prepared-data-manifest.json"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{manifest['name']}.pipeline-plan.json"
    output_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a dry-run pipeline plan for a data manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("data_work"))
    args = parser.parse_args()

    output_path = write_pipeline_plan(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
