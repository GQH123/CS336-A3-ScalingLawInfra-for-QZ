# Compute-Node Data Contract

The tokenized dataset is mounted for compute-node workers. The control-node API
does not need these files locally.

Standard compute-node-visible paths:

```text
/mnt/course-data/tokenized/train/index.json
/mnt/course-data/tokenized/train/tokens-*.bin
/mnt/course-data/tokenized/validation/index.json
/mnt/course-data/tokenized/validation/tokens-*.bin
/mnt/course-data/tokenized/final-validation/index.json
/mnt/course-data/tokenized/final-validation/tokens-*.bin
```

Each `index.json` uses schema version 1:

```json
{
  "schema_version": 1,
  "token_dtype": "uint32",
  "byte_order": "little",
  "shard_format": "flat_binary_uint32_le",
  "total_tokens": 3,
  "source_token_counts": {"sample": 3},
  "shards": [
    {
      "path": "tokens-000000.bin",
      "tokens": 3,
      "dtype": "uint32",
      "byte_order": "little",
      "sha256": "64 lowercase hex characters",
      "source_token_counts": {"sample": 3}
    }
  ]
}
```

Shard paths are relative to the directory that contains `index.json`; absolute
paths and `..` path components are invalid. Workers should use metadata
validation for normal production startup and full validation only for offline
artifact audits.
