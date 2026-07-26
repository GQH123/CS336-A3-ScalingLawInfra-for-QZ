import unittest
import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import types
from array import array
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from scaling_data import download as download_module
from scaling_data import prepare as prepare_module
from scaling_data import tokenize as tokenize_module
from scaling_data.blend import allocate_token_budget
from scaling_data.download import download_source
from scaling_data.io import iter_jsonl_records, open_text_writer, write_jsonl_record
from scaling_data.lifecycle_dataset import (
    LifecycleDatasetError,
    package_lifecycle_dataset,
)
from scaling_data.pipeline import write_pipeline_plan
from scaling_data.postprocess import assign_split, postprocess_directory
from scaling_data.prepare import PrepareDataConfig, prepare_data
from scaling_data.rebuild_tokenized import (
    RebuildTokenizedConfig,
    rebuild_tokenized_from_processed,
)
from scaling_data.shuffle_processed import (
    resolve_processed_shuffle_max_open_buckets,
    shuffle_processed_splits,
)
from scaling_data.text_processing import clean_text, document_hash, should_keep_text
from scaling_data.shuffle_tokenized import shuffle_tokenized_corpus
from scaling_data.tokenize import (
    TokenShardWriter,
    TokenizedIndexError,
    load_tokenized_index,
    read_token_shard,
    tokenize_records_by_split,
)


class _FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [len(part) for part in text.split()]


class _BatchFakeTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.batch_calls = 0
        self.encode_calls = 0

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        self.batch_calls += 1
        return {
            "input_ids": [
                [len(part) for part in text.split()]
                for text in texts
            ]
        }

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        return [len(part) for part in text.split()]


def _write_uint32_token_shard(path: Path, tokens: list[int]) -> None:
    values = array("I", tokens)
    if sys.byteorder != "little":
        values.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        values.tofile(handle)


class BlendAllocationTests(unittest.TestCase):
    def test_allocates_integer_tokens_from_weights(self):
        allocations = allocate_token_budget(
            total_tokens=100,
            weights={
                "general_web": 0.55,
                "education": 0.20,
                "long_form": 0.10,
                "math": 0.07,
                "code": 0.06,
                "multilingual": 0.02,
            },
        )

        self.assertEqual(sum(allocations.values()), 100)
        self.assertEqual(allocations["general_web"], 55)
        self.assertEqual(allocations["education"], 20)
        self.assertEqual(allocations["multilingual"], 2)

    def test_allocates_rounding_remainder_to_largest_fractional_parts(self):
        allocations = allocate_token_budget(
            total_tokens=10,
            weights={"a": 0.333, "b": 0.333, "c": 0.334},
        )

        self.assertEqual(sum(allocations.values()), 10)
        self.assertEqual(allocations["c"], 4)

    def test_blend_plan_preserves_huggingface_loader_options(self):
        from scaling_data.blend import build_blend_plan

        plan = build_blend_plan(
            {
                "name": "loader_options_mix",
                "target_tokens": 10,
                "sources": [
                    {
                        "id": "remote",
                        "weight": 1.0,
                        "hf_path": "org/repo",
                        "hf_name": "config",
                        "hf_data_dir": "data/subset",
                        "hf_data_files": ["train/*.parquet"],
                        "hf_revision": "abc123",
                        "hf_trust_remote_code": True,
                        "text_field": "text",
                    }
                ],
            }
        )

        source = plan["sources"][0]
        self.assertEqual(source["hf_data_dir"], "data/subset")
        self.assertEqual(source["hf_data_files"], ["train/*.parquet"])
        self.assertEqual(source["hf_revision"], "abc123")
        self.assertTrue(source["hf_trust_remote_code"])


class TextProcessingTests(unittest.TestCase):
    def test_clean_text_normalizes_whitespace(self):
        self.assertEqual(clean_text(" alpha\n\n beta\t gamma "), "alpha beta gamma")

    def test_should_keep_text_rejects_short_or_duplicate_text(self):
        seen = set()

        self.assertFalse(should_keep_text("too short", seen_hashes=seen, min_chars=20))
        self.assertTrue(
            should_keep_text(
                "This document is long enough to keep.",
                seen_hashes=seen,
                min_chars=20,
            )
        )
        self.assertFalse(
            should_keep_text(
                "This document is long enough to keep.",
                seen_hashes=seen,
                min_chars=20,
            )
        )


class PostprocessTests(unittest.TestCase):
    def test_assign_split_is_deterministic_and_respects_validation_rate(self):
        text = "This is a stable document."

        self.assertEqual(assign_split(text, validation_rate=0.0), "train")
        self.assertEqual(assign_split(text, validation_rate=1.0), "validation")
        self.assertEqual(
            assign_split(text, validation_rate=0.15),
            assign_split(text, validation_rate=0.15),
        )

    def test_postprocess_reuses_trusted_download_hashes(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "raw"
            input_dir.mkdir()
            cleaned = "Alpha Beta Gamma"
            digest = document_hash(cleaned)
            with (input_dir / "source.jsonl").open("w", encoding="utf-8") as handle:
                write_jsonl_record(
                    handle,
                    {
                        "text": cleaned,
                        "source_id": "local_sample",
                        "document_hash": digest,
                        "text_preprocessed": True,
                    },
                )

            with patch(
                "scaling_data.postprocess.clean_text",
                side_effect=AssertionError("clean_text should be skipped"),
            ):
                stats = postprocess_directory(
                    input_dir=input_dir,
                    output_dir=root / "processed",
                    validation_rate=0.0,
                    min_chars=5,
                )

            records = list(iter_jsonl_records(root / "processed" / "train.jsonl.gz"))
            self.assertEqual(stats["trusted_preprocessed_documents"], 1)
            self.assertEqual(records[0]["text"], cleaned)
            self.assertEqual(records[0]["document_hash"], digest)

    def test_postprocess_parallel_workers_preserve_global_dedupe_order(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "raw"
            input_dir.mkdir()
            with (input_dir / "a.jsonl").open("w", encoding="utf-8") as handle:
                write_jsonl_record(handle, {"text": "alpha beta gamma"})
                write_jsonl_record(handle, {"text": "delta epsilon zeta"})
            with (input_dir / "b.jsonl").open("w", encoding="utf-8") as handle:
                write_jsonl_record(handle, {"text": "alpha beta gamma"})
                write_jsonl_record(handle, {"text": "eta theta iota"})

            stats = postprocess_directory(
                input_dir=input_dir,
                output_dir=root / "processed",
                validation_rate=0.0,
                min_chars=5,
                workers=2,
            )

            records = list(iter_jsonl_records(root / "processed" / "train.jsonl.gz"))
            self.assertEqual(stats["input_documents"], 4)
            self.assertEqual(stats["kept_train_documents"], 3)
            self.assertEqual(stats["dropped_duplicate_documents"], 1)
            self.assertEqual(
                [record["text"] for record in records],
                [
                    "alpha beta gamma",
                    "delta epsilon zeta",
                    "eta theta iota",
                ],
            )


class ProcessedShuffleTests(unittest.TestCase):
    def test_shuffle_processed_splits_preserves_whole_records_by_split(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "processed"
            input_dir.mkdir()
            train_records = [
                {
                    "text": f"train document {index}",
                    "source_id": "code" if index % 2 else "math",
                    "split": "train",
                    "document_hash": f"train-{index}",
                }
                for index in range(12)
            ]
            validation_records = [
                {
                    "text": f"validation document {index}",
                    "source_id": "general_web",
                    "split": "validation",
                    "document_hash": f"validation-{index}",
                }
                for index in range(7)
            ]
            with open_text_writer(input_dir / "train.jsonl.gz") as handle:
                for record in train_records:
                    write_jsonl_record(handle, record)
            with open_text_writer(input_dir / "validation.jsonl.gz") as handle:
                for record in validation_records:
                    write_jsonl_record(handle, record)

            first_output = root / "shuffled-a"
            second_output = root / "shuffled-b"
            stats = shuffle_processed_splits(
                input_dir=input_dir,
                output_dir=first_output,
                seed=17,
                bucket_count=3,
                max_open_buckets=1,
            )
            shuffle_processed_splits(
                input_dir=input_dir,
                output_dir=second_output,
                seed=17,
                bucket_count=3,
                max_open_buckets=1,
            )

            shuffled_train = list(iter_jsonl_records(first_output / "train.jsonl.gz"))
            shuffled_validation = list(
                iter_jsonl_records(first_output / "validation.jsonl.gz")
            )
            self.assertEqual(
                [record["text"] for record in shuffled_train],
                [
                    record["text"]
                    for record in iter_jsonl_records(second_output / "train.jsonl.gz")
                ],
            )
            self.assertCountEqual(
                [record["document_hash"] for record in shuffled_train],
                [record["document_hash"] for record in train_records],
            )
            self.assertCountEqual(
                [record["document_hash"] for record in shuffled_validation],
                [record["document_hash"] for record in validation_records],
            )
            self.assertTrue(all(record["split"] == "train" for record in shuffled_train))
            self.assertTrue(
                all(record["split"] == "validation" for record in shuffled_validation)
            )
            self.assertEqual(
                stats["splits"]["train"]["source_document_counts"],
                {"code": 6, "math": 6},
            )
            self.assertEqual(
                stats["splits"]["validation"]["source_document_counts"],
                {"general_web": 7},
            )
            self.assertEqual(stats["requested_max_open_buckets"], 1)
            self.assertEqual(stats["max_open_buckets"], 1)
            self.assertIn("train", stats["input_files"])
            self.assertFalse(stats["reused"])
            self.assertNotEqual(
                [record["document_hash"] for record in shuffled_train],
                [record["document_hash"] for record in train_records],
            )

    def test_shuffle_processed_auto_max_open_buckets_uses_fd_limit(self):
        with patch(
            "scaling_data.shuffle_processed._safe_auto_max_open_files",
            return_value=200,
        ):
            self.assertEqual(
                resolve_processed_shuffle_max_open_buckets(
                    bucket_count=4096,
                    max_open_buckets=0,
                ),
                200,
            )
            self.assertEqual(
                resolve_processed_shuffle_max_open_buckets(
                    bucket_count=32,
                    max_open_buckets=0,
                ),
                32,
            )
            self.assertEqual(
                resolve_processed_shuffle_max_open_buckets(
                    bucket_count=4096,
                    max_open_buckets=1000,
                ),
                200,
            )


class ManifestTests(unittest.TestCase):
    def test_default_manifest_uses_content_bearing_code_source(self):
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "manifests"
            / "general_100b_mix.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources = {source["id"]: source for source in manifest["sources"]}

        self.assertEqual(sources["long_form"]["hf_path"], "monology/pile-uncopyrighted")
        self.assertIsNone(sources["long_form"]["hf_name"])
        self.assertEqual(sources["long_form"]["text_field"], "text")

        self.assertEqual(sources["code"]["hf_path"], "bigcode/the-stack-dedup")
        self.assertEqual(sources["code"]["hf_name"], "default")
        self.assertEqual(sources["code"]["text_field"], "content")
        self.assertIn("file contents", sources["code"]["notes"])

        self.assertEqual(sources["multilingual"]["hf_path"], "uonlp/CulturaX")
        self.assertEqual(sources["multilingual"]["hf_name"], "zh")
        self.assertEqual(sources["multilingual"]["text_field"], "text")


class DownloadTests(unittest.TestCase):
    def test_iter_huggingface_rows_forwards_manifest_loader_options(self):
        captured = {}

        def load_dataset(**kwargs):
            captured.update(kwargs)
            return iter([{"text": "example"}])

        fake_datasets = types.SimpleNamespace(load_dataset=load_dataset)
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            rows = list(
                download_module.iter_huggingface_rows(
                    {
                        "id": "remote",
                        "hf_path": "org/repo",
                        "hf_name": "config",
                        "split": "train",
                        "hf_data_dir": "data/subset",
                        "hf_data_files": ["train/*.parquet"],
                        "hf_revision": "abc123",
                        "hf_trust_remote_code": True,
                    }
                )
            )

        self.assertEqual(rows, [{"text": "example"}])
        self.assertEqual(captured["path"], "org/repo")
        self.assertEqual(captured["name"], "config")
        self.assertEqual(captured["data_dir"], "data/subset")
        self.assertEqual(captured["data_files"], ["train/*.parquet"])
        self.assertEqual(captured["revision"], "abc123")
        self.assertTrue(captured["trust_remote_code"])
        self.assertTrue(captured["streaming"])

    def test_download_source_records_error_details_when_source_fails(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = {
                "id": "remote",
                "hf_path": "org/repo",
                "text_field": "text",
            }

            def failing_rows(source, *, skip_rows=0):
                raise RuntimeError("missing dataset config")
                yield

            with patch.object(download_module, "iter_source_rows", failing_rows):
                with self.assertRaisesRegex(RuntimeError, "missing dataset config"):
                    download_source(
                        source=source,
                        output_dir=root / "raw",
                        min_chars=5,
                    )

            state_path = root / "raw" / ".resume" / "remote.state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["error_type"], "RuntimeError")
            self.assertIn("missing dataset config", state["error_message"])

    def test_download_source_fails_fast_when_text_field_is_missing(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = {
                "id": "code",
                "hf_path": "bigcode/the-stack-v2",
                "hf_name": "default",
                "text_field": "content",
            }

            def rows_without_content(source, *, skip_rows=0):
                self.assertEqual(skip_rows, 0)
                yield {
                    "blob_id": "abc123",
                    "src_encoding": "UTF-8",
                    "language": "Python",
                    "path": "example.py",
                }
                yield {
                    "blob_id": "def456",
                    "src_encoding": "UTF-8",
                    "language": "Python",
                    "path": "example2.py",
                }

            with patch.object(download_module, "iter_source_rows", rows_without_content):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "text_field 'content' is missing",
                ):
                    download_source(
                        source=source,
                        output_dir=root / "raw",
                        min_chars=5,
                    )

            state_path = root / "raw" / ".resume" / "code.state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["error_type"], "RuntimeError")
            self.assertEqual(state["scanned_documents"], 1)
            self.assertEqual(state["written_documents"], 0)
            self.assertIn("text_field 'content' is missing", state["error_message"])
            self.assertIn("blob_id", state["error_message"])
            self.assertIn("language", state["error_message"])

    def test_download_source_can_use_local_jsonl_for_offline_samples(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "sample.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                write_jsonl_record(handle, {"body": " Alpha\n\nBeta Gamma "})
                write_jsonl_record(handle, {"body": "tiny"})
                write_jsonl_record(handle, {"body": "Alpha Beta Gamma"})

            output_path = download_source(
                source={
                    "id": "local_sample",
                    "local_path": str(local_path),
                    "text_field": "body",
                },
                output_dir=root / "raw",
                min_chars=10,
            )

            records = list(iter_jsonl_records(output_path))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["text"], "Alpha Beta Gamma")
            self.assertEqual(records[0]["source_id"], "local_sample")
            self.assertEqual(
                records[0]["document_hash"],
                document_hash("Alpha Beta Gamma"),
            )

    def test_download_source_reports_start_progress_and_completion(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "sample.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                for index in range(3):
                    write_jsonl_record(
                        handle,
                        {"body": f"Document {index} has enough text to keep."},
                    )

            events = []
            download_source(
                source={
                    "id": "local_sample",
                    "local_path": str(local_path),
                    "text_field": "body",
                },
                output_dir=root / "raw",
                min_chars=10,
                progress_every_docs=2,
                progress_callback=events.append,
            )

            self.assertEqual(events[0]["event"], "source_start")
            self.assertEqual(events[0]["source_id"], "local_sample")
            self.assertTrue(
                any(
                    event.get("event") == "source_progress"
                    and event.get("source_id") == "local_sample"
                    and event.get("scanned_documents") == 2
                    and event.get("written_documents") == 2
                    and event.get("estimated_tokens") == 16
                    for event in events
                )
            )
            self.assertEqual(events[-1]["event"], "source_complete")
            self.assertEqual(events[-1]["scanned_documents"], 3)
            self.assertEqual(events[-1]["written_documents"], 3)

    def test_download_source_stops_after_estimated_token_budget(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "sample.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                write_jsonl_record(handle, {"body": "aaaa bbbb cccc"})
                write_jsonl_record(handle, {"body": "dddd eeee ffff"})
                write_jsonl_record(handle, {"body": "gggg hhhh iiii"})

            events = []
            output_path = download_source(
                source={
                    "id": "local_sample",
                    "local_path": str(local_path),
                    "text_field": "body",
                },
                output_dir=root / "raw",
                min_chars=5,
                max_estimated_tokens=4,
                chars_per_token=4.0,
                progress_every_docs=1,
                progress_callback=events.append,
            )

            records = list(iter_jsonl_records(output_path))
            self.assertEqual(len(records), 2)
            self.assertEqual(events[-1]["event"], "source_complete")
            self.assertEqual(events[-1]["estimated_tokens"], 6)
            self.assertEqual(events[-1]["stop_reason"], "estimated_token_budget")

    def test_download_source_resumes_after_interrupted_partial_file(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "sample.jsonl"
            documents = [
                "aaaa bbbb cccc",
                "dddd eeee ffff",
                "gggg hhhh iiii",
                "jjjj kkkk llll",
            ]
            with local_path.open("w", encoding="utf-8") as handle:
                for text in documents:
                    write_jsonl_record(handle, {"body": text})

            def interrupted_rows(source, *, skip_rows=0):
                self.assertEqual(skip_rows, 0)
                yield {"body": documents[0]}
                yield {"body": documents[1]}
                raise RuntimeError("simulated stream interruption")

            source = {
                "id": "local_sample",
                "local_path": str(local_path),
                "text_field": "body",
            }
            with patch.object(download_module, "iter_source_rows", interrupted_rows):
                with self.assertRaisesRegex(RuntimeError, "simulated stream interruption"):
                    download_source(
                        source=source,
                        output_dir=root / "raw",
                        min_chars=5,
                        max_docs=3,
                        progress_every_docs=1,
                    )

            output_path = root / "raw" / "local_sample.jsonl.gz"
            resume_path = root / "raw" / ".resume" / "local_sample.jsonl"
            state_path = root / "raw" / ".resume" / "local_sample.state.json"
            self.assertFalse(output_path.exists())
            self.assertEqual(
                [record["text"] for record in iter_jsonl_records(resume_path)],
                documents[:2],
            )
            interrupted_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(interrupted_state["status"], "interrupted")
            self.assertEqual(interrupted_state["scanned_documents"], 2)

            seen_skip_rows = []

            def resumed_rows(source, *, skip_rows=0):
                seen_skip_rows.append(skip_rows)
                for text in documents[skip_rows:]:
                    yield {"body": text}

            with patch.object(download_module, "iter_source_rows", resumed_rows):
                resumed_path = download_source(
                    source=source,
                    output_dir=root / "raw",
                    min_chars=5,
                    max_docs=3,
                    progress_every_docs=1,
                )

            self.assertEqual(resumed_path, output_path)
            self.assertEqual(seen_skip_rows, [2])
            self.assertEqual(
                [record["text"] for record in iter_jsonl_records(output_path)],
                documents[:3],
            )
            completed_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(completed_state["status"], "completed")
            self.assertEqual(completed_state["written_documents"], 3)
            self.assertEqual(completed_state["stop_reason"], "max_docs")

    def test_download_source_resumes_from_safe_cursor_after_processing_failure(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "sample.jsonl"
            documents = [
                "aaaa bbbb cccc",
                "dddd eeee ffff",
            ]
            with local_path.open("w", encoding="utf-8") as handle:
                for text in documents:
                    write_jsonl_record(handle, {"body": text})

            source = {
                "id": "local_sample",
                "local_path": str(local_path),
                "text_field": "body",
            }
            with patch.object(
                download_module,
                "clean_text",
                side_effect=RuntimeError("simulated processing failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated processing failure",
                ):
                    download_source(
                        source=source,
                        output_dir=root / "raw",
                        min_chars=5,
                        progress_every_docs=1,
                    )

            state_path = root / "raw" / ".resume" / "local_sample.state.json"
            interrupted_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(interrupted_state["scanned_documents"], 1)
            self.assertEqual(
                interrupted_state["resume_safe_scanned_documents"],
                0,
            )

            resumed_path = download_source(
                source=source,
                output_dir=root / "raw",
                min_chars=5,
                progress_every_docs=1,
            )

            self.assertEqual(
                [record["text"] for record in iter_jsonl_records(resumed_path)],
                documents,
            )

    def test_download_source_legacy_state_without_safe_cursor_replays_conservatively(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            resume_dir = raw_dir / ".resume"
            resume_dir.mkdir(parents=True)
            local_path = root / "sample.jsonl"
            documents = [
                "aaaa bbbb cccc",
                "dddd eeee ffff",
            ]
            with local_path.open("w", encoding="utf-8") as handle:
                for text in documents:
                    write_jsonl_record(handle, {"body": text})

            resume_path = resume_dir / "local_sample.jsonl"
            with resume_path.open("w", encoding="utf-8") as handle:
                cleaned = documents[0]
                write_jsonl_record(
                    handle,
                    {
                        "text": cleaned,
                        "source_id": "local_sample",
                        "document_hash": document_hash(cleaned),
                        "text_preprocessed": True,
                    },
                )

            state_path = resume_dir / "local_sample.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_id": "local_sample",
                        "logical_source_id": "local_sample",
                        "source_location": str(local_path),
                        "text_field": "body",
                        "status": "interrupted",
                        "scanned_documents": 2,
                        "written_documents": 1,
                        "estimated_tokens": 3,
                        "stop_reason": "interrupted",
                        "output_path": str(raw_dir / "local_sample.jsonl.gz"),
                        "resume_path": str(resume_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            resumed_path = download_source(
                source={
                    "id": "local_sample",
                    "local_path": str(local_path),
                    "text_field": "body",
                },
                output_dir=raw_dir,
                min_chars=5,
                progress_every_docs=1,
            )

            self.assertEqual(
                [record["text"] for record in iter_jsonl_records(resumed_path)],
                documents,
            )

    def test_download_source_skips_completed_output_when_resuming(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "sample.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                write_jsonl_record(handle, {"body": "aaaa bbbb cccc"})
                write_jsonl_record(handle, {"body": "dddd eeee ffff"})

            source = {
                "id": "local_sample",
                "local_path": str(local_path),
                "text_field": "body",
            }
            output_path = download_source(
                source=source,
                output_dir=root / "raw",
                min_chars=5,
                max_docs=1,
                progress_every_docs=1,
            )

            events = []
            with patch.object(
                download_module,
                "iter_source_rows",
                side_effect=AssertionError("completed source should not be re-streamed"),
            ):
                resumed_path = download_source(
                    source=source,
                    output_dir=root / "raw",
                    min_chars=5,
                    max_docs=1,
                    progress_callback=events.append,
                )

            self.assertEqual(resumed_path, output_path)
            self.assertEqual(events[-1]["event"], "source_complete")
            self.assertEqual(events[-1]["stop_reason"], "already_complete")

    def test_download_source_rebuilds_completed_output_when_doc_cap_increases(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "sample.jsonl"
            documents = ["aaaa bbbb cccc", "dddd eeee ffff"]
            with local_path.open("w", encoding="utf-8") as handle:
                for text in documents:
                    write_jsonl_record(handle, {"body": text})

            source = {
                "id": "local_sample",
                "local_path": str(local_path),
                "text_field": "body",
            }
            output_path = download_source(
                source=source,
                output_dir=root / "raw",
                min_chars=5,
                max_docs=1,
                progress_every_docs=1,
            )
            self.assertEqual(len(list(iter_jsonl_records(output_path))), 1)

            resumed_path = download_source(
                source=source,
                output_dir=root / "raw",
                min_chars=5,
                max_docs=2,
                progress_every_docs=1,
            )

            self.assertEqual(resumed_path, output_path)
            self.assertEqual(
                [record["text"] for record in iter_jsonl_records(output_path)],
                documents,
            )

    def test_download_source_rebuilds_completed_output_when_source_changes(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_path = root / "first.jsonl"
            second_path = root / "second.jsonl"
            with first_path.open("w", encoding="utf-8") as handle:
                write_jsonl_record(handle, {"body": "aaaa bbbb cccc"})
            with second_path.open("w", encoding="utf-8") as handle:
                write_jsonl_record(handle, {"body": "zzzz yyyy xxxx"})

            output_path = download_source(
                source={
                    "id": "local_sample",
                    "local_path": str(first_path),
                    "text_field": "body",
                },
                output_dir=root / "raw",
                min_chars=5,
                max_docs=1,
                progress_every_docs=1,
            )

            download_source(
                source={
                    "id": "local_sample",
                    "local_path": str(second_path),
                    "text_field": "body",
                },
                output_dir=root / "raw",
                min_chars=5,
                max_docs=1,
                progress_every_docs=1,
            )

            self.assertEqual(
                [record["text"] for record in iter_jsonl_records(output_path)],
                ["zzzz yyyy xxxx"],
            )

    def test_download_source_rebuilds_completed_output_when_text_field_changes(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "sample.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                write_jsonl_record(
                    handle,
                    {
                        "old_body": "aaaa bbbb cccc",
                        "new_body": "zzzz yyyy xxxx",
                    },
                )

            output_path = download_source(
                source={
                    "id": "local_sample",
                    "local_path": str(local_path),
                    "text_field": "old_body",
                },
                output_dir=root / "raw",
                min_chars=5,
                max_docs=1,
                progress_every_docs=1,
            )
            self.assertEqual(
                [record["text"] for record in iter_jsonl_records(output_path)],
                ["aaaa bbbb cccc"],
            )

            download_source(
                source={
                    "id": "local_sample",
                    "local_path": str(local_path),
                    "text_field": "new_body",
                },
                output_dir=root / "raw",
                min_chars=5,
                max_docs=1,
                progress_every_docs=1,
            )

            self.assertEqual(
                [record["text"] for record in iter_jsonl_records(output_path)],
                ["zzzz yyyy xxxx"],
            )

    def test_iter_source_rows_shards_local_rows_before_resume_skip(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "sample.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                for index in range(5):
                    write_jsonl_record(handle, {"body": f"document-{index}"})

            rows = list(
                download_module.iter_source_rows(
                    {
                        "id": "logical_source",
                        "local_path": str(local_path),
                        "text_field": "body",
                        "download_num_shards": 2,
                        "download_shard_index": 1,
                    },
                    skip_rows=1,
                )
            )

            self.assertEqual([row["body"] for row in rows], ["document-3"])

    def test_download_source_uses_download_id_for_artifacts_and_logical_records(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "sample.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                write_jsonl_record(handle, {"body": "aaaa bbbb cccc"})
                write_jsonl_record(handle, {"body": "dddd eeee ffff"})

            events = []
            output_path = download_source(
                source={
                    "id": "logical_source",
                    "download_id": "logical_source.part000",
                    "download_num_shards": 2,
                    "download_shard_index": 0,
                    "local_path": str(local_path),
                    "text_field": "body",
                },
                output_dir=root / "raw",
                min_chars=5,
                progress_every_docs=1,
                progress_callback=events.append,
            )

            self.assertEqual(output_path.name, "logical_source.part000.jsonl.gz")
            self.assertEqual(
                [
                    record["source_id"]
                    for record in iter_jsonl_records(output_path)
                ],
                ["logical_source"],
            )
            self.assertEqual(events[0]["source_id"], "logical_source.part000")
            self.assertEqual(events[0]["logical_source_id"], "logical_source")
            self.assertEqual(events[-1]["source_id"], "logical_source.part000")
            self.assertEqual(events[-1]["logical_source_id"], "logical_source")

    def test_download_cli_can_overlap_sources_when_source_parallelism_enabled(self):
        original_download_source = download_module.download_source

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sources = []
            for source_id in ("source_a", "source_b"):
                source_path = root / f"{source_id}.jsonl"
                with source_path.open("w", encoding="utf-8") as handle:
                    write_jsonl_record(
                        handle,
                        {
                            "body": (
                                f"{source_id} document alpha beta gamma delta "
                                "epsilon"
                            )
                        },
                    )
                sources.append(
                    {
                        "id": source_id,
                        "weight": 0.5,
                        "local_path": str(source_path),
                        "text_field": "body",
                        "enabled": True,
                    }
                )

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "download_cli_mix",
                        "target_tokens": 20,
                        "sources": sources,
                    }
                ),
                encoding="utf-8",
            )

            entered_dir = root / "entered"
            overlap_path = root / "cross-source-overlap"
            output_dir = root / "raw"
            expected_source_ids = {"source_a", "source_b"}

            def blocking_download_source(**kwargs):
                source_id = str(kwargs["source"]["id"])
                entered_dir.mkdir(parents=True, exist_ok=True)
                (entered_dir / source_id).touch()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    if all((entered_dir / item).exists() for item in expected_source_ids):
                        overlap_path.touch()
                        break
                    time.sleep(0.01)
                return original_download_source(**kwargs)

            argv = [
                "scaling_data.download",
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--max-docs-per-source",
                "1",
                "--download-workers",
                "2",
                "--download-shards-per-source",
                "1",
                "--download-source-parallelism",
                "2",
                "--min-chars",
                "5",
            ]

            with patch.object(download_module, "download_source", blocking_download_source):
                with patch.object(sys, "argv", argv):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with contextlib.redirect_stderr(io.StringIO()):
                            download_module.main()

            self.assertTrue(overlap_path.exists())
            self.assertTrue((output_dir / "source_a.jsonl.gz").exists())
            self.assertTrue((output_dir / "source_b.jsonl.gz").exists())

    def test_download_cli_continues_scheduling_after_shard_failure(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sources = []
            for source_id in ("source_a", "source_b"):
                source_path = root / f"{source_id}.jsonl"
                with source_path.open("w", encoding="utf-8") as handle:
                    write_jsonl_record(
                        handle,
                        {"body": f"{source_id} alpha beta gamma delta"},
                    )
                sources.append(
                    {
                        "id": source_id,
                        "weight": 0.5,
                        "local_path": str(source_path),
                        "text_field": "body",
                        "enabled": True,
                    }
                )

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "download_cli_continue_mix",
                        "target_tokens": 20,
                        "sources": sources,
                    }
                ),
                encoding="utf-8",
            )

            started_dir = root / "started"

            def fake_download_source(**kwargs):
                source = kwargs["source"]
                download_id = str(source["download_id"])
                started_dir.mkdir(parents=True, exist_ok=True)
                (started_dir / download_id).touch()
                if download_id == "source_a.part000":
                    raise RuntimeError("simulated shard failure")
                out_path = kwargs["output_dir"] / f"{download_id}.jsonl.gz"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"")
                return out_path

            argv = [
                "scaling_data.download",
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(root / "raw"),
                "--download-workers",
                "2",
                "--download-shards-per-source",
                "3",
                "--download-source-parallelism",
                "2",
                "--min-chars",
                "5",
            ]

            with patch.object(download_module, "download_source", fake_download_source):
                with patch.object(sys, "argv", argv):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with contextlib.redirect_stderr(io.StringIO()):
                            with self.assertRaisesRegex(
                                download_module.DownloadStageError,
                                "source_a.part000",
                            ):
                                download_module.main()

            expected_download_ids = {
                "source_a.part000",
                "source_a.part001",
                "source_a.part002",
                "source_b.part000",
                "source_b.part001",
                "source_b.part002",
            }
            self.assertEqual(
                {path.name for path in started_dir.iterdir()},
                expected_download_ids,
            )
            for download_id in expected_download_ids:
                state_path = root / "raw" / ".resume" / f"{download_id}.state.json"
                self.assertTrue(state_path.exists(), download_id)
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "queued")
                self.assertEqual(state["resume_safe_scanned_documents"], 0)


class PipelinePlanTests(unittest.TestCase):
    def test_pipeline_plan_records_expected_stage_outputs(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "sample_mix",
                        "target_tokens": 10,
                        "tokenizer": "test-tokenizer",
                        "sources": [
                            {
                                "id": "local_sample",
                                "weight": 1.0,
                                "local_path": str(root / "sample.jsonl"),
                                "text_field": "text",
                                "enabled": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output_path = write_pipeline_plan(
                manifest_path=manifest_path,
                output_dir=root / "plans",
                work_dir=root / "work",
            )

            plan = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["manifest_name"], "sample_mix")
            self.assertEqual(
                plan["expected_outputs"]["tokenized_train_index"],
                str(root / "work" / "tokenized" / "train" / "index.json"),
            )
            self.assertEqual(
                plan["expected_outputs"]["tokenized_validation_index"],
                str(root / "work" / "tokenized" / "validation" / "index.json"),
            )
            self.assertEqual(
                plan["expected_outputs"]["prepared_data_manifest"],
                str(root / "work" / "prepared-data-manifest.json"),
            )
            self.assertEqual(
                [stage["name"] for stage in plan["stages"]],
                ["blend", "download", "postprocess", "tokenize"],
            )


class PrepareDataTests(unittest.TestCase):
    def test_prepare_data_defaults_disable_download_token_budget(self):
        config = PrepareDataConfig(
            manifest_path=Path("manifest.json"),
            work_dir=Path("work"),
        )

        self.assertEqual(config.download_token_overfetch_ratio, 0.0)

    def test_prepare_data_runs_full_pipeline_and_writes_deployment_manifest(self):
        class FakeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [len(part) for part in text.split()]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "source.jsonl"
            selected_docs = _documents_covering_both_splits(validation_rate=0.5)
            with local_path.open("w", encoding="utf-8") as handle:
                for text in selected_docs:
                    write_jsonl_record(handle, {"body": text})

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "sample_mix",
                        "target_tokens": 100,
                        "tokenizer": "fake-tokenizer",
                        "sources": [
                            {
                                "id": "local_sample",
                                "weight": 1.0,
                                "local_path": str(local_path),
                                "text_field": "body",
                                "enabled": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = prepare_data(
                PrepareDataConfig(
                    manifest_path=manifest_path,
                    work_dir=root / "work",
                    tokenizer_name="fake-tokenizer",
                    target_shard_tokens=8,
                    validation_rate=0.5,
                    min_chars=20,
                    max_docs_per_source=None,
                    index_validation_mode="metadata",
                ),
                tokenizer_loader=lambda name: FakeTokenizer(),
            )

            final_manifest_path = Path(summary["prepared_manifest_path"])
            final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertTrue(Path(summary["pipeline_plan_path"]).exists())
            self.assertTrue(Path(summary["blend_plan_path"]).exists())
            self.assertEqual(final_manifest["manifest_name"], "sample_mix")
            self.assertEqual(
                set(final_manifest["tokenized_indexes"]),
                {"train", "validation"},
            )
            train_index = Path(final_manifest["tokenized_indexes"]["train"]["path"])
            validation_index = Path(
                final_manifest["tokenized_indexes"]["validation"]["path"]
            )
            self.assertTrue(train_index.exists())
            self.assertTrue(validation_index.exists())
            self.assertEqual(
                final_manifest["api_runtime_overrides"][
                    "SCALING_TOKENIZED_TRAIN_INDEX_URI"
                ],
                train_index.resolve().as_uri(),
            )
            self.assertEqual(
                final_manifest["api_runtime_overrides"][
                    "SCALING_TOKENIZED_VALIDATION_INDEX_URI"
                ],
                validation_index.resolve().as_uri(),
            )

    def test_prepare_data_tokenizes_shuffled_processed_records(self):
        class FakeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [len(part) for part in text.split()]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "source.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                for index in range(12):
                    write_jsonl_record(
                        handle,
                        {"body": f"document {index} alpha beta gamma"},
                    )

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "sample_mix",
                        "target_tokens": 100,
                        "tokenizer": "fake-tokenizer",
                        "sources": [
                            {
                                "id": "local_sample",
                                "weight": 1.0,
                                "local_path": str(local_path),
                                "text_field": "body",
                                "enabled": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = prepare_data(
                PrepareDataConfig(
                    manifest_path=manifest_path,
                    work_dir=root / "work",
                    tokenizer_name="fake-tokenizer",
                    target_shard_tokens=16,
                    validation_rate=0.0,
                    min_chars=5,
                    processed_shuffle_seed=17,
                    processed_shuffle_bucket_count=3,
                    processed_shuffle_max_open_buckets=1,
                    index_validation_mode="metadata",
                ),
                tokenizer_loader=lambda name: FakeTokenizer(),
            )

            original_records = list(
                iter_jsonl_records(root / "work" / "processed_jsonl" / "train.jsonl.gz")
            )
            shuffled_path = (
                root / "work" / "processed_jsonl_shuffled" / "train.jsonl.gz"
            )
            shuffled_records = list(iter_jsonl_records(shuffled_path))
            self.assertEqual(
                summary["tokenization_input_dir"],
                str(root / "work" / "processed_jsonl_shuffled"),
            )
            self.assertEqual(
                summary["processed_shuffle_stats"]["splits"]["train"]["documents"],
                12,
            )
            self.assertCountEqual(
                [record["document_hash"] for record in shuffled_records],
                [record["document_hash"] for record in original_records],
            )
            self.assertNotEqual(
                [record["document_hash"] for record in shuffled_records],
                [record["document_hash"] for record in original_records],
            )
            self.assertTrue(
                (root / "work" / "tokenized" / "train" / "index.json").exists()
            )

    def test_prepare_data_plan_only_writes_audit_plans_without_downloads(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "sample_mix",
                        "target_tokens": 10,
                        "tokenizer": "fake-tokenizer",
                        "sources": [
                            {
                                "id": "local_sample",
                                "weight": 1.0,
                                "local_path": str(root / "source.jsonl"),
                                "text_field": "text",
                                "enabled": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = prepare_data(
                PrepareDataConfig(
                    manifest_path=manifest_path,
                    work_dir=root / "work",
                    tokenizer_name="fake-tokenizer",
                    plan_only=True,
                )
            )

            self.assertEqual(summary["status"], "planned")
            self.assertTrue(Path(summary["pipeline_plan_path"]).exists())
            self.assertTrue(Path(summary["blend_plan_path"]).exists())
            self.assertFalse((root / "work" / "raw_jsonl").exists())

    def test_rebuild_tokenized_from_existing_processed_records_shuffles_first(self):
        class FakeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [len(part) for part in text.split()]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed_dir = root / "processed_jsonl"
            processed_dir.mkdir()
            with open_text_writer(processed_dir / "train.jsonl.gz") as handle:
                for index in range(10):
                    write_jsonl_record(
                        handle,
                        {
                            "text": f"train document {index} alpha beta",
                            "source_id": "local_sample",
                            "split": "train",
                            "document_hash": f"train-{index}",
                        },
                    )
            with open_text_writer(processed_dir / "validation.jsonl.gz") as handle:
                for index in range(4):
                    write_jsonl_record(
                        handle,
                        {
                            "text": f"validation document {index} gamma",
                            "source_id": "local_sample",
                            "split": "validation",
                            "document_hash": f"validation-{index}",
                        },
                    )

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "sample_mix",
                        "target_tokens": 100,
                        "tokenizer": "fake-tokenizer",
                        "sources": [
                            {
                                "id": "local_sample",
                                "weight": 1.0,
                                "local_path": str(root / "unused.jsonl"),
                                "text_field": "body",
                                "enabled": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = rebuild_tokenized_from_processed(
                RebuildTokenizedConfig(
                    manifest_path=manifest_path,
                    processed_dir=processed_dir,
                    work_dir=root / "work",
                    tokenizer_name="fake-tokenizer",
                    target_shard_tokens=16,
                    shuffle_seed=17,
                    shuffle_bucket_count=3,
                    shuffle_max_open_buckets=1,
                    index_validation_mode="metadata",
                ),
                tokenizer_loader=lambda name: FakeTokenizer(),
            )

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(
                summary["tokenization_input_dir"],
                str(root / "work" / "processed_jsonl_shuffled"),
            )
            self.assertTrue(
                Path(summary["tokenized_indexes"]["train"]["path"]).exists()
            )
            self.assertTrue(
                Path(summary["tokenized_indexes"]["validation"]["path"]).exists()
            )
            self.assertNotEqual(
                [
                    record["document_hash"]
                    for record in iter_jsonl_records(
                        root / "work" / "processed_jsonl_shuffled" / "train.jsonl.gz"
                    )
                ],
                [f"train-{index}" for index in range(10)],
            )
            self.assertFalse(summary["processed_shuffle_stats"]["reused"])
            self.assertEqual(summary["requested_shuffle_max_open_buckets"], 1)
            self.assertEqual(summary["shuffle_max_open_buckets"], 1)

            stats_path = root / "work" / "processed_jsonl_shuffled" / "shuffle-stats.json"
            stats_before = stats_path.read_text(encoding="utf-8")
            second_summary = rebuild_tokenized_from_processed(
                RebuildTokenizedConfig(
                    manifest_path=manifest_path,
                    processed_dir=processed_dir,
                    work_dir=root / "work",
                    tokenizer_name="fake-tokenizer",
                    target_shard_tokens=16,
                    shuffle_seed=17,
                    shuffle_bucket_count=3,
                    shuffle_max_open_buckets=1,
                    index_validation_mode="metadata",
                ),
                tokenizer_loader=lambda name: FakeTokenizer(),
            )

            self.assertTrue(second_summary["processed_shuffle_stats"]["reused"])
            self.assertEqual(
                stats_path.read_text(encoding="utf-8"),
                stats_before,
            )

    def test_prepare_data_uses_blend_token_targets_when_tokenizing(self):
        class FakeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [len(part) for part in text.split()]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "source.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                write_jsonl_record(handle, {"body": "alpha beta"})
                write_jsonl_record(handle, {"body": "gamma delta"})

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "sample_mix",
                        "target_tokens": 4,
                        "tokenizer": "fake-tokenizer",
                        "sources": [
                            {
                                "id": "local_sample",
                                "weight": 1.0,
                                "local_path": str(local_path),
                                "text_field": "body",
                                "enabled": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = prepare_data(
                PrepareDataConfig(
                    manifest_path=manifest_path,
                    work_dir=root / "work",
                    tokenizer_name="fake-tokenizer",
                    target_shard_tokens=8,
                    validation_rate=0.0,
                    min_chars=5,
                    index_validation_mode="metadata",
                ),
                tokenizer_loader=lambda name: FakeTokenizer(),
            )

            self.assertEqual(
                summary["tokenized_indexes"]["train"]["total_tokens"],
                3,
            )

    def test_prepare_data_uses_blend_token_targets_to_limit_downloads(self):
        class FakeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [len(part) for part in text.split()]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "source.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                write_jsonl_record(handle, {"body": "aaaa bbbb cccc"})
                write_jsonl_record(handle, {"body": "dddd eeee ffff"})
                write_jsonl_record(handle, {"body": "gggg hhhh iiii"})

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "sample_mix",
                        "target_tokens": 4,
                        "tokenizer": "fake-tokenizer",
                        "sources": [
                            {
                                "id": "local_sample",
                                "weight": 1.0,
                                "local_path": str(local_path),
                                "text_field": "body",
                                "enabled": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = prepare_data(
                PrepareDataConfig(
                    manifest_path=manifest_path,
                    work_dir=root / "work",
                    tokenizer_name="fake-tokenizer",
                    target_shard_tokens=8,
                    validation_rate=0.0,
                    min_chars=5,
                    download_token_overfetch_ratio=1.0,
                    download_chars_per_token=4.0,
                    index_validation_mode="metadata",
                ),
                tokenizer_loader=lambda name: FakeTokenizer(),
            )

            raw_records = list(iter_jsonl_records(Path(summary["raw_paths"][0])))
            self.assertEqual(len(raw_records), 2)
            self.assertEqual(summary["download_token_budgets"]["local_sample"], 4)
            self.assertLess(
                summary["download_stats"]["local_sample"]["scanned_documents"],
                3,
            )

    def test_prepare_data_processes_sources_sequentially_with_parallel_shards(self):
        class FakeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [len(part) for part in text.split()]

        original_download_source = prepare_module.download_source

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sources = []
            for source_id in ("source_a", "source_b"):
                source_path = root / f"{source_id}.jsonl"
                with source_path.open("w", encoding="utf-8") as handle:
                    for index in range(4):
                        write_jsonl_record(
                            handle,
                            {
                                "body": (
                                    f"{source_id} document {index} alpha beta "
                                    "gamma delta epsilon"
                                )
                            },
                        )
                sources.append(
                    {
                        "id": source_id,
                        "weight": 0.5,
                        "local_path": str(source_path),
                        "text_field": "body",
                        "enabled": True,
                    }
                )

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "parallel_mix",
                        "target_tokens": 20,
                        "tokenizer": "fake-tokenizer",
                        "sources": sources,
                    }
                ),
                encoding="utf-8",
            )

            active_dir = root / "active"
            started_dir = root / "started"
            parallel_dir = root / "parallel"
            cross_source_overlap = root / "cross-source-overlap"

            def blocking_download_source(**kwargs):
                source = kwargs["source"]
                logical_source_id = str(source["id"])
                download_id = str(source["download_id"])
                shard_count = int(source.get("download_num_shards", 1))
                active_source_dir = active_dir / logical_source_id
                started_source_dir = started_dir / logical_source_id
                active_source_dir.mkdir(parents=True, exist_ok=True)
                started_source_dir.mkdir(parents=True, exist_ok=True)
                active_marker = active_source_dir / download_id
                active_marker.touch()
                (started_source_dir / download_id).touch()
                active_source_count = sum(
                    1
                    for source_dir in active_dir.iterdir()
                    if source_dir.is_dir() and any(source_dir.iterdir())
                )
                if active_source_count > 1:
                    cross_source_overlap.touch()
                if shard_count > 1:
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline:
                        if len(list(started_source_dir.iterdir())) >= shard_count:
                            break
                        time.sleep(0.01)
                if len(list(active_source_dir.iterdir())) >= 2:
                    parallel_dir.mkdir(parents=True, exist_ok=True)
                    (parallel_dir / logical_source_id).touch()
                try:
                    return original_download_source(**kwargs)
                finally:
                    active_marker.unlink(missing_ok=True)

            with patch.object(
                prepare_module,
                "download_source",
                side_effect=blocking_download_source,
            ):
                summary = prepare_data(
                    PrepareDataConfig(
                        manifest_path=manifest_path,
                        work_dir=root / "work",
                        tokenizer_name="fake-tokenizer",
                        target_shard_tokens=8,
                        validation_rate=0.0,
                        min_chars=5,
                        download_workers=2,
                        download_shards_per_source=2,
                        index_validation_mode="metadata",
                    ),
                    tokenizer_loader=lambda name: FakeTokenizer(),
                )

            self.assertFalse(cross_source_overlap.exists())
            self.assertEqual(
                {path.name for path in parallel_dir.iterdir()},
                {"source_a", "source_b"},
            )
            self.assertEqual(summary["download_workers"], 2)
            self.assertEqual(summary["download_shards_per_source"], 2)
            self.assertEqual(
                [Path(path).name for path in summary["raw_paths"]],
                [
                    "source_a.part000.jsonl.gz",
                    "source_a.part001.jsonl.gz",
                    "source_b.part000.jsonl.gz",
                    "source_b.part001.jsonl.gz",
                ],
            )
            self.assertEqual(
                set(summary["download_stats"]),
                {
                    "source_a.part000",
                    "source_a.part001",
                    "source_b.part000",
                    "source_b.part001",
                },
            )
            self.assertEqual(
                {
                    stats["logical_source_id"]
                    for stats in summary["download_stats"].values()
                },
                {"source_a", "source_b"},
            )
            self.assertEqual(
                {
                    record["source_id"]
                    for raw_path in summary["raw_paths"]
                    for record in iter_jsonl_records(Path(raw_path))
                },
                {"source_a", "source_b"},
            )

    def test_prepare_data_can_overlap_sources_when_source_parallelism_enabled(self):
        class FakeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [len(part) for part in text.split()]

        original_download_source = prepare_module.download_source

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sources = []
            for source_id in ("source_a", "source_b"):
                source_path = root / f"{source_id}.jsonl"
                with source_path.open("w", encoding="utf-8") as handle:
                    write_jsonl_record(
                        handle,
                        {
                            "body": (
                                f"{source_id} document alpha beta gamma delta "
                                "epsilon"
                            )
                        },
                    )
                sources.append(
                    {
                        "id": source_id,
                        "weight": 0.5,
                        "local_path": str(source_path),
                        "text_field": "body",
                        "enabled": True,
                    }
                )

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "source_parallel_mix",
                        "target_tokens": 20,
                        "tokenizer": "fake-tokenizer",
                        "sources": sources,
                    }
                ),
                encoding="utf-8",
            )

            active_dir = root / "active"
            overlap_path = root / "cross-source-overlap"
            expected_source_ids = {"source_a", "source_b"}

            def blocking_download_source(**kwargs):
                source_id = str(kwargs["source"]["id"])
                active_source_dir = active_dir / source_id
                active_source_dir.mkdir(parents=True, exist_ok=True)
                active_marker = active_source_dir / str(
                    kwargs["source"].get("download_id", source_id)
                )
                active_marker.touch()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    if all(
                        any((active_dir / item).iterdir())
                        for item in expected_source_ids
                        if (active_dir / item).exists()
                    ) and all((active_dir / item).exists() for item in expected_source_ids):
                        overlap_path.touch()
                        break
                    time.sleep(0.01)
                try:
                    return original_download_source(**kwargs)
                finally:
                    active_marker.unlink(missing_ok=True)

            with patch.object(
                prepare_module,
                "download_source",
                side_effect=blocking_download_source,
            ):
                summary = prepare_data(
                    PrepareDataConfig(
                        manifest_path=manifest_path,
                        work_dir=root / "work",
                        tokenizer_name="fake-tokenizer",
                        target_shard_tokens=8,
                        validation_rate=0.0,
                        min_chars=5,
                        download_workers=2,
                        download_shards_per_source=1,
                        download_source_parallelism=2,
                        index_validation_mode="metadata",
                    ),
                    tokenizer_loader=lambda name: FakeTokenizer(),
                )

            self.assertTrue(overlap_path.exists())
            self.assertEqual(summary["download_source_parallelism"], 2)
            self.assertEqual(
                [Path(path).name for path in summary["raw_paths"]],
                ["source_a.jsonl.gz", "source_b.jsonl.gz"],
            )

    def test_prepare_data_writes_queued_state_for_all_planned_shards(self):
        class FakeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [len(part) for part in text.split()]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "source.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                write_jsonl_record(
                    handle,
                    {"body": "alpha beta gamma delta epsilon"},
                )

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "queued_mix",
                        "target_tokens": 10,
                        "tokenizer": "fake-tokenizer",
                        "sources": [
                            {
                                "id": "local_sample",
                                "weight": 1.0,
                                "local_path": str(local_path),
                                "text_field": "body",
                                "enabled": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def failing_download_source(**kwargs):
                queued_state_path = (
                    root
                    / "work"
                    / "raw_jsonl"
                    / ".resume"
                    / "local_sample.part001.state.json"
                )
                queued_state = json.loads(
                    queued_state_path.read_text(encoding="utf-8")
                )
                self.assertEqual(queued_state["status"], "queued")
                self.assertEqual(
                    queued_state["resume_safe_scanned_documents"],
                    0,
                )
                raise RuntimeError("stop after queued-state assertion")

            with patch.object(
                prepare_module,
                "download_source",
                side_effect=failing_download_source,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "stop after queued-state assertion",
                ):
                    prepare_data(
                        PrepareDataConfig(
                            manifest_path=manifest_path,
                            work_dir=root / "work",
                            tokenizer_name="fake-tokenizer",
                            target_shard_tokens=8,
                            validation_rate=0.0,
                            min_chars=5,
                            download_workers=1,
                            download_shards_per_source=2,
                            index_validation_mode="metadata",
                        ),
                        tokenizer_loader=lambda name: FakeTokenizer(),
                    )

    def test_prepare_data_continues_scheduling_after_shard_failure(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sources = []
            for source_id in ("source_a", "source_b"):
                local_path = root / f"{source_id}.jsonl"
                with local_path.open("w", encoding="utf-8") as handle:
                    write_jsonl_record(
                        handle,
                        {"body": f"{source_id} alpha beta gamma delta"},
                    )
                sources.append(
                    {
                        "id": source_id,
                        "weight": 0.5,
                        "local_path": str(local_path),
                        "text_field": "body",
                        "enabled": True,
                    }
                )

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "continue_after_failure_mix",
                        "target_tokens": 20,
                        "tokenizer": "fake-tokenizer",
                        "sources": sources,
                    }
                ),
                encoding="utf-8",
            )

            started_dir = root / "started"

            def fake_download_source(**kwargs):
                source = kwargs["source"]
                download_id = str(source["download_id"])
                started_dir.mkdir(parents=True, exist_ok=True)
                (started_dir / download_id).touch()
                if download_id == "source_a.part000":
                    raise RuntimeError("simulated shard failure")
                out_path = kwargs["output_dir"] / f"{download_id}.jsonl.gz"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"")
                return out_path

            with patch.object(
                prepare_module,
                "download_source",
                side_effect=fake_download_source,
            ):
                with self.assertRaisesRegex(
                    prepare_module.DownloadStageError,
                    "source_a.part000",
                ):
                    prepare_module._download_sources(
                        sources=sources,
                        output_dir=root / "raw",
                        config=PrepareDataConfig(
                            manifest_path=manifest_path,
                            work_dir=root / "work",
                            tokenizer_name="fake-tokenizer",
                            download_workers=2,
                            download_shards_per_source=3,
                            download_source_parallelism=2,
                            min_chars=5,
                        ),
                        download_token_budgets={},
                    )

            self.assertEqual(
                {path.name for path in started_dir.iterdir()},
                {
                    "source_a.part000",
                    "source_a.part001",
                    "source_a.part002",
                    "source_b.part000",
                    "source_b.part001",
                    "source_b.part002",
                },
            )

    def test_prepare_data_plan_records_download_shards_per_source(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "sample_mix",
                        "target_tokens": 10,
                        "tokenizer": "fake-tokenizer",
                        "sources": [
                            {
                                "id": "local_sample",
                                "weight": 1.0,
                                "local_path": str(root / "source.jsonl"),
                                "text_field": "text",
                                "enabled": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = prepare_data(
                PrepareDataConfig(
                    manifest_path=manifest_path,
                    work_dir=root / "work",
                    tokenizer_name="fake-tokenizer",
                    download_workers=8,
                    download_shards_per_source=4,
                    download_source_parallelism=2,
                    plan_only=True,
                )
            )

            self.assertEqual(summary["download_workers"], 8)
            self.assertEqual(summary["download_shards_per_source"], 4)
            self.assertEqual(summary["download_source_parallelism"], 2)
            self.assertEqual(summary["postprocess_workers"], 8)
            self.assertEqual(summary["tokenize_workers"], 8)
            self.assertEqual(summary["tokenize_chunk_records"], 512)

    def test_prepare_data_prints_progress_to_stderr_when_enabled(self):
        class FakeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [len(part) for part in text.split()]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_path = root / "source.jsonl"
            with local_path.open("w", encoding="utf-8") as handle:
                write_jsonl_record(
                    handle,
                    {"body": "alpha beta gamma delta epsilon zeta eta theta"},
                )

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "sample_mix",
                        "target_tokens": 100,
                        "tokenizer": "fake-tokenizer",
                        "sources": [
                            {
                                "id": "local_sample",
                                "weight": 1.0,
                                "local_path": str(local_path),
                                "text_field": "body",
                                "enabled": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                prepare_data(
                    PrepareDataConfig(
                        manifest_path=manifest_path,
                        work_dir=root / "work",
                        tokenizer_name="fake-tokenizer",
                        target_shard_tokens=8,
                        validation_rate=0.0,
                        min_chars=20,
                        progress_every_docs=1,
                        index_validation_mode="metadata",
                        show_progress=True,
                    ),
                    tokenizer_loader=lambda name: FakeTokenizer(),
                )

            progress_output = stderr.getvalue()
            self.assertIn("[prepare] downloading", progress_output)
            self.assertIn("source=local_sample", progress_output)
            self.assertIn("[prepare] tokenization complete", progress_output)


class LifecycleDatasetTests(unittest.TestCase):
    def test_package_lifecycle_dataset_from_train_only_partial_shards(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_train = root / "tokenized" / "train"
            _write_uint32_token_shard(source_train / "tokens-000000.bin", [1, 2, 3, 4])
            _write_uint32_token_shard(source_train / "tokens-000001.bin", [5, 6, 7, 8])
            _write_uint32_token_shard(source_train / "tokens-000002.bin", [9, 10, 11, 12])

            manifest = package_lifecycle_dataset(
                source_tokenized_dir=root / "tokenized",
                output_dir=root / "lifecycle",
                train_shards=2,
                validation_shards=1,
                link_mode="copy",
                sequence_length=2,
                train_tokens=4,
                validation_tokens_per_eval=2,
                num_evals=2,
            )

            self.assertTrue(manifest["validation_built_from_train_shards"])
            self.assertEqual(manifest["train_total_tokens"], 8)
            self.assertEqual(manifest["validation_total_tokens"], 4)
            self.assertTrue(manifest["train_index_uri"].startswith("file://"))
            self.assertTrue(manifest["validation_index_uri"].startswith("file://"))
            self.assertEqual(
                manifest["runtime_overrides"]["SCALING_VALIDATION_TOKENS_PER_EVAL"],
                "2",
            )
            self.assertEqual(
                manifest["runtime_overrides"][
                    "SCALING_FINAL_TOKENIZED_VALIDATION_INDEX_URI"
                ],
                manifest["validation_index_uri"],
            )
            self.assertEqual(
                manifest["recommended_student_config"]["training"]["train_tokens"],
                4,
            )
            self.assertEqual(
                manifest["recommended_student_config"]["training"]["num_evals"],
                2,
            )
            self.assertNotIn("notes", manifest["recommended_student_config"])

            output_root = root / "lifecycle"
            train_index = load_tokenized_index(
                output_root / "tokenized" / "train" / "index.json"
            )
            validation_index = load_tokenized_index(
                output_root / "tokenized" / "validation" / "index.json"
            )
            self.assertEqual(train_index["total_tokens"], 8)
            self.assertEqual(validation_index["total_tokens"], 4)
            self.assertEqual(
                read_token_shard(
                    output_root / "tokenized" / "train",
                    train_index["shards"][0],
                ),
                [1, 2, 3, 4],
            )
            self.assertEqual(
                read_token_shard(
                    output_root / "tokenized" / "validation",
                    validation_index["shards"][0],
                ),
                [9, 10, 11, 12],
            )
            self.assertTrue((output_root / "api-runtime-overrides.env").exists())
            self.assertTrue((output_root / "api-runtime-overrides.json").exists())
            self.assertTrue((output_root / "README.md").exists())

    def test_package_lifecycle_dataset_rejects_insufficient_train_shards(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_train = root / "tokenized" / "train"
            _write_uint32_token_shard(source_train / "tokens-000000.bin", [1, 2, 3, 4])

            with self.assertRaisesRegex(LifecycleDatasetError, "not enough train shards"):
                package_lifecycle_dataset(
                    source_tokenized_dir=root / "tokenized",
                    output_dir=root / "lifecycle",
                    train_shards=1,
                    validation_shards=1,
                    sequence_length=2,
                    train_tokens=2,
                    validation_tokens_per_eval=2,
                )

    def test_package_lifecycle_dataset_rejects_output_that_contains_source(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_train = root / "data_work" / "tokenized" / "train"
            _write_uint32_token_shard(source_train / "tokens-000000.bin", [1, 2, 3, 4])
            _write_uint32_token_shard(source_train / "tokens-000001.bin", [5, 6, 7, 8])

            with self.assertRaisesRegex(
                LifecycleDatasetError,
                "must not contain the source shard directory",
            ):
                package_lifecycle_dataset(
                    source_tokenized_dir=root / "data_work" / "tokenized",
                    output_dir=root / "data_work",
                    train_shards=1,
                    validation_shards=1,
                    overwrite=True,
                    sequence_length=2,
                    train_tokens=2,
                    validation_tokens_per_eval=2,
                )


class TokenShardWriterTests(unittest.TestCase):
    def test_index_includes_shard_hashes_and_source_token_accounting(self):
        with TemporaryDirectory() as tmpdir:
            writer = TokenShardWriter(
                output_dir=Path(tmpdir),
                target_shard_tokens=3,
            )

            writer.add([1, 2], source_id="general_web")
            writer.add([3, 4, 5], source_id="code")
            index = writer.close()

            self.assertEqual(index["schema_version"], 1)
            self.assertEqual(index["token_dtype"], "uint32")
            self.assertEqual(index["byte_order"], "little")
            self.assertEqual(index["shard_format"], "flat_binary_uint32_le")
            self.assertEqual(index["total_tokens"], 5)
            self.assertEqual(
                index["source_token_counts"],
                {"code": 3, "general_web": 2},
            )
            self.assertEqual(len(index["shards"]), 2)
            self.assertEqual(index["shards"][0]["source_token_counts"], {
                "code": 1,
                "general_web": 2,
            })
            self.assertEqual(index["shards"][1]["source_token_counts"], {"code": 2})
            for shard in index["shards"]:
                self.assertRegex(shard["sha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(shard["byte_order"], "little")

    def test_tokenize_records_by_split_writes_train_and_validation_indexes(self):
        class FakeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [len(part) for part in text.split()]

        with TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "processed"
            input_dir.mkdir()
            with (input_dir / "mixed.jsonl").open("w", encoding="utf-8") as handle:
                write_jsonl_record(
                    handle,
                    {"text": "alpha beta", "source_id": "general_web", "split": "train"},
                )
                write_jsonl_record(
                    handle,
                    {"text": "gamma", "source_id": "math", "split": "validation"},
                )

            output_dir = Path(tmpdir) / "tokenized"
            indexes = tokenize_records_by_split(
                input_dir=input_dir,
                output_dir=output_dir,
                tokenizer=FakeTokenizer(),
                target_shard_tokens=8,
            )

            self.assertEqual(set(indexes), {"train", "validation"})
            self.assertTrue((output_dir / "train" / "index.json").exists())
            self.assertTrue((output_dir / "validation" / "index.json").exists())
            self.assertEqual(indexes["train"]["source_token_counts"], {
                "general_web": 3,
            })
            self.assertEqual(indexes["validation"]["source_token_counts"], {
                "math": 2,
            })

    def test_tokenize_records_by_split_caps_train_tokens_by_source_without_truncating_documents(self):
        class FakeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [len(part) for part in text.split()]

        with TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "processed"
            input_dir.mkdir()
            with (input_dir / "mixed.jsonl").open("w", encoding="utf-8") as handle:
                write_jsonl_record(
                    handle,
                    {
                        "text": "alpha beta gamma delta",
                        "source_id": "general_web",
                        "split": "train",
                    },
                )
                write_jsonl_record(
                    handle,
                    {
                        "text": "epsilon zeta",
                        "source_id": "general_web",
                        "split": "train",
                    },
                )
                write_jsonl_record(
                    handle,
                    {
                        "text": "held out",
                        "source_id": "general_web",
                        "split": "validation",
                    },
                )

            output_dir = Path(tmpdir) / "tokenized"
            indexes = tokenize_records_by_split(
                input_dir=input_dir,
                output_dir=output_dir,
                tokenizer=FakeTokenizer(),
                target_shard_tokens=8,
                train_source_token_limits={"general_web": 6},
            )

            self.assertEqual(indexes["train"]["total_tokens"], 5)
            self.assertEqual(indexes["train"]["source_token_counts"], {
                "general_web": 5,
            })
            self.assertEqual(indexes["validation"]["total_tokens"], 3)

    def test_tokenize_records_by_split_parallel_workers_preserve_token_caps_without_truncating_documents(self):
        with TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "processed"
            input_dir.mkdir()
            with (input_dir / "mixed.jsonl").open("w", encoding="utf-8") as handle:
                write_jsonl_record(
                    handle,
                    {
                        "text": "alpha beta gamma delta",
                        "source_id": "general_web",
                        "split": "train",
                    },
                )
                write_jsonl_record(
                    handle,
                    {
                        "text": "epsilon zeta",
                        "source_id": "general_web",
                        "split": "train",
                    },
                )
                write_jsonl_record(
                    handle,
                    {
                        "text": "held out",
                        "source_id": "math",
                        "split": "validation",
                    },
                )

            output_dir = Path(tmpdir) / "tokenized"
            indexes = tokenize_records_by_split(
                input_dir=input_dir,
                output_dir=output_dir,
                tokenizer=_FakeTokenizer(),
                target_shard_tokens=4,
                train_source_token_limits={"general_web": 6},
                workers=2,
                chunk_records=1,
            )

            self.assertEqual(indexes["train"]["total_tokens"], 5)
            self.assertEqual(indexes["train"]["source_token_counts"], {
                "general_web": 5,
            })
            self.assertEqual(indexes["validation"]["total_tokens"], 3)
            self.assertEqual(
                [shard["tokens"] for shard in indexes["train"]["shards"]],
                [4, 1],
            )

    def test_tokenize_worker_batches_records_when_tokenizer_supports_batch_call(self):
        tokenizer = _BatchFakeTokenizer()
        tokenize_module._initialize_tokenizer_object_worker(tokenizer)

        _, records = tokenize_module._tokenize_record_chunk_worker(
            3,
            [
                {"text": "alpha beta", "source_id": "math", "split": "train"},
                {"text": "gamma delta", "source_id": "code", "split": "validation"},
            ],
            True,
        )

        self.assertEqual(tokenizer.batch_calls, 1)
        self.assertEqual(tokenizer.encode_calls, 0)
        self.assertEqual(
            records,
            [
                ("train", "math", [5, 4, 0]),
                ("validation", "code", [5, 5, 0]),
            ],
        )

    def test_parallel_tokenization_respects_max_pending_chunks_and_reports_progress(self):
        with TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "processed"
            input_dir.mkdir()
            with (input_dir / "mixed.jsonl").open("w", encoding="utf-8") as handle:
                for index in range(6):
                    write_jsonl_record(
                        handle,
                        {
                            "text": f"document {index} alpha beta",
                            "source_id": "general_web",
                            "split": "train",
                        },
                    )

            events = []
            indexes = tokenize_records_by_split(
                input_dir=input_dir,
                output_dir=Path(tmpdir) / "tokenized",
                tokenizer=_FakeTokenizer(),
                target_shard_tokens=64,
                workers=2,
                chunk_records=1,
                max_pending_chunks=3,
                progress_callback=events.append,
            )

            self.assertEqual(indexes["train"]["total_tokens"], 30)
            submitted = [
                event
                for event in events
                if event.get("event") == "tokenize_chunk_submitted"
            ]
            completed = [
                event
                for event in events
                if event.get("event") == "tokenize_chunk_completed"
            ]
            consumed = [
                event
                for event in events
                if event.get("event") == "tokenize_chunk_consumed"
            ]
            self.assertEqual(len(submitted), 6)
            self.assertEqual(len(completed), 6)
            self.assertEqual(len(consumed), 6)
            self.assertLessEqual(
                max(event["pending_chunks"] for event in submitted),
                3,
            )

    def test_load_tokenized_index_validates_hashes_and_reads_tokens(self):
        with TemporaryDirectory() as tmpdir:
            writer = TokenShardWriter(
                output_dir=Path(tmpdir),
                target_shard_tokens=4,
            )
            writer.add([10, 20, 30], source_id="sample")
            writer.close()

            index = load_tokenized_index(Path(tmpdir) / "index.json")
            self.assertEqual(index["total_tokens"], 3)
            self.assertEqual(
                read_token_shard(Path(tmpdir), index["shards"][0]),
                [10, 20, 30],
            )

    def test_load_tokenized_index_rejects_hash_mismatch(self):
        with TemporaryDirectory() as tmpdir:
            writer = TokenShardWriter(
                output_dir=Path(tmpdir),
                target_shard_tokens=4,
            )
            writer.add([10, 20, 30], source_id="sample")
            writer.close()
            shard_path = Path(tmpdir) / "tokens-000000.bin"
            shard_path.write_bytes(b"\x1e\x00\x00\x00\x14\x00\x00\x00\x0a\x00\x00\x00")

            with self.assertRaisesRegex(TokenizedIndexError, "hash mismatch"):
                load_tokenized_index(Path(tmpdir) / "index.json")

    def test_load_tokenized_index_metadata_mode_skips_full_hashing(self):
        with TemporaryDirectory() as tmpdir:
            writer = TokenShardWriter(
                output_dir=Path(tmpdir),
                target_shard_tokens=4,
            )
            writer.add([10, 20, 30], source_id="sample")
            writer.close()
            shard_path = Path(tmpdir) / "tokens-000000.bin"
            shard_path.write_bytes(b"\x63\x00\x00\x00\x62\x00\x00\x00\x61\x00\x00\x00")

            index = load_tokenized_index(
                Path(tmpdir) / "index.json",
                validation_mode="metadata",
            )

            self.assertEqual(index["total_tokens"], 3)

    def test_shuffle_tokenized_corpus_interleaves_existing_source_blocks(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "tokenized" / "train"
            writer = TokenShardWriter(
                output_dir=input_dir,
                target_shard_tokens=4,
            )
            writer.add([101, 102, 103, 104], source_id="code")
            writer.add([201, 202, 203, 204], source_id="math")
            writer.add([301, 302, 303, 304], source_id="general_web")
            writer.close()

            output_dir = root / "tokenized-shuffled" / "train"
            shuffled = shuffle_tokenized_corpus(
                input_index_path=input_dir / "index.json",
                output_dir=output_dir,
                seed=17,
                chunk_tokens=2,
                target_shard_tokens=4,
                validation_mode="metadata",
            )

            self.assertEqual(shuffled["total_tokens"], 12)
            self.assertEqual(
                shuffled["source_token_counts"],
                {"code": 4, "general_web": 4, "math": 4},
            )
            self.assertTrue((output_dir / "index.json").exists())
            reloaded = load_tokenized_index(output_dir / "index.json")
            self.assertEqual(reloaded["total_tokens"], 12)

            output_tokens = [
                token
                for shard in reloaded["shards"]
                for token in read_token_shard(output_dir, shard)
            ]
            self.assertCountEqual(
                [tuple(output_tokens[index : index + 2]) for index in range(0, 12, 2)],
                [
                    (101, 102),
                    (103, 104),
                    (201, 202),
                    (203, 204),
                    (301, 302),
                    (303, 304),
                ],
            )
            first_three_chunk_sources = {
                "code" if token < 200 else "math" if token < 300 else "general_web"
                for token in output_tokens[:6:2]
            }
            self.assertEqual(first_three_chunk_sources, {"code", "math", "general_web"})
            self.assertNotEqual(
                output_tokens,
                [101, 102, 103, 104, 201, 202, 203, 204, 301, 302, 303, 304],
            )


class ScriptPackagingTests(unittest.TestCase):
    def test_full_data_launcher_uses_env_token_without_embedding_secret(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_full_data.sh"
        script = script_path.read_text(encoding="utf-8")

        self.assertIn(
            "/home/qhgao/.workspace/projects/fnlp-summer-training-2026/assignment-3",
            script,
        )
        self.assertIn("HF_TOKEN", script)
        self.assertIn("python -m scaling_data.prepare", script)
        self.assertIn("SCALING_DATA_PACKAGE_ROOT", script)
        self.assertIn("SCALING_MAX_DOCS_PER_SOURCE", script)
        self.assertIn("SCALING_PROGRESS_EVERY_DOCS", script)
        self.assertIn("SCALING_DOWNLOAD_TOKEN_OVERFETCH_RATIO", script)
        self.assertIn("SCALING_DOWNLOAD_CHARS_PER_TOKEN", script)
        self.assertIn("SCALING_DOWNLOAD_WORKERS", script)
        self.assertIn("SCALING_DOWNLOAD_SHARDS_PER_SOURCE", script)
        self.assertIn("SCALING_DOWNLOAD_SOURCE_PARALLELISM", script)
        self.assertIn("SCALING_TOKENIZE_MAX_PENDING_CHUNKS", script)
        self.assertIn("SCALING_PROCESSED_SHUFFLE_SEED", script)
        self.assertIn("SCALING_PROCESSED_SHUFFLE_BUCKET_COUNT", script)
        self.assertIn("SCALING_NO_SHUFFLE_PROCESSED_RECORDS", script)
        self.assertIn("SCALING_NO_RESUME_DOWNLOADS", script)
        self.assertIn("execution_root/data_work", script)
        self.assertNotIn("project_rootdir/data_work", script)
        self.assertNotIn("hf_", script)

    def test_full_data_launcher_defaults_work_dir_to_invocation_cwd(self):
        data_root = Path(__file__).resolve().parents[1]
        project_root = data_root.parents[1]
        script_path = data_root / "scripts" / "prepare_full_data.sh"

        with TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            env = {
                **os.environ,
                "PROJECT_ROOTDIR": str(project_root),
                "HF_TOKEN": "dummy-token",
            }
            env.pop("SCALING_DATA_WORK_DIR", None)
            result = subprocess.run(
                [str(script_path), "--plan-only"],
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

            summary = json.loads(result.stdout)
            self.assertEqual(summary["work_dir"], str(cwd / "data_work"))
            self.assertTrue(Path(summary["pipeline_plan_path"]).exists())

    def test_full_data_launcher_can_run_from_standalone_data_package(self):
        data_root = Path(__file__).resolve().parents[1]
        script_path = data_root / "scripts" / "prepare_full_data.sh"

        with TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            env = {**os.environ, "HF_TOKEN": "dummy-token"}
            env.pop("PROJECT_ROOTDIR", None)
            env.pop("SCALING_DATA_WORK_DIR", None)
            env.pop("SCALING_DATA_PACKAGE_ROOT", None)
            result = subprocess.run(
                [str(script_path), "--plan-only"],
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

            summary = json.loads(result.stdout)
            self.assertEqual(summary["work_dir"], str(cwd / "data_work"))
            self.assertEqual(summary["manifest_name"], "general_100b_mix")
            self.assertTrue(Path(summary["blend_plan_path"]).exists())

    def test_tokenized_shuffle_script_runs_from_existing_data_work(self):
        data_root = Path(__file__).resolve().parents[1]
        script_path = data_root / "scripts" / "shuffle_tokenized_train.sh"
        script = script_path.read_text(encoding="utf-8")

        self.assertIn("SCALING_TOKENIZED_INPUT_INDEX", script)
        self.assertIn("SCALING_TOKENIZED_OUTPUT_DIR", script)
        self.assertIn("SCALING_SHUFFLED_TOKENIZED_TRAIN_DIR", script)

        with TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            input_dir = cwd / "data_work" / "tokenized" / "train"
            writer = TokenShardWriter(
                output_dir=input_dir,
                target_shard_tokens=4,
            )
            writer.add([101, 102, 103, 104], source_id="code")
            writer.add([201, 202, 203, 204], source_id="math")
            writer.close()

            env = {
                **os.environ,
                "SCALING_DATA_PACKAGE_ROOT": str(data_root),
                "SCALING_TOKENIZED_SHUFFLE_SEED": "17",
                "SCALING_TOKENIZED_SHUFFLE_CHUNK_TOKENS": "2",
                "SCALING_TARGET_SHARD_TOKENS": "4",
            }
            result = subprocess.run(
                [str(script_path)],
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

            summary = json.loads(result.stdout)
            output_index = cwd / "data_work" / "tokenized_shuffled" / "train" / "index.json"
            self.assertEqual(summary["index_path"], str(output_index))
            reloaded = load_tokenized_index(output_index)
            self.assertEqual(reloaded["total_tokens"], 8)
            self.assertEqual(
                reloaded["source_token_counts"],
                {"code": 4, "math": 4},
            )

    def test_rebuild_tokenized_launcher_uses_existing_processed_dir(self):
        data_root = Path(__file__).resolve().parents[1]
        script_path = data_root / "scripts" / "rebuild_tokenized_from_processed.sh"
        script = script_path.read_text(encoding="utf-8")

        self.assertIn("SCALING_PROCESSED_JSONL_DIR", script)
        self.assertIn("python -m scaling_data.rebuild_tokenized", script)
        self.assertIn("SCALING_PROCESSED_SHUFFLE_SEED", script)
        self.assertIn("SCALING_TOKENIZE_WORKERS", script)
        self.assertIn("SCALING_TOKENIZE_MAX_PENDING_CHUNKS", script)
        self.assertIn("SCALING_TOKENIZE_PROGRESS_EVERY_CHUNKS", script)
        self.assertIn("SCALING_FORCE_RESHUFFLE_PROCESSED_RECORDS", script)
        self.assertIn("SCALING_NO_PROGRESS", script)
        self.assertIn("SCALING_PROCESSED_SHUFFLE_MAX_OPEN_BUCKETS:-0", script)
        self.assertNotIn("HF_TOKEN", script)

    def test_pyproject_exposes_processed_shuffle_and_rebuild_entrypoints(self):
        data_root = Path(__file__).resolve().parents[1]
        pyproject = (data_root / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn(
            'scaling-shuffle-processed-data = "scaling_data.shuffle_processed:main"',
            pyproject,
        )
        self.assertIn(
            'scaling-rebuild-tokenized-data = "scaling_data.rebuild_tokenized:main"',
            pyproject,
        )

    def test_deployment_bundle_contains_only_data_package_artifacts(self):
        data_root = Path(__file__).resolve().parents[1]
        script_path = data_root / "scripts" / "build_deployment_bundle.sh"

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bundle.tar.gz"
            result = subprocess.run(
                [str(script_path), str(output_path)],
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertEqual(result.stdout.strip(), str(output_path))
            with tarfile.open(output_path, "r:gz") as archive:
                names = set(archive.getnames())

            self.assertIn("data/scaling_data/prepare.py", names)
            self.assertIn("data/scaling_data/rebuild_tokenized.py", names)
            self.assertIn("data/scaling_data/shuffle_processed.py", names)
            self.assertIn("data/scaling_data/shuffle_tokenized.py", names)
            self.assertIn("data/scripts/prepare_full_data.sh", names)
            self.assertIn("data/scripts/rebuild_tokenized_from_processed.sh", names)
            self.assertIn("data/scripts/shuffle_tokenized_train.sh", names)
            self.assertIn("data/manifests/general_100b_mix.json", names)
            self.assertNotIn("data/scaling_data/__pycache__", names)
            self.assertFalse(any(name.endswith(".pyc") for name in names))


def _documents_covering_both_splits(*, validation_rate: float) -> list[str]:
    selected = {"train": None, "validation": None}
    index = 0
    while selected["train"] is None or selected["validation"] is None:
        text = (
            f"This is a deterministic sample document number {index} with "
            "enough characters for processing."
        )
        split = assign_split(text, validation_rate=validation_rate)
        if selected[split] is None:
            selected[split] = text
        index += 1
    return [selected["train"], selected["validation"]]


if __name__ == "__main__":
    unittest.main()
