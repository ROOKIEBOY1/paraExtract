import json

import pytest

from pp_uie.models import _download_file, validate_model_dir


def _base_files(path):
    for name in ["config.json", "generation_config.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"]:
        (path / name).write_text("{}" if name.endswith(".json") else "fixture", encoding="utf-8")


def test_missing_indexed_shard_fails_before_load(tmp_path):
    _base_files(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}
    }))
    with pytest.raises(FileNotFoundError, match="model-00001-of-00001.safetensors"):
        validate_model_dir(tmp_path)


def test_complete_directory_reports_total_bytes(tmp_path):
    _base_files(tmp_path)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": shard.name}}))
    report = validate_model_dir(tmp_path)
    assert report.valid is True
    assert report.total_bytes >= len(b"weights")


def test_manifest_size_mismatch_fails_validation(tmp_path):
    _base_files(tmp_path)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"short")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": shard.name}}))
    (tmp_path / "download_manifest.json").write_text(json.dumps({shard.name: {"bytes": 99}}))
    with pytest.raises(ValueError, match="size mismatch"):
        validate_model_dir(tmp_path)


def test_download_resumes_existing_part_with_http_range(tmp_path):
    target = tmp_path / "weights.bin"
    part = tmp_path / "weights.bin.part"
    part.write_bytes(b"abc")

    class Response:
        status_code = 206
        headers = {"Content-Range": "bytes 3-5/6"}

        def __enter__(self): return self
        def __exit__(self, *args): return None
        def raise_for_status(self): return None
        def iter_content(self, chunk_size): yield b"def"

    class Session:
        def get(self, url, **kwargs):
            assert kwargs["headers"] == {"Range": "bytes=3-"}
            return Response()

    metadata = _download_file("https://example.invalid/weights.bin", target, Session())
    assert target.read_bytes() == b"abcdef"
    assert metadata["bytes"] == 6
