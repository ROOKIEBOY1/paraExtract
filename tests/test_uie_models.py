import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import text_extractor.backends.uie.models as uie_models
from text_extractor.backends.uie import (
    UIE_MODEL_SPECS,
    download_uie_mini,
    validate_uie_model_dir,
)
from text_extractor.backends.uie.models import (
    UIE_MINI_FILES,
    UIEModelSpec,
)


REQUIRED_FILES = {
    "model_state.pdparams",
    "config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_required_files(path):
    path.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_FILES:
        (path / filename).write_bytes(f"fixture-{filename}".encode())


def test_uie_model_requires_all_official_resources(tmp_path):
    assert set(UIE_MINI_FILES) == REQUIRED_FILES
    _write_required_files(tmp_path)
    (tmp_path / "vocab.txt").unlink()

    with pytest.raises(FileNotFoundError, match="vocab.txt"):
        validate_uie_model_dir(tmp_path)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"bytes": 999}, "size mismatch"),
        ({"sha256": "0" * 64}, "sha256 mismatch"),
    ],
)
def test_uie_model_rejects_manifest_size_or_digest_mismatch(tmp_path, metadata, message):
    _write_required_files(tmp_path)
    (tmp_path / "download_manifest.json").write_text(
        json.dumps({"files": {"config.json": metadata}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        validate_uie_model_dir(tmp_path)


def test_uie_download_records_url_size_sha256_and_official_md5(tmp_path, monkeypatch):
    body = b"downloaded-model"
    expected_md5 = hashlib.md5(body).hexdigest()
    expected_sha256 = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(
        uie_models,
        "UIE_MINI_FILES",
        {"model_state.pdparams": {"url": "https://example.invalid/model", "md5": expected_md5}},
    )

    class Response:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield body

    class Session:
        def get(self, url, **kwargs):
            assert url == "https://example.invalid/model"
            return Response()

    manifest = download_uie_mini(tmp_path, Session())
    model_dir = tmp_path / "UIE-mini"
    entry = manifest["files"]["model_state.pdparams"]

    assert entry == {
        "url": "https://example.invalid/model",
        "bytes": len(body),
        "md5": expected_md5,
        "sha256": expected_sha256,
    }
    assert json.loads((model_dir / "download_manifest.json").read_text())["files"] == manifest["files"]
    assert validate_uie_model_dir(model_dir).valid is True


def test_uie_download_rejects_official_md5_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        uie_models,
        "UIE_MINI_FILES",
        {"config.json": {"url": "https://example.invalid/config", "md5": "0" * 32}},
    )

    class Response:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"wrong"

    class Session:
        def get(self, url, **kwargs):
            return Response()

    with pytest.raises(ValueError, match="md5 mismatch"):
        download_uie_mini(tmp_path, Session())


def test_download_cli_dry_run_supports_uie_mini():
    result = subprocess.run(
        [sys.executable, "scripts/download_models.py", "--model", "uie-mini", "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == {
        metadata["url"] for metadata in UIE_MINI_FILES.values()
    }


def test_download_cli_all_includes_uie_mini():
    result = subprocess.run(
        [sys.executable, "scripts/download_models.py", "--model", "all", "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert UIE_MINI_FILES["model_state.pdparams"]["url"] in result.stdout.splitlines()


def test_uie_base_uses_the_installed_taskflow_official_resources():
    spec = UIE_MODEL_SPECS["uie-base"]

    assert spec.directory == "UIE-base"
    assert set(spec.files) == REQUIRED_FILES
    assert spec.files["model_state.pdparams"] == {
        "url": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.1/model_state.pdparams",
        "md5": "47b93cf6a85688791699548210048085",
    }
    assert spec.files["config.json"] == {
        "url": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/config.json",
        "md5": "ad8b5442c758fb2dc18ea53b61e867f7",
    }


def test_download_cli_dry_run_supports_uie_base():
    result = subprocess.run(
        [sys.executable, "scripts/download_models.py", "--model", "uie-base", "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == {
        metadata["url"] for metadata in UIE_MODEL_SPECS["uie-base"].files.values()
    }


def test_download_cli_all_includes_uie_base():
    result = subprocess.run(
        [sys.executable, "scripts/download_models.py", "--model", "all", "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert UIE_MODEL_SPECS["uie-base"].files["model_state.pdparams"]["url"] in result.stdout.splitlines()


def test_validation_checks_official_md5_without_download_manifest(tmp_path, monkeypatch):
    _write_required_files(tmp_path)
    files = {
        name: {
            "url": f"https://example.invalid/{name}",
            "md5": hashlib.md5((tmp_path / name).read_bytes()).hexdigest(),
        }
        for name in REQUIRED_FILES
    }
    monkeypatch.setitem(
        UIE_MODEL_SPECS,
        "fixture-uie",
        UIEModelSpec("fixture-uie", "Fixture-UIE", files),
    )
    (tmp_path / "config.json").write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="official md5 mismatch for config.json"):
        validate_uie_model_dir(tmp_path, "fixture-uie")


def test_validation_rejects_manifest_for_different_uie_model(tmp_path):
    _write_required_files(tmp_path)
    (tmp_path / "download_manifest.json").write_text(
        json.dumps({"model_id": "uie-mini", "files": {}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="manifest model mismatch"):
        validate_uie_model_dir(tmp_path, "uie-base")
