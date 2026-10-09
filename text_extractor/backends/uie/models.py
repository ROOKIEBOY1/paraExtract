"""Official Taskflow UIE resource manifests, downloads, and validation."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from ...core.model_files import ValidationReport, _download_file, _sha256


UIE_MINI_DIRECTORY = "UIE-mini"
UIE_BASE_DIRECTORY = "UIE-base"
UIE_MINI_FILES = {
    "model_state.pdparams": {
        "url": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini_v1.1/model_state.pdparams",
        "md5": "9a0805762c41b104d590c15fbe9b19fd",
    },
    "config.json": {
        "url": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini/config.json",
        "md5": "8ddebbf64c3f32a49e6f9e1c220e7322",
    },
    "vocab.txt": {
        "url": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
        "md5": "1c1c1f4fd93c5bed3b4eebec4de976a8",
    },
    "special_tokens_map.json": {
        "url": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
        "md5": "8b3fb1023167bb4ab9d70708eb05f6ec",
    },
    "tokenizer_config.json": {
        "url": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        "md5": "59acb0ce78e79180a2491dfd8382b28c",
    },
}

UIE_BASE_FILES = {
    "model_state.pdparams": {
        "url": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.1/model_state.pdparams",
        "md5": "47b93cf6a85688791699548210048085",
    },
    "config.json": {
        "url": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/config.json",
        "md5": "ad8b5442c758fb2dc18ea53b61e867f7",
    },
    "vocab.txt": UIE_MINI_FILES["vocab.txt"],
    "special_tokens_map.json": UIE_MINI_FILES["special_tokens_map.json"],
    "tokenizer_config.json": UIE_MINI_FILES["tokenizer_config.json"],
}


@dataclass(frozen=True)
class UIEModelSpec:
    model_id: str
    directory: str
    files: dict[str, dict[str, str]]


UIE_MODEL_SPECS = {
    "uie-mini": UIEModelSpec("uie-mini", UIE_MINI_DIRECTORY, UIE_MINI_FILES),
    "uie-base": UIEModelSpec("uie-base", UIE_BASE_DIRECTORY, UIE_BASE_FILES),
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec_for(model_id: str) -> UIEModelSpec:
    if model_id == "uie-mini":
        return UIEModelSpec("uie-mini", UIE_MINI_DIRECTORY, UIE_MINI_FILES)
    try:
        return UIE_MODEL_SPECS[model_id]
    except KeyError as exc:
        raise ValueError(f"unsupported UIE model: {model_id}") from exc


def validate_uie_model_dir(path: Path, model_id: str = "uie-mini") -> ValidationReport:
    path = Path(path)
    spec = _spec_for(model_id)
    missing = [name for name in spec.files if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {model_id} model files: {', '.join(missing)}")

    manifest_path = path / "download_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_model = manifest.get("model_id")
        if manifest_model is not None and manifest_model != spec.model_id:
            raise ValueError(
                f"manifest model mismatch: expected {spec.model_id}, got {manifest_model}"
            )
        entries = manifest.get("files", manifest)
        for name, metadata in entries.items():
            file_path = path / name
            if not file_path.is_file():
                continue
            if "bytes" in metadata and file_path.stat().st_size != metadata["bytes"]:
                raise ValueError(
                    f"size mismatch for {name}: expected {metadata['bytes']}, got {file_path.stat().st_size}"
                )
            if metadata.get("sha256") and _sha256(file_path) != metadata["sha256"]:
                raise ValueError(f"sha256 mismatch for {name}")
            if metadata.get("md5") and _md5(file_path) != metadata["md5"]:
                raise ValueError(f"md5 mismatch for {name}")
        missing_entries = [name for name in spec.files if name not in entries]
        if missing_entries:
            raise ValueError(
                f"manifest missing files: {', '.join(missing_entries)}"
            )

    for name, resource in spec.files.items():
        if _md5(path / name) != resource["md5"]:
            raise ValueError(f"official md5 mismatch for {name}")

    files = tuple(sorted(item.name for item in path.iterdir() if item.is_file()))
    total_bytes = sum((path / name).stat().st_size for name in spec.files)
    return ValidationReport(True, total_bytes, files)


def download_uie_model(model_id: str, destination: Path, session: Any) -> dict[str, Any]:
    spec = _spec_for(model_id)
    model_dir = Path(destination) / spec.directory
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "download_manifest.json"
    entries: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        entries.update(json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {}))

    for filename, resource in spec.files.items():
        target = model_dir / filename
        existing = entries.get(filename, {})
        complete = (
            target.is_file()
            and target.stat().st_size == existing.get("bytes")
            and _sha256(target) == existing.get("sha256")
            and _md5(target) == resource["md5"]
        )
        if not complete:
            metadata = _download_file(resource["url"], target, session)
            actual_md5 = _md5(target)
            if actual_md5 != resource["md5"]:
                target.unlink(missing_ok=True)
                raise ValueError(
                    f"md5 mismatch for {filename}: expected {resource['md5']}, got {actual_md5}"
                )
            entries[filename] = {**metadata, "md5": actual_md5}
        manifest = {
            "model_id": spec.model_id,
            "directory": spec.directory,
            "files": entries,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    validate_uie_model_dir(model_dir, spec.model_id)
    return manifest


def download_uie_mini(destination: Path, session: Any) -> dict[str, Any]:
    return download_uie_model("uie-mini", destination, session)
