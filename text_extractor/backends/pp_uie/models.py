"""Official PP-UIE model manifests, downloads, and local validation."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from ...core.model_files import ValidationReport, _download_file, _sha256


MODEL_FILES = (
    "added_tokens.json", "config.json", "generation_config.json", "merges.txt",
    "model-00001-of-00001.safetensors", "model.safetensors.index.json",
    "special_tokens_map.json", "tokenizer_config.json", "vocab.json",
)
BASE_URL = "https://paddlenlp.bj.bcebos.com/models/community"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    directory: str
    approximate_bytes: int

    def url(self, filename: str) -> str:
        return f"{BASE_URL}/{self.model_id}/{filename}"


MODEL_SPECS = {
    "0.5b": ModelSpec("paddlenlp/PP-UIE-0.5B", "PP-UIE-0.5B", 942_300_000),
    "1.5b": ModelSpec("paddlenlp/PP-UIE-1.5B", "PP-UIE-1.5B", 2_900_000_000),
}


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    directory: str
    files: dict[str, dict[str, Any]]


def validate_model_dir(path: Path) -> ValidationReport:
    path = Path(path)
    missing = [name for name in MODEL_FILES if not (path / name).is_file()]
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for shard in sorted(set(index.get("weight_map", {}).values())):
            if not (path / shard).is_file():
                raise FileNotFoundError(f"missing indexed shard: {shard}")
    if missing:
        raise FileNotFoundError(f"missing model files: {', '.join(missing)}")

    manifest_path = path / "download_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("files", manifest)
        for name, metadata in entries.items():
            file_path = path / name
            if file_path.is_file() and "bytes" in metadata and file_path.stat().st_size != metadata["bytes"]:
                raise ValueError(f"size mismatch for {name}: expected {metadata['bytes']}, got {file_path.stat().st_size}")
            if file_path.is_file() and metadata.get("sha256") and _sha256(file_path) != metadata["sha256"]:
                raise ValueError(f"sha256 mismatch for {name}")
    files = tuple(sorted(item.name for item in path.iterdir() if item.is_file()))
    return ValidationReport(True, sum((path / name).stat().st_size for name in files), files)


def download_model(spec: ModelSpec, destination: Path, session: Any) -> ModelManifest:
    model_dir = Path(destination) / spec.directory
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "download_manifest.json"
    entries: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        entries.update(json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {}))
    for filename in MODEL_FILES:
        url = spec.url(filename)
        target = model_dir / filename
        existing = entries.get(filename, {})
        if not (target.is_file() and target.stat().st_size == existing.get("bytes") and _sha256(target) == existing.get("sha256")):
            entries[filename] = _download_file(url, target, session)
        manifest_path.write_text(
            json.dumps({"model_id": spec.model_id, "files": entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    validate_model_dir(model_dir)
    return ModelManifest(spec.model_id, spec.directory, entries)


def manifest_as_dict(manifest: ModelManifest) -> dict[str, Any]:
    return asdict(manifest)
