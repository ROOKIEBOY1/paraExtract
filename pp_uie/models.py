"""Official PP-UIE model manifests, downloads, and local validation."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


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
class ValidationReport:
    valid: bool
    total_bytes: int
    files: tuple[str, ...]


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    directory: str
    files: dict[str, dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _download_file(url: str, target: Path, session: Any, max_attempts: int = 3) -> dict[str, Any]:
    part = target.with_name(target.name + ".part")
    last_error = None
    for _ in range(max_attempts):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(url, stream=True, timeout=(30, 300), headers=headers) as response:
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                mode = "ab" if resumed else "wb"
                with part.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                content_range = response.headers.get("Content-Range", "")
                if "/" in content_range:
                    expected_total = int(content_range.rsplit("/", 1)[1])
                    if part.stat().st_size != expected_total:
                        raise IOError(f"incomplete ranged download: {part.stat().st_size}/{expected_total}")
            part.replace(target)
            return {"url": url, "bytes": target.stat().st_size, "sha256": _sha256(target)}
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"download failed after {max_attempts} attempts: {url}") from last_error


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
