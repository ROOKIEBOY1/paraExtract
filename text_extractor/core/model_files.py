"""Model-independent file validation and resumable download helpers."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    total_bytes: int
    files: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(
    url: str, target: Path, session: Any, max_attempts: int = 3
) -> dict[str, Any]:
    part = target.with_name(target.name + ".part")
    last_error = None
    for _ in range(max_attempts):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(
                url, stream=True, timeout=(30, 300), headers=headers
            ) as response:
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
                        raise IOError(
                            f"incomplete ranged download: {part.stat().st_size}/{expected_total}"
                        )
            part.replace(target)
            return {
                "url": url,
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"download failed after {max_attempts} attempts: {url}"
    ) from last_error
