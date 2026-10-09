#!/usr/bin/env python3
"""Download or validate PP-UIE model snapshots in explicit local directories."""

import argparse
import json
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_extractor.backends.pp_uie import (
    MODEL_FILES,
    MODEL_SPECS,
    download_model,
    manifest_as_dict,
    validate_model_dir,
)
from text_extractor.backends.uie import (
    UIE_MODEL_SPECS,
    download_uie_model,
    validate_uie_model_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["0.5b", "1.5b", *UIE_MODEL_SPECS, "all"], required=True
    )
    parser.add_argument("--destination", type=Path, default=Path("models"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    keys = [*MODEL_SPECS, *UIE_MODEL_SPECS] if args.model == "all" else [args.model]
    manifests = {}
    for key in keys:
        if key in UIE_MODEL_SPECS:
            spec = UIE_MODEL_SPECS[key]
            if args.dry_run:
                for resource in spec.files.values():
                    print(resource["url"])
                continue
            if args.verify_only:
                report = validate_uie_model_dir(args.destination / spec.directory, key)
                manifests[key] = {
                    "valid": report.valid,
                    "total_bytes": report.total_bytes,
                    "files": report.files,
                }
            else:
                manifests[key] = download_uie_model(key, args.destination, requests.Session())
            continue
        spec = MODEL_SPECS[key]
        if args.dry_run:
            for filename in MODEL_FILES:
                print(spec.url(filename))
            continue
        if args.verify_only:
            report = validate_model_dir(args.destination / spec.directory)
            manifests[key] = {"valid": report.valid, "total_bytes": report.total_bytes, "files": report.files}
        else:
            manifests[key] = manifest_as_dict(download_model(spec, args.destination, requests.Session()))
    if not args.dry_run:
        args.destination.mkdir(parents=True, exist_ok=True)
        (args.destination / "manifest.json").write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifests, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
