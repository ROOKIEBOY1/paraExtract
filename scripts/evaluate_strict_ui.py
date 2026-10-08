#!/usr/bin/env python3
"""Run every bundled Pocket sample through the resident strict-mode API."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def get_json(url: str):
    with urlopen(url) as response:
        return json.load(response)


def post_json(url: str, payload):
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output", type=Path, default=Path("results/strict_mode_evaluation.json"))
    args = parser.parse_args()

    config = get_json(f"{args.url}/api/config")
    fields = config["field_groups"][0]["fields"]
    records = []
    for sample in config["samples"]:
        result = post_json(
            f"{args.url}/api/extract",
            {"text": sample["text"], "fields": fields, "strict": True},
        )
        records.append({"id": sample["id"], **result})
        print(
            f"{sample['id']}: verified={sum(map(len, result['verified_output'].values()))} "
            f"rejected={len(result['rejected_candidates'])} "
            f"inference={result['timing_ms']['inference']:.1f}ms",
            flush=True,
        )

    artifact = {
        "model": config["model"],
        "model_load_ms": config["load_ms"],
        "strict": True,
        "fields": fields,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
