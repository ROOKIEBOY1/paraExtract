#!/usr/bin/env python3
"""Start a localhost extraction UI that keeps one selected model resident."""

import argparse
from dataclasses import replace
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pp_uie.backend import PPUIEBackend, RuntimeConfig
from pp_uie.model_registry import ModelDefinition, index_definitions
from pp_uie.models import MODEL_SPECS, validate_model_dir
from pp_uie.uie_backend import UIETaskflowBackend, UIETaskflowConfig
from pp_uie.uie_models import UIE_MODEL_SPECS, validate_uie_model_dir
from pp_uie.webserver import make_handler
from pp_uie.webui import SwitchingExtractionService, WebApplication, load_web_samples


def build_model_definitions(
    root: Path, max_length: int, max_new_tokens: int
) -> list[ModelDefinition]:
    root = Path(root)
    definitions = []
    for model_id, spec in MODEL_SPECS.items():
        definitions.append(ModelDefinition(
            model_id=model_id,
            label=spec.directory,
            config=RuntimeConfig(
                (root / "models" / spec.directory).resolve(),
                "cpu",
                "float32",
                1,
                max_length,
                max_new_tokens,
            ),
            backend_factory=PPUIEBackend,
        ))
    for model_id, spec in UIE_MODEL_SPECS.items():
        definitions.append(ModelDefinition(
            model_id=model_id,
            label=spec.directory,
            config=UIETaskflowConfig(
                (root / "models" / spec.directory).resolve(),
                device="cpu",
                batch_size=1,
                max_seq_len=max_length,
                position_prob=0.5,
                model=model_id,
            ),
            backend_factory=UIETaskflowBackend,
        ))
    return definitions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the persistent-model local extraction interface")
    parser.add_argument(
        "--model", choices=["0.5b", "1.5b", *UIE_MODEL_SPECS], default="0.5b"
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--open", action="store_true", help="Open the page in the default browser after loading")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    definitions = build_model_definitions(ROOT, args.max_length, args.max_new_tokens)
    if args.model_path:
        definitions = [
            replace(
                definition,
                config=replace(definition.config, model_path=args.model_path.resolve()),
            ) if definition.model_id == args.model else definition
            for definition in definitions
        ]
    for definition in definitions:
        if definition.model_id in UIE_MODEL_SPECS:
            validate_uie_model_dir(definition.config.model_path, definition.model_id)
        else:
            validate_model_dir(definition.config.model_path)
    os.environ.setdefault("PPNLP_HOME", str(ROOT / ".paddlenlp"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("AISTUDIO_OFFLINE", "1")

    indexed = index_definitions(definitions)
    initial = indexed[args.model]
    print(f"Loading initial {initial.label} from {initial.config.model_path} ...", flush=True)
    service = SwitchingExtractionService.create(
        definitions,
        initial_model=args.model,
    )
    samples = load_web_samples(
        ROOT / "testdata/pocket_user_six_samples.jsonl",
        ROOT / "testdata/pocket_edge_cases.jsonl",
    )
    application = WebApplication(service, samples)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(application, ROOT / "web/index.html"))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Model ready in {service.load_ms / 1000:.2f}s. Open {url}", flush=True)
    print("Only the selected model remains loaded; switching replaces it in memory.", flush=True)
    if args.open:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server ...", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
