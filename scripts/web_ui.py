#!/usr/bin/env python3
"""Start a localhost PP-UIE UI whose model is loaded once and kept resident."""

import argparse
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pp_uie.backend import RuntimeConfig
from pp_uie.models import MODEL_SPECS, validate_model_dir
from pp_uie.webserver import make_handler
from pp_uie.webui import SwitchingExtractionService, WebApplication, load_text_samples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the persistent-model PP-UIE local web interface")
    parser.add_argument("--model", choices=["0.5b", "1.5b"], default="0.5b")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--open", action="store_true", help="Open the page in the default browser after loading")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    configs = {}
    for model_name, spec in MODEL_SPECS.items():
        model_path = (
            args.model_path if model_name == args.model and args.model_path
            else ROOT / "models" / spec.directory
        ).resolve()
        validate_model_dir(model_path)
        configs[model_name] = RuntimeConfig(
            model_path, "cpu", "float32", 1, args.max_length, args.max_new_tokens
        )
    os.environ.setdefault("PPNLP_HOME", str(ROOT / ".paddlenlp"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("AISTUDIO_OFFLINE", "1")

    print(f"Loading initial PP-UIE-{args.model} from {configs[args.model].model_path} ...", flush=True)
    service = SwitchingExtractionService.create(
        configs,
        initial_model=args.model,
    )
    samples = load_text_samples(ROOT / "testdata/pocket_user_six_samples.jsonl")
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
