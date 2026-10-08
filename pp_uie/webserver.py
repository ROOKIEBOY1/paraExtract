"""Small localhost-only JSON/HTML server for the persistent PP-UIE service."""

from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path


MAX_REQUEST_BYTES = 1024 * 1024


def make_handler(application, page_path: Path):
    page_path = Path(page_path)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value) -> None:
            self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, page_path.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/config":
                self._json(200, application.config())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path not in {"/api/extract", "/api/model"}:
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("request body size is invalid")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/api/model":
                    self._json(200, application.switch_model(payload))
                else:
                    self._json(200, application.extract(payload))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format, *args):
            return

    return Handler
