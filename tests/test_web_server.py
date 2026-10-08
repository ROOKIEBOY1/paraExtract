import json
from http.server import ThreadingHTTPServer
import threading
from urllib.request import Request, urlopen

import pytest

from pp_uie.webserver import make_handler


def test_http_server_serves_page_config_and_extraction(tmp_path):
    page = tmp_path / "index.html"
    page.write_text("<h1>fixture page</h1>", encoding="utf-8")

    class App:
        def config(self):
            return {"model": "fixture"}

        def extract(self, payload):
            return {"received": payload}

        def switch_model(self, payload):
            return {"switched": payload["model"]}

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(), page))
    except PermissionError:
        pytest.skip("sandbox does not permit binding a localhost test port")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/") as response:
            assert response.read().decode() == "<h1>fixture page</h1>"
        with urlopen(base + "/api/config") as response:
            assert json.load(response) == {"model": "fixture"}
        request = Request(
            base + "/api/extract",
            data=json.dumps({"text": "x", "fields": ["ISO"]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            assert json.load(response) == {"received": {"text": "x", "fields": ["ISO"]}}
        switch_request = Request(
            base + "/api/model",
            data=json.dumps({"model": "1.5b"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(switch_request) as response:
            assert json.load(response) == {"switched": "1.5b"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
