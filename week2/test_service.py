"""HTTP contract tests with synthetic backend responses (no model required)."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import service


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.server = service.ThreadingHTTPServer(("127.0.0.1", 0), service.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, payload):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server.server_port}/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            return json.load(response)

    def test_gpu_outage_falls_back_to_cpu(self):
        def backend(url, payload=None, timeout=5):
            if "gpu-backend" in url:
                raise ConnectionError("GPU stopped")
            if url.endswith("/health"):
                return {"status": "ready"}
            return {"status": "ok", "backend": "cpu", "text": "test"}

        with patch.object(service, "call", side_effect=backend):
            result = self.request({"prompt": "hello", "max_new_tokens": 4})
        self.assertEqual(result["selected_backend"], "cpu")
        self.assertEqual(result["fallback_reasons"][0]["backend"], "gpu")

    def test_invalid_token_budget_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request({"max_new_tokens": 100000})
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
