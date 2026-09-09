"""Persistent HTTP backend or router; each backend loads its model once."""
import json
import os
import sys
import threading
import urllib.request
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROLE = os.environ.get("ROLE", "router")
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", "8000"))
STATE = {"status": "loading", "backend": ROLE}
LOCK = threading.Lock()
URLS = {
    "gpu": f"http://gpu-backend:{SERVICE_PORT}",
    "cpu": f"http://cpu-backend:{SERVICE_PORT}",
}


def call(url, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def initialize():
    global STATE
    try:
        from inference import load_model, probe_backend
        threads = int(os.environ.get("CPU_THREADS", "32"))
        profile = probe_backend(ROLE, threads)
        if not profile.get("available") or not profile.get("bfloat16_supported"):
            raise RuntimeError(profile.get("detail", "Backend unavailable"))
        if ROLE == "gpu":
            from pathlib import Path
            weights = sum(p.stat().st_size for p in Path('/model').glob('*.safetensors'))
            required = int(weights * 1.15) + 1024**3
            free = profile['free_memory_bytes']
            if free < required:
                raise RuntimeError(
                    f"GPU memory insufficient: {free / 1024**3:.2f} GiB free; "
                    f"approximately {required / 1024**3:.2f} GiB required. "
                    "CPU fallback remains available."
                )
        load_model(ROLE, "/model", threads)
        STATE = {**probe_backend(ROLE, threads), "status": "ready"}
        print(f"{ROLE}: model loaded and ready for inference", flush=True)
    except Exception as exc:
        STATE = {"status": "unavailable", "backend": ROLE, "detail": str(exc)}
        print(f"{ROLE}: startup failed: {exc}", file=sys.stderr, flush=True)


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            return self.reply(404, {"error": "unknown endpoint"})
        if ROLE != "router":
            return self.reply(200, STATE)
        profiles = {}
        for name, url in URLS.items():
            try:
                profiles[name] = call(url + "/health")
            except Exception as exc:
                profiles[name] = {"status": "unavailable", "detail": str(exc)}
        self.reply(200, {"status": "ready", "backends": profiles})

    def do_POST(self):
        if self.path != "/generate":
            return self.reply(404, {"error": "unknown endpoint"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("Request must contain 1–16384 bytes")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            prompt = payload.get("prompt", "Explain why GPU memory matters.")
            tokens = payload.get("max_new_tokens", 16)
            prefer = payload.get("prefer", "gpu")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a nonempty string")
            if type(tokens) is not int or not 1 <= tokens <= 128:
                raise ValueError("max_new_tokens must be an integer from 1 to 128")
            if prefer not in URLS:
                raise ValueError("prefer must be cpu or gpu")
        except (ValueError, TypeError) as exc:
            return self.reply(400, {"error": str(exc)})
        if ROLE == "router":
            attempts = []
            for name in [prefer, "cpu" if prefer == "gpu" else "gpu"]:
                try:
                    health = call(URLS[name] + "/health")
                    if health.get("status") != "ready":
                        raise RuntimeError(health.get("detail", health.get("status")))
                    result = call(URLS[name] + "/generate", payload, timeout=600)
                    return self.reply(200, {**result, "selected_backend": name, "fallback_reasons": attempts})
                except Exception as exc:
                    attempts.append({"backend": name, "error": str(exc)})
            return self.reply(503, {"error": "No backend completed the request", "attempts": attempts})
        if STATE.get("status") != "ready":
            return self.reply(503, STATE)
        if not LOCK.acquire(blocking=False):
            return self.reply(503, {"error": "backend busy"})
        try:
            from inference import run_inference
            result = run_inference(Namespace(
                backend=ROLE, model_path="/model", cpu_threads=int(os.environ.get("CPU_THREADS", "32")),
                prompt=prompt, max_new_tokens=tokens, warmup=False, simulate_failure=False,
            ))
            self.reply(200, result)
        except Exception as exc:
            self.reply(503, {"error": str(exc)})
        finally:
            LOCK.release()


if __name__ == "__main__":
    if ROLE != "router":
        threading.Thread(target=initialize, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", SERVICE_PORT), Handler).serve_forever()
