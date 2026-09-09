"""Host GPU selection helper. The HTTP router lives in service.py."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def output(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"{command[0]} failed")
    return result.stdout.strip()


def discover_gpu_device(index):
    """Resolve one host GPU index to a UUID-named CDI device; never select all."""
    if not re.fullmatch(r"[0-9]+", index):
        raise ValueError("GPU_INDEX must be a nonnegative integer")
    uuid = output([
        "nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader",
    ])
    if not re.fullmatch(r"GPU-[A-Fa-f0-9-]+", uuid):
        raise RuntimeError(f"Cannot resolve GPU_INDEX={index} to one GPU UUID")

    # The toolkit reads both JSON and YAML CDI specifications.
    try:
        devices = output(["nvidia-ctk", "cdi", "list"]).splitlines()
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        devices = []

    # Support device-plugin JSON specs even when the toolkit CLI is absent.
    for directory in (Path("/etc/cdi"), Path("/var/run/cdi")):
        for path in sorted(directory.glob("*.json")):
            try:
                spec = json.loads(path.read_text())
                devices.extend(
                    f"{spec['kind']}={device['name']}" for device in spec["devices"]
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
    for device in devices:
        device = device.strip()
        if re.fullmatch(r"[\w.-]+/[\w.-]+=GPU-[A-Fa-f0-9-]+", device):
            if device.split("=", 1)[1] == uuid:
                return device
    raise RuntimeError(
        f"No UUID-named CDI device found for GPU_INDEX={index} ({uuid}). "
        "Configure NVIDIA CDI with UUID device names; inspect nvidia-ctk cdi list."
    )


if __name__ == "__main__":
    try:
        print(discover_gpu_device(os.environ.get("GPU_INDEX", "0")))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"GPU selection failed: {exc}", file=sys.stderr)
        sys.exit(1)
