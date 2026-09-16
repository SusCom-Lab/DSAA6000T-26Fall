"""CUDA mechanisms backed by PyTorch; scheduling lives in policies.py."""

import numpy as np


class CudaRuntime:
    """Teaching interface exposing buffers, streams, events, and GPU operations.

    Handles are PyTorch objects, not a portable public ABI. Buffers remain alive
    until the policy waits for completion; no allocation occurs during launch.
    """

    def __init__(self, device=0):
        import torch

        self.torch = torch
        self.device = torch.device("cuda", device)
        torch.cuda.set_device(self.device)
        self.one = torch.ones((), dtype=torch.float32, device=self.device)
        self.finish()

    def host_buffer(self, elements):
        # The NumPy view retains its backing pinned PyTorch tensor.
        return self.torch.empty(elements, dtype=self.torch.float32,
                                pin_memory=True).numpy()

    def device_buffer(self, elements):
        return self.torch.empty(elements, dtype=self.torch.float32,
                                device=self.device)

    def stream(self):
        return self.torch.cuda.Stream(device=self.device)

    def event(self):
        return self.torch.cuda.Event(enable_timing=False)

    def copy_to_device(self, host, device, stream):
        with self.torch.cuda.stream(stream):
            device.copy_(self.torch.from_numpy(host), non_blocking=True)

    def copy_to_host(self, device, host, stream):
        with self.torch.cuda.stream(stream):
            self.torch.from_numpy(host).copy_(device, non_blocking=True)

    def launch(self, name, source, target, stream):
        with self.torch.cuda.stream(stream):
            if name == "affine":
                # torch.add(input, other, alpha) = input + alpha * other.
                self.torch.add(self.one, source, alpha=2, out=target)
            elif name == "square":
                self.torch.square(source, out=target)
            else:
                raise ValueError(f"Unknown operation: {name}")

    def record_event(self, event, stream):
        event.record(stream)

    def wait_event(self, event, stream):
        stream.wait_event(event)  # Orders GPU work without blocking the CPU.

    def synchronize(self, event):
        event.synchronize()

    def finish(self):
        self.torch.cuda.synchronize(self.device)

    def info(self):
        return {
            "device": self.device.index,
            "gpu": self.torch.cuda.get_device_name(self.device),
            "torch": self.torch.__version__,
            "numpy": np.__version__,
            "cuda_build": self.torch.version.cuda,
        }
