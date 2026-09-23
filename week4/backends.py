"""CPU/CUDA teaching plugins implemented with ordinary PyTorch operations."""

import torch

from runtime import Capability, Probe

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
RTOL = 2e-3
ATOL = 2e-3


def make_inputs(m, k, n, seed=42):
    """Bounded, nonconstant CPU FP16 inputs; do not touch the global RNG."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(
        (torch.rand(shape, generator=generator) * 0.5 - 0.25).half()
        for shape in ((m, k), (k, n), (n,))
    )


def validate(result, x, w, b):
    """Independent FP32 CPU reference using the same quantized inputs."""
    reference = x.float() @ w.float() + b.float()
    if result.device.type != "cpu" or result.dtype != torch.float16:
        raise AssertionError("Output contract requires a CPU FP16 tensor")
    torch.testing.assert_close(result.float(), reference, rtol=RTOL, atol=ATOL)
    return (result.float() - reference).abs().max().item()


class TorchBackend:
    name = ""
    capabilities = ()

    def __init__(self):
        self._availability = None
        self._probes = {}

    def available(self):
        if self._availability is None:
            if self.name == "cpu":
                self._availability = Probe(True, "available")
            elif not torch.cuda.is_available():
                self._availability = Probe(False, "CUDA unavailable")
            else:
                try:
                    name = torch.cuda.get_device_name(0)
                    self._availability = Probe(True, f"available: {name} (cuda:0)")
                except RuntimeError as exc:
                    self._availability = Probe(False, str(exc).splitlines()[0])
        return self._availability

    def probe(self, capability):
        if capability not in self._probes:
            if not self.available().ok:
                return self.available()
            # One aligned representative shape, cached per backend/path/dtype.
            x, w, b = make_inputs(8, 16, 8, seed=7)
            try:
                result = self.execute(capability, x, w, b)
                validate(result, x, w, b)
                status = Probe(True, "PASS (representative shape 8x16x8)")
            except (RuntimeError, NotImplementedError) as exc:
                status = Probe(False, f"unusable: {str(exc).splitlines()[0]}")
            # Numerical mismatches are bugs and intentionally propagate.
            self._probes[capability] = status
        return self._probes[capability]

    @torch.inference_mode()
    def execute(self, capability, x, w, b):
        if capability not in self.capabilities:
            raise ValueError("Path is not declared by this backend")
        if x.ndim != 2 or w.ndim != 2 or b.ndim != 1:
            raise ValueError("Expected X[M,K], W[K,N], b[N]")
        if x.shape[1] != w.shape[0] or w.shape[1] != b.shape[0]:
            raise ValueError("Incompatible matrix or bias dimensions")
        shape = (x.shape[0], x.shape[1], w.shape[1])
        reason = capability.rejection(shape)
        if any(d <= 0 for d in shape) or reason:
            raise ValueError(reason or "Dimensions must be positive")
        if any(t.device.type != "cpu" or t.dtype != torch.float16 for t in (x, w, b)):
            raise ValueError("Inputs must be CPU FP16 tensors")
        device = "cuda:0" if self.name == "cuda" else "cpu"
        dtype = DTYPES[capability.dtype]
        a, weights, bias = (t.to(device=device, dtype=dtype) for t in (x, w, b))
        if capability.path == "combined":
            output = torch.addmm(bias, a, weights)
        elif capability.path == "separate":
            output = torch.mm(a, weights)
            output.add_(bias)
        else:
            raise ValueError("Unknown operator path")
        # Blocking D2H transfer also waits for this result's GPU work.
        return output.to(device="cpu", dtype=torch.float16)


class CudaBackend(TorchBackend):
    name = "cuda"
    capabilities = (
        Capability("combined", "fp16", alignment=8),
        Capability("separate", "fp16"),
        Capability("separate", "fp32"),
    )


class CpuBackend(TorchBackend):
    name = "cpu"
    # This is the teaching plugin's scope, not a claim that CPUs lack FP16.
    capabilities = (Capability("separate", "fp32"),)
