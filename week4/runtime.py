"""Capability declarations and planning policy; no PyTorch/CUDA calls here."""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Capability:
    path: str
    dtype: str
    alignment: int = 1

    def rejection(self, shape):
        _, k, n = shape
        if k % self.alignment or n % self.alignment:
            return f"K and N must be divisible by {self.alignment}"
        return None


@dataclass(frozen=True)
class Probe:
    ok: bool
    reason: str


@dataclass(frozen=True)
class Policy:
    preferred_backend: str = "cuda"
    preferred_dtype: str = "fp16"
    allow_promotion: bool = True
    allow_offload: bool = True


@dataclass(frozen=True)
class Mask:
    disable_fp16: bool = False
    disable_cuda: bool = False


@dataclass(frozen=True)
class Plan:
    backend: str
    capability: Capability
    decision: str


class UnsupportedRequest(RuntimeError):
    pass


class Runtime:
    def __init__(self, backends, emit: Callable[[str], None] = print):
        self.backends = {backend.name: backend for backend in backends}
        self.emit = emit

    def plan(self, shape, policy=Policy(), mask=Mask()):
        if len(shape) != 3 or any(not isinstance(d, int) or d <= 0 for d in shape):
            raise ValueError("M, K and N must be positive integers")
        if policy.preferred_backend not in self.backends:
            raise ValueError("Preferred backend is not registered")
        if policy.preferred_dtype not in ("fp16", "fp32"):
            raise ValueError("Preferred dtype must be fp16 or fp32")

        m, k, n = shape
        self.emit(f"REQUEST   linear: M={m} K={k} N={n}, "
                  f"prefer={policy.preferred_backend}/{policy.preferred_dtype}")
        self.emit(f"POLICY    promotion={policy.allow_promotion}, "
                  f"offload={policy.allow_offload}, output=cpu/fp16")
        if mask.disable_cuda or mask.disable_fp16:
            self.emit(f"MASK      INJECTED: CUDA disabled={mask.disable_cuda}, "
                      f"CUDA FP16 disabled={mask.disable_fp16}")

        names = [policy.preferred_backend]
        if policy.allow_offload and "cpu" in self.backends and "cpu" not in names:
            names.append("cpu")
        dtypes = [policy.preferred_dtype]
        if policy.allow_promotion and policy.preferred_dtype == "fp16":
            dtypes.append("fp32")

        for name in names:
            backend = self.backends[name]
            if name == "cuda" and mask.disable_cuda:
                self.emit("DISCOVER  cuda rejected: injected unavailable mask")
                continue
            status = backend.available()
            self.emit(f"DISCOVER  {name}: {status.reason}")
            if not status.ok:
                continue
            for dtype in dtypes:
                for cap in backend.capabilities:
                    if cap.dtype != dtype:
                        continue
                    label = f"{name}/{dtype}/{cap.path}"
                    if name == "cuda" and dtype == "fp16" and mask.disable_fp16:
                        self.emit(f"CHECK     {label}: rejected by injected FP16 mask")
                        continue
                    reason = cap.rejection(shape)
                    if reason:
                        self.emit(f"CHECK     {label}: rejected ({reason})")
                        continue
                    probe = backend.probe(cap)
                    self.emit(f"PROBE     {label}: {probe.reason}")
                    if not probe.ok:
                        continue
                    decision = (
                        "offload" if name != policy.preferred_backend else
                        "promote" if dtype != policy.preferred_dtype else
                        "decompose" if cap.path == "separate" else "direct"
                    )
                    self.emit(f"PLAN      {decision}: {label} -> output cpu/fp16")
                    return Plan(name, cap, decision)

        raise UnsupportedRequest(
            "No eligible path under the declared capabilities, probes and policy."
        )

    def linear(self, x, w, b, policy=Policy(), mask=Mask()):
        if x.ndim != 2 or w.ndim != 2 or b.ndim != 1:
            raise ValueError("Expected X[M,K], W[K,N], b[N]")
        if x.shape[1] != w.shape[0] or w.shape[1] != b.shape[0]:
            raise ValueError("Incompatible matrix or bias dimensions")
        shape = (x.shape[0], x.shape[1], w.shape[1])
        plan = self.plan(shape, policy, mask)
        # Execution errors propagate: an unexpected failure is not a capability mask.
        result = self.backends[plan.backend].execute(plan.capability, x, w, b)
        return result, plan
