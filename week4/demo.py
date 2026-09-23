#!/usr/bin/env python3
"""Week 4: declare capabilities, probe support, select and validate a plan."""

import argparse
import sys

from runtime import Mask, Policy, Runtime, UnsupportedRequest

SCENARIOS = {
    "baseline": ((64, 128, 128), Mask(), ("cuda", "fp16", "combined", "direct")),
    "decompose": ((64, 130, 128), Mask(), ("cuda", "fp16", "separate", "decompose")),
    "promote": ((64, 128, 128), Mask(disable_fp16=True),
                ("cuda", "fp32", "separate", "promote")),
    "offload": ((64, 128, 128), Mask(disable_cuda=True),
                ("cpu", "fp32", "separate", "offload")),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "capabilities"))
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-promote", action="store_true",
                        help="Reject paths that change the preferred compute dtype")
    parser.add_argument("--no-offload", action="store_true",
                        help="Reject paths that change the preferred backend")
    args = parser.parse_args(argv)

    try:
        import torch
        from backends import CpuBackend, CudaBackend, make_inputs, validate
    except ImportError as exc:
        parser.exit(1, f"PyTorch is required: {exc}\nSee week4/README.md for setup.\n")

    # FP32 promotion should use full FP32 matmul precision, not TF32.
    torch.set_float32_matmul_precision("highest")
    backends = [CudaBackend(), CpuBackend()]
    print(f"ENV       torch={torch.__version__}, CUDA build={torch.version.cuda}")
    print("CONTRACT  CPU FP16 inputs -> Y = X @ W + b -> CPU FP16 output")
    runtime = Runtime(backends)

    if args.command == "capabilities":
        for backend in backends:
            print(f"\nBACKEND   {backend.name}: {backend.available().reason}")
            for cap in backend.capabilities:
                print(f"DECLARE   {cap.path}/{cap.dtype}: K,N multiple of {cap.alignment}")
                print(f"PROBE     {backend.probe(cap).reason}")
        print("\nAlignment and CPU FP32-only declarations are teaching restrictions.")
        return 0

    policy = Policy(allow_promotion=not args.no_promote,
                    allow_offload=not args.no_offload)
    names = SCENARIOS if args.scenario == "all" else [args.scenario]
    for name in names:
        shape, mask, expected = SCENARIOS[name]
        print(f"\n=== {name} ===")
        inputs = make_inputs(*shape, seed=args.seed)
        result, plan = runtime.linear(*inputs, policy=policy, mask=mask)
        error = validate(result, *inputs)
        print(f"OUTPUT    shape={list(result.shape)}, dtype=fp16, device=cpu")
        print(f"VALIDATE  PASS; max_abs_error={error:.6g}, rtol=0.002, atol=0.002")
        actual = (plan.backend, plan.capability.dtype, plan.capability.path, plan.decision)
        if actual != expected:
            raise UnsupportedRequest(
                f"Scenario {name!r} needs {expected}, but selected {actual}. "
                "The fallback result is correct, but the intended GPU demonstration "
                "was not exercised. Check the CUDA build/probes, or run "
                "'check --scenario offload' for the CPU-only demonstration."
            )
        print(f"SCENARIO  PASS: {name}")
    print("\nAll selected scenarios passed numerical and execution-plan checks.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (UnsupportedRequest, ValueError, RuntimeError, AssertionError) as exc:
        print(f"ERROR     {exc}", file=sys.stderr)
        sys.exit(1)
