#!/usr/bin/env python3
"""Week 3 Case B: benchmark or validate runtime scheduling policies."""

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

from policies import Experiment, POLICIES
from runtime import CudaRuntime


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("benchmark", "check"))
    parser.add_argument("--policy", choices=("all", *POLICIES), default="all")
    parser.add_argument("--elements", type=positive,
                        default=os.environ.get("DEMO_ELEMENTS", "4194304"))
    parser.add_argument("--batches", type=positive,
                        default=os.environ.get("DEMO_BATCHES", "16"))
    parser.add_argument("--repeats", type=positive,
                        default=os.environ.get("DEMO_REPEATS", "5"))
    parser.add_argument("--warmup", type=positive,
                        default=os.environ.get("DEMO_WARMUP", "2"))
    args = parser.parse_args()
    args.output = Path(os.environ.get("DEMO_OUTPUT_DIR", "profile-results"))
    selected = POLICIES if args.policy == "all" else (args.policy,)

    setup_start = perf_counter()
    try:
        runtime = CudaRuntime()
        experiment = Experiment(runtime, args.elements)
        runtime.finish()
    except (ImportError, RuntimeError) as error:
        parser.exit(1, f"CUDA initialization failed: {error}\n")
    setup_ms = (perf_counter() - setup_start) * 1000
    metadata = {**runtime.info(), "elements": args.elements,
                "batches": args.batches, "repeats": args.repeats,
                "warmup_batches": args.warmup, "setup_ms": setup_ms,
                "policies": list(selected),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "timing": "CPU preparation through final D2H completion; no validation"}
    print(f"GPU: {metadata['gpu']} | initialization: {setup_ms:.1f} ms")

    # Warm up all selected paths. Touch pinned pages and execute both kernels.
    for policy in selected:
        experiment.run(policy, args.warmup)
        experiment.run(policy, args.batches, validate=True)
        print(f"PASS {policy}: all {args.batches} batches match CPU reference")
    if args.command == "check":
        return

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    # Rotate order between repetitions to reduce consistent ordering bias.
    for repeat in range(args.repeats):
        offset = repeat % len(selected)
        for policy in selected[offset:] + selected[:offset]:
            elapsed = experiment.run(policy, args.batches)
            rows.append({"policy": policy, "repeat": repeat + 1,
                         "elapsed_ms": elapsed})
    medians = {policy: median(row["elapsed_ms"] for row in rows
                              if row["policy"] == policy) for policy in selected}
    print("\nPolicy          Median total (ms)  Relative to immediate")
    for policy, elapsed in medians.items():
        speedup = (f"{medians['immediate'] / elapsed:.2f}x"
                   if "immediate" in medians and policy != "dependency" else "—")
        print(f"{policy:<16} {elapsed:>16.3f}  {speedup}")
    with (args.output / "latency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("policy", "repeat", "elapsed_ms"))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
