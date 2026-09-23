# Week 4: backend capabilities and fallback

Implement the Case B demo in `week04.md`: the same application requests
`Y = X @ W + b`, while the runtime chooses an operator path, compute precision,
and device from declared capabilities and runtime probes.

The demo uses small matrices, ordinary PyTorch operations, and no model
checkpoints. All four scenarios run sequentially.

## Prerequisites

Use an existing Python environment with PyTorch compatible with your device.
The first three scenarios require a working CUDA backend; the CPU fallback
scenario can run without a GPU.

## Runtime configuration

The wrapper works from any directory, loads `week4/.env` when present, and
defaults to GPU 0. It sets visibility before importing PyTorch.
Optionally copy `env.example` to `.env` to select your Python executable and GPU.

| Setting | Default | Meaning |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` | Physical GPU index or UUID to expose; the demo uses its `cuda:0`. |
| `PYTHON` | `python3` | Python executable containing PyTorch. |

An existing `.env` overrides exported values. Direct invocation with
`python demo.py ...` uses exported variables and does not load `.env`.

## Run the four scenarios

```bash
cd DSAA6000T/week4
./run_demo.sh capabilities                 # declarations and real device probes
./run_demo.sh check                        # all four scenarios
./run_demo.sh check --scenario baseline
./run_demo.sh check --scenario decompose
./run_demo.sh check --scenario promote
./run_demo.sh check --scenario offload
```

The baseline dimensions are `M=64, K=128, N=128`. Inputs are CPU FP16 tensors.
The application prefers CUDA/FP16 and permits promotion and CPU fallback.
Every path returns a CPU FP16 tensor of shape `[M,N]`.

| Scenario | Changed condition | Expected plan |
| --- | --- | --- |
| `baseline` | None | CUDA FP16 combined linear via `torch.addmm`. |
| `decompose` | `K=130` | CUDA FP16 `torch.mm`, then bias addition. |
| `promote` | Inject a CUDA FP16 capability mask | CUDA FP32 matmul + add, then cast output to FP16. |
| `offload` | Inject a CUDA availability mask | CPU FP32 matmul + add, then cast output to FP16. |

The combined path's requirement that `K,N` be divisible by 8 is a **teaching
plugin restriction**, not a CUDA/PyTorch limitation. The CPU plugin's FP32-only
declaration is also an implementation choice. Masks remove advertised
capabilities for one request; they do not change the GPU, driver, or probe cache.
Every injected restriction is labeled in the trace.

The first three scenarios use the same GPU. A combined API operation does not
establish kernel fusion or guarantee a speedup. This demo makes decisions and
correctness visible; it does not report benchmark timings.

## What the runtime does

| Stage | Implementation | Purpose |
| --- | --- | --- |
| Declare | `Capability` entries in `backends.py` | List implemented paths, dtypes, and alignment constraints. |
| Discover | `available()` | Check device availability. |
| Probe | `probe(capability)` | Run and validate a representative operation on that device. |
| Plan | `Runtime.plan()` in `runtime.py` | Check the request against masks, shape constraints, probes, and caller policy. |
| Execute | `execute()` in `backends.py` | Transfer/convert inputs, run the selected path, return CPU FP16 output. |
| Validate | `demo.py` / `validate()` | Check numerical results and the scenario's expected execution plan. |

The planner tries combined FP16, separate FP16, then separate FP32 on CUDA,
followed by the permitted CPU path. Probes are lazy and cached per backend
instance, operator path, and dtype. They use shape `8×16×8`; request shapes
still undergo declared constraint checks. A small probe does not prove support
for every shape or guarantee future memory availability.

An operation probe that raises a runtime/unsupported-operation error marks that
candidate unusable. Numerical probe failures and errors during actual workload
execution stop the run; they are not silently converted into fallback.

The application calls the same interface for every scenario:

```python
result, plan = runtime.linear(x, w, b, policy=policy, mask=mask)
```

Example excerpt for `decompose`:

```text
REQUEST   linear: M=64 K=130 N=128, prefer=cuda/fp16
CHECK     cuda/fp16/combined: rejected (K and N must be divisible by 8)
PROBE     cuda/fp16/separate: PASS (representative shape 8x16x8)
PLAN      decompose: cuda/fp16/separate -> output cpu/fp16
OUTPUT    shape=[64, 128], dtype=fp16, device=cpu
```

## Correctness and caller policy

Inputs are deterministic, bounded, nonconstant random values. Each result is
compared elementwise against CPU FP32 `X @ W + b` computed from the **same
quantized inputs**, using `rtol=0.002, atol=0.002`. The checker prints maximum
absolute error and verifies output shape, device, and dtype. FP32 matmul
precision is set to `highest` to avoid TF32 in the promotion path.

Different operator paths can round differently, so bitwise equality is not
required. The tolerances apply to this small, bounded teaching workload.

The following commands intentionally fail, illustrating caller restrictions:

```bash
./run_demo.sh check --scenario promote --no-promote
./run_demo.sh check --scenario offload --no-offload
```

With no usable CUDA backend, planning can fall back to CPU. However, `check`
returns a nonzero status if the selected plan differs from the scenario's
expected GPU plan, even when its numerical result is correct. This prevents a
CPU run from being reported as a successful GPU demonstration. To exercise just
the CPU path, run `check --scenario offload`; this scenario never probes CUDA.

## Verification

```bash
python -m unittest -v test_runtime
./run_demo.sh check
```

Unit tests cover shape rejection, failed probes, request-local masks, fallback
permissions, cached probes, CPU output contracts, and detection of corrupted
results. They do not require a GPU. The second command validates the four real
execution paths.

Validated locally on one RTX 5090 with PyTorch `2.13.0+cu130` and NVIDIA driver
`595.71.05`: all four scenario plans and numerical checks passed.

## References

- [PyTorch installation selector](https://pytorch.org/get-started/locally/)
