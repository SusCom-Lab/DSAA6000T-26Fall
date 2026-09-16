# Week 3: a small CUDA runtime demo

## Goal

Implement Case B from `week03.md`: expose runtime primitives, change scheduling
policy, and observe execution and synchronization. The workload processes batches
of float32 arrays, using elementwise GPU operations:

```text
CPU prepares x → H2D copy → y = 2x + 1 → D2H copy → CPU waits for completion
```

Unlike Weeks 1 and 2, this demo needs no model checkpoint. PyTorch supplies the GPU
operations; students can focus on the runtime calls and policies.

## Runtime responsibilities

| Layer | Responsibility | Implementation |
| --- | --- | --- |
| Application | Input preparation, array computation, CPU reference | `policies.py`, operations in `runtime.py` |
| Scheduling policy | Stream assignment, buffer reuse, placement of CPU waits | `Experiment` in `policies.py` |
| Runtime interface | Buffer allocation, copies, launches, streams, events, completion | `CudaRuntime` in `runtime.py` |
| CUDA backend | Map interface operations to PyTorch/CUDA and launch device kernels | `runtime.py` |

The interface and its single backend live together to keep the implementation
small. Scheduling contains no PyTorch or CUDA API calls. Backend handles are exposed;
this is a teaching interface, not a completed multi-vendor abstraction.

## Execution policies

All three comparison policies perform exactly the same input preparation,
transfers, and `y = 2x + 1` computation using the same two allocated buffer slots.

| Command option | Behavior |
| --- | --- |
| `--policy immediate` | Submit a batch and wait for its result before preparing the next. |
| `--policy later` | Submit a batch, prepare the next in a separate buffer, then wait for the prior result before submitting the next batch. Uses one stream. |
| `--policy two-streams` | Alternate batches between two streams and two buffer slots. Wait before reusing a slot. Each stream preserves H2D → kernel → D2H order. |

Input preparation is a NumPy addition into a pinned buffer, not a sleep. Inputs
vary across both elements and batches. Each slot owns separate host input, host
output, device input, device output, and completion event. All final copies are
waited on, including when the number of batches is odd.

### Cross-stream dependency

`--policy dependency` runs the separate extension from the slides:

```text
Stream A: [H2D x] → [y = 2x + 1] → [record ready]
                                        │
Stream B:                         [wait ready] → [z = y²] → [D2H z] → [done]
CPU:      submit both streams ─────────────────────────────────────→ wait done
```

The consumer stream waits for the producer event. Submitting this wait does not
block the CPU. The CPU waits for the final D2H completion before checking the
result or reusing buffers. This example completes each dependent batch before
starting the next; it demonstrates ordering, not an extra pipelining strategy.
It performs additional arithmetic, so its timing is not a speedup comparison
against the three independent-batch policies.

## Configuration

```bash
cd week3
cp -n env.example .env
```

Set `CUDA_VISIBLE_DEVICES` to your selected GPU index or UUID. The program uses
`cuda:0` within the visible list. The wrapper loads `.env` before importing PyTorch.
Use Bash-compatible assignments and quote paths containing spaces.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU visibility, set before Python starts |
| `PYTHON` | `python3` | Python executable to use |
| `DEMO_ELEMENTS` | `4194304` | Float32 elements per batch |
| `DEMO_BATCHES` | `16` | Batches per full run |
| `DEMO_REPEATS` | `5` | Timed repetitions per policy |
| `DEMO_WARMUP` | `2` | Warm-up batches per policy |
| `DEMO_OUTPUT_DIR` | `profile-results` | Output directory, relative to `week3/` |

At the default size, persistent buffers use approximately 64 MiB pinned host
memory, 32 MiB ordinary host memory, and 80 MiB GPU memory. Allow additional
memory for validation temporaries, CUDA context, and PyTorch caches. Buffer use is
bounded by two slots rather than the total number of batches.

## Run the demo

`check` runs all four policies and verifies every output against the CPU reference.
It prints pass/fail results without producing benchmark files.

`benchmark` checks correctness first, then measures repeated runs and reports
median execution times. It saves timings and run metadata in `profile-results/`.
Both commands run all policies by default; use `--policy` to select one.

```bash
./run_demo.sh check                  # validate every policy against the CPU
./run_demo.sh benchmark              # validate, then measure repeated runs
./run_demo.sh benchmark --policy later
./run_demo.sh check --policy dependency --elements 1048579 --batches 3
```

CLI options override the corresponding `.env` defaults. `--elements`, `--batches`,
`--repeats`, and `--warmup` must be positive. `python3 demo.py --help` lists options;
direct Python invocation uses exported environment variables and does not load
`.env`. The wrapper works regardless of your current directory.

Every selected policy warms up first and then validates **all elements of all
batches** in a separate pass. A mismatch raises an error and stops the run.
Validation recreates expected inputs independently and checks results before
output buffers are reused. Normal timed passes discard completed outputs.

## Timing and outputs

CPU wall-clock timing starts before the first input preparation and ends after
the last D2H completion wait. It includes CPU preparation, submissions, transfers,
GPU execution, and required waits. Initialization, allocation,
warm-up, reference checks, and file writing are outside this timing interval.

The terminal reports initialization time separately, validation results, median total
latency, and speedup relative to `immediate` when selected. Repetition order
rotates to reduce consistent ordering bias. Files under `profile-results/` are:

- `latency.csv`: total elapsed milliseconds for each policy and repetition.
- `metadata.json`: GPU, library/CUDA versions, workload configuration, and initialization time.

Each benchmark run replaces its output files; use another `DEMO_OUTPUT_DIR` to
retain a previous experiment.

**Overlap is not guaranteed.** This arithmetic has little computation per byte;
transfers, CPU preparation, memory bandwidth, and submission overhead can dominate.
Try different array sizes and compare latency. Pinned buffers
and independent streams permit asynchronous transfers, but hardware capabilities
and resource contention determine actual concurrency. A correct two-stream run
can be slower than the one-stream baseline.

## Tests

```bash
python3 -m unittest -v test_policies
CUDA_VISIBLE_DEVICES=0 WEEK3_TEST_CUDA=1 python3 -m unittest -v test_policies
```

The first command uses a deferred execution backend to check buffer
reuse, final draining, and event dependencies; it does not test CUDA or simulate
performance. The second also runs real GPU checks for all policies, including
irregular array sizes and odd batch counts.

## References

- [PyTorch CUDA streams](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Stream.html)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
