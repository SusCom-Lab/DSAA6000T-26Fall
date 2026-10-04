# Week 6: NCCL AllGather and buffer sizes

## Goal

A short tutorial supporting the distributed-execution case in `week05.md`.
Run three AllGather calls on two GPUs and inspect the send and receive buffers.
The program calls NCCL directly from C++; no model, training loop, or MPI is needed.

**Within one AllGather, every rank uses the same element count and datatype.**
The data, buffer addresses, and element count can change between calls.
Each rank receives all inputs in rank order, including its own input.

## Prerequisites

- Linux with two accessible NVIDIA GPUs and a working NVIDIA driver.
- CUDA Toolkit headers and runtime library.
- NCCL development headers and library (`nccl.h` and `libnccl.so`).
- A C++17 compiler, such as `g++`.

## Compile first

Compile `demo.cpp` before running the tutorial:

```bash
cd DSAA6000T/week6
./run_demo.sh build
```

This creates `build/demo` using `g++` and links against CUDA and NCCL.

## Run the three cases

```bash
CUDA_VISIBLE_DEVICES=0,1 ./build/demo check
```

This command selects physical GPUs **0 and 1**. The demo uses them as NCCL
ranks 0 and 1, respectively.

Run from `DSAA6000T/week6` after compilation. The executable runs all three cases
sequentially using the same two communicators. Each case prints both ranks'
inputs, outputs, and buffer sizes, and checks every received value.

| Case | Rank 0 sends | Rank 1 sends | Both ranks receive |
| --- | --- | --- | --- |
| `equal` | `[1, 2]` | `[10, 20]` | `[1, 2, 10, 20]` |
| `resize` | `[3, 4, 5]` | `[30, 40, 50]` | `[3, 4, 5, 30, 40, 50]` |
| `padded` | `[6, 7, 0]` | `[60, 70, 80]` | `[6, 7, 0, 60, 70, 80]` |

The first call sends 2 float32 elements (8 bytes) per rank and receives 4
elements (16 bytes) per rank. The next two calls send 3 elements (12 bytes) and
receive 6 elements (24 bytes) per rank. These are application buffer sizes,
not measurements of physical link traffic.

## Read the NCCL calls

All implementation code is in `demo.cpp`. One host process controls both GPUs.

| Stage | API | Purpose |
| --- | --- | --- |
| Initialize | `ncclCommInitAll` | Create one communicator per GPU in the same group. |
| Prepare | `cudaMalloc`, `cudaMemcpy` | Allocate GPU buffers and copy inputs. |
| Submit | `ncclGroupStart`, `ncclAllGather`, `ncclGroupEnd` | Submit both ranks' participation from one host thread. |
| Complete | `cudaStreamSynchronize` | Wait before reading or freeing each rank's buffers. |
| Inspect | `cudaMemcpy` | Copy results to the CPU, print, and validate. |
| Release | `cudaFree`, `ncclCommDestroy`, `cudaStreamDestroy` | Release buffers, communicators, and streams. |

The central call, once for each rank inside the group, is:

```cpp
ncclAllGather(send[rank], recv[rank], count, ncclFloat,
              comms[rank], streams[rank]);
```

`count` is in **elements**, not bytes. Allocate at least `count * sizeof(float)`
send bytes and `2 * count * sizeof(float)` receive bytes on each GPU.
Source rank `r` occupies the output slice starting at element `r * count`.
Group calls allow a single thread to submit work for both GPUs; returning from
`ncclGroupEnd` does not mean the GPU output is ready for CPU inspection.

## Unequal payloads: pad, gather, trim

In `padded`, the original lengths are `[2, 3]`. The application pads rank 0 to
the maximum length, 3, before calling AllGather. NCCL transmits the padding as
ordinary data; it does not add or remove padding itself.

After gathering, keep the first 2 elements of rank 0's slice and the first 3
of rank 1's slice. Both ranks then obtain `[6, 7, 60, 70, 80]`.
Use lengths to trim; removing all zeros would incorrectly remove valid zeros.

This single-process example already knows both lengths. With independently
generated multi-process inputs, first AllGather one length per rank, compute
the maximum, then pad and gather the payloads.

Unequal send counts within the same `ncclAllGather` violate its contract and
can cause undefined behavior. The tutorial does not execute that mismatch.
For variable-length transfers without padding, matching `ncclSend`/`ncclRecv`
pairs are an alternative; each pair must agree on count and datatype.

## Verification

```bash
./run_demo.sh build    # compile without requiring accessible GPUs
CUDA_VISIBLE_DEVICES=0,1 ./build/demo check  # execute and validate on two GPUs
```

A successful GPU run prints `PASS equal`, `PASS resize`, and `PASS padded`.
Both raw results and trimmed payloads are checked on both GPUs. CUDA/NCCL errors
or incorrect results stop the program with a nonzero exit code.
This is a buffer-layout demonstration, not a performance benchmark.

## References

- [NCCL AllGather API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather)
- [NCCL single-process, multiple-device examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
- [NCCL point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html)
