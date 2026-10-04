#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::cerr << #call << ": " << cudaGetErrorString(status) << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define NCCL_CHECK(call) do { \
    const ncclResult_t status = (call); \
    if (status != ncclSuccess) { \
        std::cerr << #call << ": " << ncclGetErrorString(status) << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int ranks = 2;
using Payloads = std::array<std::vector<float>, ranks>;

void print_values(const std::vector<float>& values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << values[i];
    }
    std::cout << ']';
}

void allgather_case(const char* name, const Payloads& payloads,
                    ncclComm_t* comms, cudaStream_t* streams) {
    // Original lengths are known by this single-process teaching application.
    // A multi-process application would exchange lengths before padding.
    const size_t count = std::max(payloads[0].size(), payloads[1].size());
    const size_t send_bytes = count * sizeof(float);
    const size_t recv_bytes = ranks * send_bytes;
    float* send[ranks]{};
    float* recv[ranks]{};
    Payloads padded = payloads;
    std::vector<float> expected, expected_valid;

    std::cout << "\nCASE " << name << ": sendcount=" << count
              << ", send=" << send_bytes << " bytes/rank, receive="
              << recv_bytes << " bytes/rank\n";
    for (int rank = 0; rank < ranks; ++rank) {
        padded[rank].resize(count, 0.0f);
        expected.insert(expected.end(), padded[rank].begin(), padded[rank].end());
        expected_valid.insert(expected_valid.end(), payloads[rank].begin(),
                              payloads[rank].end());
        std::cout << "  rank " << rank << " valid=" << payloads[rank].size()
                  << " send=";
        print_values(padded[rank]);
        std::cout << '\n';
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&send[rank]), send_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&recv[rank]), recv_bytes));
        CUDA_CHECK(cudaMemcpy(send[rank], padded[rank].data(), send_bytes,
                              cudaMemcpyHostToDevice));
    }

    // One host thread submits both ranks. Grouping lets both participate
    // before NCCL needs them to make progress. All ranks use the SAME count.
    NCCL_CHECK(ncclGroupStart());
    for (int rank = 0; rank < ranks; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        NCCL_CHECK(ncclAllGather(send[rank], recv[rank], count, ncclFloat,
                                 comms[rank], streams[rank]));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int rank = 0; rank < ranks; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        // Submission is asynchronous: wait before reading or freeing buffers.
        CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
        std::vector<float> output(ranks * count), valid;
        CUDA_CHECK(cudaMemcpy(output.data(), recv[rank], recv_bytes,
                              cudaMemcpyDeviceToHost));
        for (int source = 0; source < ranks; ++source) {
            const auto start = output.begin() + source * count;
            valid.insert(valid.end(), start, start + payloads[source].size());
        }
        std::cout << "  rank " << rank << " receive=";
        print_values(output);
        std::cout << " valid=";
        print_values(valid);
        std::cout << '\n';
        if (output != expected || valid != expected_valid) {
            std::cerr << "FAIL: gathered values or unpadding are incorrect\n";
            std::exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaFree(send[rank]));
        CUDA_CHECK(cudaFree(recv[rank]));
    }
    std::cout << "PASS " << name << '\n';
}

int main(int argc, char** argv) {
    const std::string command = argc == 2 ? argv[1] : "";
    if (argc == 2 && (command == "--help" || command == "-h")) {
        std::cout << "Usage: demo check\nRuns three AllGather cases on two GPUs.\n";
        return 0;
    }
    if (argc != 2 || command != "check") {
        std::cerr << "Usage: demo check\n";
        return 2;
    }
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices < ranks) {
        std::cerr << "Two visible NVIDIA GPUs are required; found " << devices << '\n';
        return 1;
    }
    ncclComm_t comms[ranks]{};
    cudaStream_t streams[ranks]{};
    const int device_ids[ranks] = {0, 1};
    NCCL_CHECK(ncclCommInitAll(comms, ranks, device_ids));
    for (int rank = 0; rank < ranks; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        CUDA_CHECK(cudaStreamCreate(&streams[rank]));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, rank));
        std::cout << "rank " << rank << " -> visible GPU " << rank
                  << " (" << properties.name << ")\n";
    }

    // The same communicators support different counts on successive calls.
    allgather_case("equal", Payloads{{{1, 2}, {10, 20}}}, comms, streams);
    allgather_case("resize", Payloads{{{3, 4, 5}, {30, 40, 50}}}, comms, streams);
    allgather_case("padded", Payloads{{{6, 7}, {60, 70, 80}}}, comms, streams);

    for (int rank = 0; rank < ranks; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        NCCL_CHECK(ncclCommDestroy(comms[rank]));
        CUDA_CHECK(cudaStreamDestroy(streams[rank]));
    }
    return 0;
}
