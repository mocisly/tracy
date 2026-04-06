// Tracy CUDA Graph GPU Zone Repro
//
// Demonstrates that Tracy correctly shows GPU zones for kernels launched
// via CUDA Graphs (cuGraphLaunch). Uses TracyCUDA to create a GPU context
// and verifies that GPU zones appear with proper CPU-to-GPU correlation.
//
// Build:
//   make          # release build
//   make debug    # debug build (asserts enabled)
//
// Run (start tracy-capture first, then run repro):
//   tracy-capture -o out.tracy &
//   ./repro

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "tracy/Tracy.hpp"
#include "tracy/TracyCUDA.hpp"

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

#define CHECK_CUDA(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

int main() {
    ZoneScoped;

    auto ctx = TracyCUDAContext();
    TracyCUDAStartProfiling(ctx);

    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // --- Create a CUDA Graph via stream capture ---
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaMemcpyAsync(d_c, d_c, bytes, cudaMemcpyDeviceToDevice, stream));
    vector_add<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_c, d_c, N);

    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    printf("CUDA Graph created with 3 nodes (kernel + memcpy + kernel)\n");
    printf("Launching graph 10 times...\n");

    // Each launch should produce 3 GPU zones (2 kernels + 1 memcpy), all
    // correlated back to the cuGraphLaunch CPU call site.
    for (int i = 0; i < 10; i++) {
        ZoneScopedN("cuGraphLaunch iteration");
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("Done. Expected 30 GPU zones in Tracy (10 launches x 3 ops).\n");

    float* h_c = (float*)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    printf("Result check: c[0] = %.1f (expected 4.0)\n", h_c[0]);

    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    TracyCUDAStopProfiling(ctx);
    TracyCUDAContextDestroy(ctx);

    return 0;
}
