#pragma once

#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 8;

inline dim3 make_grid(int m, int n) {
    return dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (m + BLOCK_SIZE - 1) / BLOCK_SIZE,
                1);
}

__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int N,
                                  int K) {
    // Each thread computes one C(row, col) element
    const int col = static_cast<int>(blockIdx.x) * BLOCK_SIZE + static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.y) * BLOCK_SIZE + static_cast<int>(threadIdx.y);

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int p = 0; p < K; ++p) {
        acc += A[row * K + p] * B[p * N + col];
    }
    C[row * N + col] = acc;
}

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int N,
                                  int K) {
    // allocate shared memory for the tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    const int col = static_cast<int>(blockIdx.x) * BLOCK_SIZE + static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.y) * BLOCK_SIZE + static_cast<int>(threadIdx.y);

    float acc = 0.0f;

    // loop over k dimension
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        const int pA = tile * BLOCK_SIZE + static_cast<int>(threadIdx.x);  // k-index for A load
        const int pB = tile * BLOCK_SIZE + static_cast<int>(threadIdx.y);  // k-index for B load

        // load A tile element As[ty][tx] = A[row, pA] (or 0 if OOB)
        if (row < M && pA < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + pA];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load B tile element Bs[ty][tx] = B[pB, col] (or 0 if OOB)
        if (pB < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[pB * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ensure we are done with the tile
        __syncthreads();

        // compute the partial dot product for this tile
#pragma unroll // may be able to save some time by unrolling this loop
        for (int t = 0; t < BLOCK_SIZE; ++t) {
            acc += As[threadIdx.y][t] * Bs[t][threadIdx.x];
        }

        // ensure we are done with this tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

inline void launch_naive_gemm(const float* d_a,
                              const float* d_b,
                              float* d_c,
                              int M,
                              int N,
                              int K,
                              cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_grid(M, N);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
}

inline void launch_tiled_gemm(const float* d_a,
                              const float* d_b,
                              float* d_c,
                              int M,
                              int N,
                              int K,
                              cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_grid(M, N);
    gemm_tiled_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
}

