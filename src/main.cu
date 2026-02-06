#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm_kernel.cuh"

struct Options {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    std::string impl = "baseline";
    bool verify = true;
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--m") == 0 || strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
            opt.m = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--n") == 0 || strcmp(argv[i], "-n") == 0) && i + 1 < argc) {
            opt.n = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--k") == 0 || strcmp(argv[i], "-k") == 0) && i + 1 < argc) {
            opt.k = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./dgemm [--m int] [--n int] [--k int] [--impl baseline|naive|tiled|cublas] [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }
    return opt;
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

double gflops(int m, int n, int k, double millis) {
    double flops = 2.0 * m * n * k;
    return flops / (millis * 1e6);
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    const int m = opt.m, n = opt.n, k = opt.k;
    const size_t bytes_a = static_cast<size_t>(m) * k * sizeof(float);
    const size_t bytes_b = static_cast<size_t>(k) * n * sizeof(float);
    const size_t bytes_c = static_cast<size_t>(m) * n * sizeof(float);

    std::vector<float> h_a(m * k), h_b(k * n), h_c(m * n, 0.0f), h_ref(m * n, 0.0f);

    /* TODO(student): initialize h_a, h_b with reproducible random data (e.g., std::sin / std::cos). */
    // Fill the A and B matrix with random data, using std::sin and std::cos
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            h_a[i * k + j] = std::sin(i * j);
        }
    }
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            h_b[i * n + j] = std::cos(i * j);
        }
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_ref = nullptr;
    /* TODO(student): allocate device buffers and copy host data over. */
    // since check_cuda is provided, we check at every step whether our allocation is successful
    check_cuda(cudaMalloc(&d_a, bytes_a), "cudaMalloc Device d_a");
    check_cuda(cudaMalloc(&d_b, bytes_b), "cudaMalloc Device d_b");
    check_cuda(cudaMalloc(&d_c, bytes_c), "cudaMalloc Device d_c");
    check_cuda(cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice), "cudaMemcpy Host to Device d_a");
    check_cuda(cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice), "cudaMemcpy Host to Device d_b");
    check_cuda(cudaMemcpy(d_c, h_c.data(), bytes_c, cudaMemcpyHostToDevice), "cudaMemcpy Host to Device d_c");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start");
    check_cuda(cudaEventCreate(&stop), "create stop");

    float elapsed_ms = 0.0f;
    if (opt.impl == "baseline" || opt.impl == "naive" || opt.impl == "tiled") {
        /* TODO(student): choose the right launch helper based on opt.impl and record elapsed_ms. */
        cudaStream_t stream = 0;
        // record the start event
        check_cuda(cudaEventRecord(start), "record start");

        if (opt.impl == "baseline" || opt.impl == "naive") {
            // baseline and naive are assumed to be the same
            launch_naive_gemm(d_a, d_b, d_c, m, n, k, stream);
        } else if (opt.impl == "tiled") {
            launch_tiled_gemm(d_a, d_b, d_c, m, n, k, stream);
        } else {
            // same handling as below
            throw std::invalid_argument("Unknown implementation: " + opt.impl);
        }

        // kernel was launched successfully
        check_cuda(cudaGetLastError(), "kernel launch");
        // record the stop event
        check_cuda(cudaEventRecord(stop), "record stop");
        // wait for the stop event to be signaled
        check_cuda(cudaEventSynchronize(stop), "sync stop");
        // get the elapsed time
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    } else if (opt.impl == "cublas") {
        cublasHandle_t handle;
        check_cublas(cublasCreate(&handle), "cublasCreate");
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // warm up
        check_cublas(
            cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        d_b,
                        n,
                        d_a,
                        k,
                        &beta,
                        d_c,
                        n),
            "cublas warmup");
        check_cuda(cudaDeviceSynchronize(), "warmup");

        check_cuda(cudaEventRecord(start), "record start");
        check_cublas(
            cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        d_b,
                        n,
                        d_a,
                        k,
                        &beta,
                        d_c,
                        n),
            "cublasSgemm");
        check_cuda(cudaEventRecord(stop), "record stop");
        check_cuda(cudaEventSynchronize(stop), "sync stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
        check_cublas(cublasDestroy(handle), "cublasDestroy");
    } else {
        throw std::invalid_argument("Unknown implementation: " + opt.impl);
    }

    /* TODO(student): copy d_c back into h_c. */
    check_cuda(cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost), "cudaMemcpy Device to Host h_c");
    if (opt.verify) {
        /* TODO(student): run cuBLAS reference into h_ref (or reuse above) and compute max error. */
        
        // we dont want to verify if cuBLAS was used in the first place. 
        if (opt.impl != "cublas") {
            check_cuda(cudaMalloc(&d_ref, bytes_c), "cudaMalloc Device d_ref");

            // make d_ref start with zeros
            check_cuda(cudaMemset(d_ref, 0, bytes_c), "cudaMemset Device d_ref");

            // copied from above:
            cublasHandle_t handle;
            check_cublas(cublasCreate(&handle), "cublasCreate");
            const float alpha = 1.0f;
            const float beta = 0.0f;
            check_cublas(
                cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &alpha,
                    d_b,
                    n,
                    d_a,
                    k,
                    &beta,
                    d_ref,
                    n),
            "cublasSgemm reference");
            check_cublas(cublasDestroy(handle), "cublasDestroy");
            
            // copy the reference to the host
            check_cuda(cudaMemcpy(h_ref.data(), d_ref, bytes_c, cudaMemcpyDeviceToHost), "cudaMemcpy Device to Host h_ref");

            // compute the max absolute error in the host
            float max_abs_err = 0.0f;
            for (int idx = 0; idx < m * n; ++idx) {
                // absolute value of the difference between the two values
                float e = std::fabs(h_c[idx] - h_ref[idx]);
                // max error
                if (e > max_abs_err) max_abs_err = e;
            }
            std::cout << "Max abs error: " << max_abs_err << "\n";
        } else {
            std::cout << "Skipping verification for cuBLAS implementation\n";
        }
    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Impl=" << opt.impl << " M=" << m << " N=" << n << " K=" << k
                  << " Time(ms)=" << elapsed_ms << " GFLOP/s=" << gflops(m, n, k, elapsed_ms)
                  << std::endl;
    }

    /* TODO(student): free device memory and destroy CUDA events. */
    check_cuda(cudaFree(d_a), "cudaFree Device d_a");
    check_cuda(cudaFree(d_b), "cudaFree Device d_b");
    check_cuda(cudaFree(d_c), "cudaFree Device d_c");
    if (d_ref) check_cuda(cudaFree(d_ref), "cudaFree Device d_ref");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start event");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop event");
    return 0;
}

