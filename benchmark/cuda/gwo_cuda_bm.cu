// gwo_cuda_bm.cu
// CUDA GWO benchmark with Rastrigin fitness (double precision)
// Build: nvcc -O3 -std=c++17 gwo_cuda_bm.cu -o gwo_cuda_bm
// Run:   ./gwo_cuda_bm

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------- CUDA error check --------------------
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(_e) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

// -------------------- RNG kernels --------------------
__global__ void init_rng(curandStatePhilox4_32_10_t* states, unsigned long long seed, int pop) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pop) {
        // sequence = i for per-wolf independent streams
        curand_init(seed, (unsigned long long)i, 0ULL, &states[i]);
    }
}

// Initialize positions uniformly in [minB, maxB]
__global__ void init_positions(double* X, int pop, int N, double minB, double maxB,
                               curandStatePhilox4_32_10_t* states) {
    int wolf = blockIdx.x;
    int tid  = threadIdx.x;

    if (wolf >= pop) return;

    curandStatePhilox4_32_10_t local = states[wolf];

    for (int d = tid; d < N; d += blockDim.x) {
        double u = curand_uniform_double(&local); // (0,1]
        double x = minB + (maxB - minB) * u;
        X[wolf * N + d] = x;
    }

    states[wolf] = local;
}

// -------------------- Fitness: Rastrigin --------------------
__global__ void rastrigin_fitness(const double* X, double* fitness, int pop, int N) {
    int wolf = blockIdx.x;
    int tid  = threadIdx.x;
    if (wolf >= pop) return;

    // partial sum over dimensions
    double sum = 0.0;
    const double A = 10.0;
    const double two_pi = 2.0 * M_PI;

    for (int d = tid; d < N; d += blockDim.x) {
        double x = X[wolf * N + d];
        sum += x * x - A * cos(two_pi * x);
    }

    // reduce within block
    __shared__ double sh[256]; // blockDim.x must be <= 256 in this file
    sh[tid] = sum;
    __syncthreads();

    // reduction (power-of-two friendly; assume blockDim=256)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        fitness[wolf] = A * (double)N + sh[0];
    }
}

// -------------------- Update positions (GWO) --------------------
// We keep alpha/beta/delta vectors in device arrays of size N each.
__global__ void gwo_update(double* X, int pop, int N,
                           const double* alpha, const double* beta, const double* delta,
                           double a, double minB, double maxB,
                           curandStatePhilox4_32_10_t* states) {
    int wolf = blockIdx.x;
    int tid  = threadIdx.x;
    if (wolf >= pop) return;

    curandStatePhilox4_32_10_t local = states[wolf];

    for (int d = tid; d < N; d += blockDim.x) {
        double x = X[wolf * N + d];

        // For alpha
        double r1 = curand_uniform_double(&local);
        double r2 = curand_uniform_double(&local);
        double A1 = 2.0 * a * r1 - a;
        double C1 = 2.0 * r2;
        double D_alpha = fabs(C1 * alpha[d] - x);
        double X1 = alpha[d] - A1 * D_alpha;

        // For beta
        r1 = curand_uniform_double(&local);
        r2 = curand_uniform_double(&local);
        double A2 = 2.0 * a * r1 - a;
        double C2 = 2.0 * r2;
        double D_beta = fabs(C2 * beta[d] - x);
        double X2 = beta[d] - A2 * D_beta;

        // For delta
        r1 = curand_uniform_double(&local);
        r2 = curand_uniform_double(&local);
        double A3 = 2.0 * a * r1 - a;
        double C3 = 2.0 * r2;
        double D_delta = fabs(C3 * delta[d] - x);
        double X3 = delta[d] - A3 * D_delta;

        double x_new = (X1 + X2 + X3) / 3.0;

        // clamp to bounds
        if (x_new < minB) x_new = minB;
        if (x_new > maxB) x_new = maxB;

        X[wolf * N + d] = x_new;
    }

    states[wolf] = local;
}

// -------------------- Host helpers --------------------
static void pick_top3_indices(const std::vector<double>& fit, int& ia, int& ib, int& ic) {
    // Find 3 smallest fitness values; pop is small => simple scan
    ia = ib = ic = -1;
    double fa = std::numeric_limits<double>::infinity();
    double fb = std::numeric_limits<double>::infinity();
    double fc = std::numeric_limits<double>::infinity();

    for (int i = 0; i < (int)fit.size(); ++i) {
        double f = fit[i];
        if (f < fa) {
            fc = fb; ic = ib;
            fb = fa; ib = ia;
            fa = f;  ia = i;
        } else if (f < fb) {
            fc = fb; ic = ib;
            fb = f;  ib = i;
        } else if (f < fc) {
            fc = f;  ic = i;
        }
    }
}

// copy wolf vector X[idx] -> vec (device to device)
static void copy_wolf_vector_dev(double* d_vec, const double* d_X, int idx, int N) {
    CUDA_CHECK(cudaMemcpy(d_vec, d_X + (size_t)idx * (size_t)N,
                          sizeof(double) * (size_t)N,
                          cudaMemcpyDeviceToDevice));
}

int main() {
    // -------------------- Benchmark parameters --------------------
    std::vector<int> N_list   = {10, 50, 100, 500, 1000};
    std::vector<int> Pop_list = {50, 100, 200, 500, 1000};

    const int RUNS      = 5;
    const int MAX_ITERS = 1000;

    // Rastrigin common domain
    const double minB = -5.12;
    const double maxB =  5.12;

    // Seed for reproducibility (per-run deterministic)
    const unsigned long long GLOBAL_SEED = 123456789ULL;

    std::string problem_name = "Rastrigin";
    std::string filename = "gwo_cuda_rastrigin.csv";

    // GPU setup
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // CSV header
    bool need_header = (!std::filesystem::exists(filename) ||
                        std::filesystem::file_size(filename) == 0);
    {
        std::ofstream csv(filename, std::ios::app);
        if (!csv) {
            std::cerr << "Cannot open " << filename << " for writing\n";
            return 1;
        }
        if (need_header) {
            csv << "problem,N,POP_SIZE,max_iters,avg_ms,best_fitness\n";
        }
    }

    // We fix block size = 256 for both update + fitness reduction
    // (works well for N up to 1024 via striding)
    const int BLOCK = 256;

    for (int N : N_list) {
        for (int POP : Pop_list) {
            // Allocate device buffers
            double* d_X = nullptr;
            double* d_fit = nullptr;
            double* d_alpha = nullptr;
            double* d_beta  = nullptr;
            double* d_delta = nullptr;
            curandStatePhilox4_32_10_t* d_states = nullptr;

            CUDA_CHECK(cudaMalloc(&d_X,     sizeof(double) * (size_t)POP * (size_t)N));
            CUDA_CHECK(cudaMalloc(&d_fit,   sizeof(double) * (size_t)POP));
            CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(double) * (size_t)N));
            CUDA_CHECK(cudaMalloc(&d_beta,  sizeof(double) * (size_t)N));
            CUDA_CHECK(cudaMalloc(&d_delta, sizeof(double) * (size_t)N));
            CUDA_CHECK(cudaMalloc(&d_states, sizeof(curandStatePhilox4_32_10_t) * (size_t)POP));

            // init rng + positions
            {
                int gridR = (POP + 255) / 256;
                init_rng<<<gridR, 256>>>(d_states, GLOBAL_SEED, POP);
                CUDA_CHECK(cudaGetLastError());

                init_positions<<<POP, BLOCK>>>(d_X, POP, N, minB, maxB, d_states);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            std::vector<double> h_fit(POP);

            long long total_ms = 0;
            double best_fitness_last_run = 0.0;

            std::cout << "================================\n";
            std::cout << "CUDA GWO | Rastrigin"
                      << " | N=" << N
                      << " | POP_SIZE=" << POP
                      << " | MAX_ITERS=" << MAX_ITERS << "\n";

            for (int r = 1; r <= RUNS; ++r) {
                // Re-init RNG and positions per run for fair, reproducible runs
                {
                    int gridR = (POP + 255) / 256;
                    init_rng<<<gridR, 256>>>(d_states, GLOBAL_SEED, POP);
                    CUDA_CHECK(cudaGetLastError());
                    init_positions<<<POP, BLOCK>>>(d_X, POP, N, minB, maxB, d_states);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                auto t0 = std::chrono::steady_clock::now();

                // initial fitness
                rastrigin_fitness<<<POP, BLOCK>>>(d_X, d_fit, POP, N);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(h_fit.data(), d_fit, sizeof(double) * (size_t)POP,
                                      cudaMemcpyDeviceToHost));

                int ia, ib, ic;
                pick_top3_indices(h_fit, ia, ib, ic);
                copy_wolf_vector_dev(d_alpha, d_X, ia, N);
                copy_wolf_vector_dev(d_beta,  d_X, ib, N);
                copy_wolf_vector_dev(d_delta, d_X, ic, N);

                // main loop
                for (int it = 1; it <= MAX_ITERS; ++it) {
                    // a decreases linearly from 2 -> 0
                    double a = 2.0 - 2.0 * (double)it / (double)MAX_ITERS;

                    // update all wolves
                    gwo_update<<<POP, BLOCK>>>(d_X, POP, N, d_alpha, d_beta, d_delta,
                                              a, minB, maxB, d_states);
                    CUDA_CHECK(cudaGetLastError());

                    // evaluate
                    rastrigin_fitness<<<POP, BLOCK>>>(d_X, d_fit, POP, N);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());

                    // select alpha/beta/delta on CPU (POP small)
                    CUDA_CHECK(cudaMemcpy(h_fit.data(), d_fit, sizeof(double) * (size_t)POP,
                                          cudaMemcpyDeviceToHost));
                    pick_top3_indices(h_fit, ia, ib, ic);
                    copy_wolf_vector_dev(d_alpha, d_X, ia, N);
                    copy_wolf_vector_dev(d_beta,  d_X, ib, N);
                    copy_wolf_vector_dev(d_delta, d_X, ic, N);
                }

                // best fitness at the end of run (alpha on last selection)
                best_fitness_last_run = h_fit[0];
                for (double f : h_fit) best_fitness_last_run = std::min(best_fitness_last_run, f);

                auto t1 = std::chrono::steady_clock::now();
                long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                total_ms += ms;

                std::cout << "Run " << r
                          << " | " << ms << " ms"
                          << " | best fitness = " << best_fitness_last_run
                          << "\n";
            }

            double avg_ms = total_ms / double(RUNS);

            // append CSV row
            {
                std::ofstream csv(filename, std::ios::app);
                csv << problem_name
                    << "," << N
                    << "," << POP
                    << "," << MAX_ITERS
                    << "," << avg_ms
                    << "," << best_fitness_last_run
                    << "\n";
            }

            std::cout << ">>> Avg time: " << avg_ms << " ms\n";

            // cleanup
            CUDA_CHECK(cudaFree(d_X));
            CUDA_CHECK(cudaFree(d_fit));
            CUDA_CHECK(cudaFree(d_alpha));
            CUDA_CHECK(cudaFree(d_beta));
            CUDA_CHECK(cudaFree(d_delta));
            CUDA_CHECK(cudaFree(d_states));

            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    return 0;
}
