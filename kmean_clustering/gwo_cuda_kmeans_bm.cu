// gwo_cuda_kmeans_bm.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <chrono>
#include <filesystem>
#include <algorithm>

#include <cuda_runtime.h>

// nvcc -O2 -std=c++20 gwo_cuda_kmeans_bm.cu -o cuda_kmeans.exe
// nvcc -O2 -std=c++20 -gencode arch=compute_86,code=sm_86 -o cuda_kmeans gwo_cuda_kmeans_bm.cu

static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << msg << " | " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

// ===================== MNIST IDX LOADER (host) =====================
static inline uint32_t read_be_u32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) throw std::runtime_error("Failed to read u32");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

struct MNIST {
    int n = 0, rows = 0, cols = 0;
    std::vector<float> X;      // [n * D], normalized [0,1]
    std::vector<uint8_t> y;
    int dim() const { return rows * cols; }
};

static inline MNIST load_mnist_images_labels(const std::string& images_path,
                                             const std::string& labels_path,
                                             int limit = -1)
{
    std::ifstream fi(images_path, std::ios::binary);
    std::ifstream fl(labels_path, std::ios::binary);
    if (!fi) throw std::runtime_error("Cannot open images: " + images_path);
    if (!fl) throw std::runtime_error("Cannot open labels: " + labels_path);

    uint32_t magic_i = read_be_u32(fi);
    uint32_t n_i     = read_be_u32(fi);
    uint32_t rows    = read_be_u32(fi);
    uint32_t cols    = read_be_u32(fi);

    uint32_t magic_l = read_be_u32(fl);
    uint32_t n_l     = read_be_u32(fl);

    if (magic_i != 2051) throw std::runtime_error("Bad image magic (expected 2051)");
    if (magic_l != 2049) throw std::runtime_error("Bad label magic (expected 2049)");
    if (n_i != n_l) throw std::runtime_error("Image/label count mismatch");

    int n = (int)n_i;
    if (limit > 0 && limit < n) n = limit;

    MNIST ds;
    ds.n = n;
    ds.rows = (int)rows;
    ds.cols = (int)cols;
    const int D = ds.dim();

    ds.X.resize((size_t)n * (size_t)D);
    ds.y.resize((size_t)n);

    std::vector<uint8_t> buf((size_t)D);

    for (int i = 0; i < n; ++i) {
        fi.read(reinterpret_cast<char*>(buf.data()), D);
        if (!fi) throw std::runtime_error("Failed reading image data");
        for (int j = 0; j < D; ++j) ds.X[(size_t)i * D + j] = buf[j] / 255.0f;

        uint8_t lab;
        fl.read(reinterpret_cast<char*>(&lab), 1);
        if (!fl) throw std::runtime_error("Failed reading label data");
        ds.y[(size_t)i] = lab;
    }
    return ds;
}

// ===================== Hàm random đồng bộ với Serial =====================
__device__ __forceinline__ uint64_t splitmix64_dev(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

__device__ __forceinline__ double indexed_random_dev(int iter, int wid, int dim, int channel, uint64_t global_seed)
{
    uint64_t mix =
        (uint64_t)iter * 0x9E3779B97F4A7C15ULL ^
        (uint64_t)wid  * 0xBF58476D1CE4E5B9ULL ^
        (uint64_t)dim  * 0x94D049BB133111EBULL ^
        (uint64_t)channel * 0x123456789ABCDEFULL ^
        global_seed;

    uint64_t r = splitmix64_dev(mix);
    // (r >> 11) * 2^-53
    return (double)(r >> 11) * 0x1.0p-53;
}

__device__ __forceinline__ double clamp01(double x) {
    return fmin(1.0, fmax(0.0, x));
}

// ===================== Kernel: init population pos in [0,1] (double) =====================
__global__ void init_pos_kernel(double* pos, int POP, int NDIM, uint64_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = POP * NDIM;
    if (idx >= total) return;
    int wolf = idx / NDIM;
    int dim  = idx - wolf * NDIM;
    double r = indexed_random_dev(0, wolf, dim, 77, seed);
    pos[idx] = r; // in [0,1)
}

// ===================== Kernel: fitness SSE for all wolves =====================
// X float, pos double, sse_out double
__global__ void kmeans_sse_pop_kernel(const float* __restrict__ X,
                                      int Ndata, int D,
                                      const double* __restrict__ pos,
                                      int K, int POP,
                                      double* __restrict__ sse_out)
{
    int wolf = (int)blockIdx.y;
    if (wolf >= POP) return;

    int tid = (int)threadIdx.x;
    int i = (int)blockIdx.x * (int)blockDim.x + tid;

    double partial = 0.0;
    const double* cent = pos + (size_t)wolf * (size_t)(K * D);

    if (i < Ndata) {
        const float* xi = X + (size_t)i * (size_t)D;
        double best = 1e300;

        for (int k = 0; k < K; ++k) {
            const double* ck = cent + (size_t)k * (size_t)D;
            double dist = 0.0;
            for (int d = 0; d < D; ++d) {
                double diff = (double)xi[d] - (double)ck[d];
                dist += diff * diff;
            }
            if (dist < best) best = dist;
        }
        partial = best;
    }

    extern __shared__ double smem[];
    smem[tid] = partial;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(&sse_out[wolf], smem[0]);
}

// ===================== GPU Top-3 selection matching heap semantics =====================
struct PairFI { double f; int i; };

__device__ __forceinline__ bool better_smallest(PairFI a, PairFI b) {
    if (a.f != b.f) return a.f < b.f;
    return a.i < b.i;
}

__device__ __forceinline__ bool better_largest(PairFI a, PairFI b) {
    if (a.f != b.f) return a.f > b.f;
    return a.i > b.i;
}

struct Top3Small {
    PairFI v0, v1, v2;
};

__device__ __forceinline__ void top3small_init(Top3Small& t) {
    t.v0 = {1e300, -1};
    t.v1 = {1e300, -1};
    t.v2 = {1e300, -1};
}

__device__ __forceinline__ void top3small_push(Top3Small& t, PairFI x) {
    if (better_smallest(x, t.v0)) {
        t.v2 = t.v1;
        t.v1 = t.v0;
        t.v0 = x;
    } else if (better_smallest(x, t.v1)) {
        t.v2 = t.v1;
        t.v1 = x;
    } else if (better_smallest(x, t.v2)) {
        t.v2 = x;
    }
}

__device__ __forceinline__ void top3small_merge(Top3Small& a, const Top3Small& b) {
    top3small_push(a, b.v0);
    top3small_push(a, b.v1);
    top3small_push(a, b.v2);
}

__global__ void reduce_top3_stage1(const double* fitness, int N,
                                   PairFI* cand /*size = nb*3*/) {
    extern __shared__ Top3Small sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    Top3Small t;
    top3small_init(t);

    for (int idx = gid; idx < N; idx += gridDim.x * blockDim.x) {
        PairFI x{fitness[idx], idx};
        top3small_push(t, x);
    }

    sdata[tid] = t;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            top3small_merge(sdata[tid], sdata[tid + off]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int out = blockIdx.x * 3;
        cand[out + 0] = sdata[0].v0;
        cand[out + 1] = sdata[0].v1;
        cand[out + 2] = sdata[0].v2;
    }
}

__global__ void reduce_top3_stage2(const PairFI* cand, int N3,
                                   int* best3_idx_out /*size=3*/) {
    extern __shared__ Top3Small sdata[];
    int tid = threadIdx.x;

    Top3Small t;
    top3small_init(t);

    for (int idx = tid; idx < N3; idx += blockDim.x) {
        PairFI x = cand[idx];
        if (x.i >= 0) top3small_push(t, x);
    }

    sdata[tid] = t;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            top3small_merge(sdata[tid], sdata[tid + off]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        best3_idx_out[0] = sdata[0].v2.i; // largest among top3
        best3_idx_out[1] = sdata[0].v1.i;
        best3_idx_out[2] = sdata[0].v0.i; // smallest among top3
    }
}

// ===================== Kernel: gather leaders =====================
__global__ void gather_leaders_kernel(const double* pos, int NDIM,
                                      const int* best3_idx,
                                      double* L0, double* L1, double* L2) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= NDIM) return;

    int i0 = best3_idx[0]; // largest among top3
    int i1 = best3_idx[1];
    int i2 = best3_idx[2]; // smallest among top3

    L0[d] = pos[(size_t)i0 * NDIM + d];
    L1[d] = pos[(size_t)i1 * NDIM + d];
    L2[d] = pos[(size_t)i2 * NDIM + d];
}

// ===================== Kernel: update all positions =====================
__global__ void gwo_update_kernel(double* pos, int POP, int NDIM,
                                 const double* L0,
                                 const double* L1,
                                 const double* L2,
                                 double a,
                                 int iter,
                                 uint64_t seed)
{
    int wolf = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (wolf >= POP || d >= NDIM) return;

    size_t idx = (size_t)wolf * (size_t)NDIM + (size_t)d;
    double x = pos[idx];

    // j=0 (L0): channels 0,1
    double r1 = indexed_random_dev(iter, wolf, d, 0, seed);
    double r2 = indexed_random_dev(iter, wolf, d, 1, seed);
    double A1 = 2.0 * a * r1 - a;
    double C1 = 2.0 * r2;
    double D1 = fabs(C1 * L0[d] - x);
    double X1 = L0[d] - A1 * D1;

    // j=1 (L1): channels 2,3
    r1 = indexed_random_dev(iter, wolf, d, 2, seed);
    r2 = indexed_random_dev(iter, wolf, d, 3, seed);
    double A2 = 2.0 * a * r1 - a;
    double C2 = 2.0 * r2;
    double D2 = fabs(C2 * L1[d] - x);
    double X2 = L1[d] - A2 * D2;

    // j=2 (L2): channels 4,5
    r1 = indexed_random_dev(iter, wolf, d, 4, seed);
    r2 = indexed_random_dev(iter, wolf, d, 5, seed);
    double A3 = 2.0 * a * r1 - a;
    double C3 = 2.0 * r2;
    double D3 = fabs(C3 * L2[d] - x);
    double X3 = L2[d] - A3 * D3;

    double newx = (X1 + X2 + X3) / 3.0;
    pos[idx] = clamp01(newx);
}

// ===================== Main =====================
int main() {
    // Config
    std::vector<int> LIMIT_list = {1000, 5000};  
    const int K = 10;
    const int RUNS = 1;
    const int MAX_ITERS = 100;
    const uint64_t SEED = 123456789ULL;

    std::vector<int> Pop_list = {32, 128, 512};

    int dev = 0;
    cudaDeviceProp prop{};
    cuda_check(cudaGetDevice(&dev), "GetDevice");
    cuda_check(cudaGetDeviceProperties(&prop, dev), "GetDeviceProperties");
    std::cout << "CUDA Device: " << prop.name << "\n";

    std::string filename = "gwo_cuda.csv";
    bool need_header = (!std::filesystem::exists(filename) ||
                        std::filesystem::file_size(filename) == 0);

    for (int LIMIT_TRAIN : LIMIT_list) {
        // Load MNIST per LIMIT
        MNIST train = load_mnist_images_labels(
            "./mnist/train-images-idx3-ubyte",
            "./mnist/train-labels-idx1-ubyte",
            LIMIT_TRAIN
        );

        const int D = train.dim();
        const int Ndata = train.n;
        const int NDIM = K * D;

        std::cout << "\n================================\n";
        std::cout << "Loaded MNIST: limit=" << LIMIT_TRAIN
                  << " | N=" << Ndata << ", D=" << D
                  << " (" << train.rows << "x" << train.cols << ")"
                  << " | NDIM=" << NDIM << "\n";

        // copy X per LIMIT (because Ndata changes)
        float* d_X = nullptr;
        size_t bytesX = (size_t)Ndata * (size_t)D * sizeof(float);
        cuda_check(cudaMalloc((void**)&d_X, bytesX), "Malloc d_X");
        cuda_check(cudaMemcpy(d_X, train.X.data(), bytesX, cudaMemcpyHostToDevice), "Memcpy X");

        for (int POP : Pop_list) {
            std::cout << "--------------------------------\n";
            std::cout << "CUDA FULL (match OpenMP) | limit=" << LIMIT_TRAIN
                      << " | POP=" << POP
                      << " | ITERS=" << MAX_ITERS
                      << " | Ndata=" << Ndata
                      << " | K=" << K << "\n";

            // allocate per POP
            double* d_pos = nullptr;   // [POP * NDIM]
            double* d_sse = nullptr;   // [POP]
            double* d_L0 = nullptr;    // [NDIM]
            double* d_L1 = nullptr;
            double* d_L2 = nullptr;
            int*    d_best3 = nullptr; // [3]

            size_t bytesPos = (size_t)POP * (size_t)NDIM * sizeof(double);
            size_t bytesSse = (size_t)POP * sizeof(double);
            size_t bytesVec = (size_t)NDIM * sizeof(double);

            cuda_check(cudaMalloc((void**)&d_pos, bytesPos), "Malloc d_pos");
            cuda_check(cudaMalloc((void**)&d_sse, bytesSse), "Malloc d_sse");
            cuda_check(cudaMalloc((void**)&d_L0, bytesVec), "Malloc d_L0");
            cuda_check(cudaMalloc((void**)&d_L1, bytesVec), "Malloc d_L1");
            cuda_check(cudaMalloc((void**)&d_L2, bytesVec), "Malloc d_L2");
            cuda_check(cudaMalloc((void**)&d_best3, 3 * sizeof(int)), "Malloc d_best3");

            // init pos
            {
                int threads = 256;
                int total = POP * NDIM;
                int blocks = (total + threads - 1) / threads;
                init_pos_kernel<<<blocks, threads>>>(d_pos, POP, NDIM, SEED);
                cuda_check(cudaGetLastError(), "init_pos_kernel");
                cuda_check(cudaDeviceSynchronize(), "sync init");
            }

            // reduction buffers per POP
            const int reduceBlock = 256;
            int nb = (POP + reduceBlock - 1) / reduceBlock;
            nb = std::max(1, nb);
            PairFI* d_cand = nullptr;
            cuda_check(cudaMalloc((void**)&d_cand, (size_t)nb * 3 * sizeof(PairFI)), "Malloc d_cand");

            long long total_ms = 0;
            double best_last = 0.0;

            for (int r = 1; r <= RUNS; ++r) {
                auto t0 = std::chrono::steady_clock::now();

                for (int iter = 0; iter < MAX_ITERS; ++iter) {
                    // fitness
                    cuda_check(cudaMemset(d_sse, 0, bytesSse), "memset d_sse");

                    const int threads = 256;
                    int blocksX = (Ndata + threads - 1) / threads;
                    dim3 grid(blocksX, POP, 1);
                    dim3 block(threads, 1, 1);
                    size_t shmem = (size_t)threads * sizeof(double);

                    kmeans_sse_pop_kernel<<<grid, block, shmem>>>(d_X, Ndata, D, d_pos, K, POP, d_sse);
                    cuda_check(cudaGetLastError(), "kmeans_sse_pop_kernel");

                    // top3
                    reduce_top3_stage1<<<nb, reduceBlock, (size_t)reduceBlock * sizeof(Top3Small)>>>(d_sse, POP, d_cand);
                    cuda_check(cudaGetLastError(), "reduce_top3_stage1");

                    int t2 = 256;
                    reduce_top3_stage2<<<1, t2, (size_t)t2 * sizeof(Top3Small)>>>(d_cand, nb * 3, d_best3);
                    cuda_check(cudaGetLastError(), "reduce_top3_stage2");

                    // gather leaders
                    int gthreads = 256;
                    int gblocks = (NDIM + gthreads - 1) / gthreads;
                    gather_leaders_kernel<<<gblocks, gthreads>>>(d_pos, NDIM, d_best3, d_L0, d_L1, d_L2);
                    cuda_check(cudaGetLastError(), "gather_leaders_kernel");

                    // update
                    double a = 2.0 * (1.0 - (double)iter / (double)MAX_ITERS);
                    dim3 ugrid(POP, (NDIM + 255) / 256, 1);
                    dim3 ublock(256, 1, 1);
                    gwo_update_kernel<<<ugrid, ublock>>>(d_pos, POP, NDIM, d_L0, d_L1, d_L2, a, iter, SEED);
                    cuda_check(cudaGetLastError(), "gwo_update_kernel");
                }

                cuda_check(cudaDeviceSynchronize(), "sync end run");

                int h_best3[3] = {-1,-1,-1};
                std::vector<double> h_sse(POP);

                cuda_check(cudaMemcpy(h_best3, d_best3, 3 * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy best3");
                cuda_check(cudaMemcpy(h_sse.data(), d_sse, bytesSse, cudaMemcpyDeviceToHost), "Memcpy sse");

                int idx0 = h_best3[0];
                if (idx0 >= 0 && idx0 < POP) best_last = h_sse[(size_t)idx0];

                auto t1 = std::chrono::steady_clock::now();
                long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                total_ms += ms;

                std::cout << "Run " << r
                          << " | " << ms << " ms"
                          << " | best SSE(OpenMP-style)=" << best_last
                          << " | idx0=" << idx0 << "\n";
            }

            double avg_ms = total_ms / double(RUNS);
            std::cout << "Avg: " << avg_ms << " ms\n";

            // CSV
            {
                std::ofstream csv(filename, std::ios::app);
                if (!csv) {
                    std::cerr << "Cannot open csv: " << filename << "\n";
                    return 1;
                }
                if (need_header) {
                    csv << "impl,limit_train,Ndata,D,K,NDIM,POP_SIZE,max_iters,runs,avg_ms,best_sse,device\n";
                    need_header = false;
                }

                csv << "cuda"
                    << "," << LIMIT_TRAIN
                    << "," << Ndata
                    << "," << D
                    << "," << K
                    << "," << NDIM
                    << "," << POP
                    << "," << MAX_ITERS
                    << "," << RUNS
                    << "," << avg_ms
                    << "," << best_last
                    << "," << prop.name
                    << "\n";
            }

            // free per POP
            cudaFree(d_cand);
            cudaFree(d_best3);
            cudaFree(d_L2);
            cudaFree(d_L1);
            cudaFree(d_L0);
            cudaFree(d_sse);
            cudaFree(d_pos);
        }

        cudaFree(d_X);
    }

    std::cout << "Done. CSV: " << filename << "\n";
    return 0;
}
