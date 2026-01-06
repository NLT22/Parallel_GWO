// gwo_parallel_kmeans_bm.cpp
#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "gwo_parallel_sync_fp32.hpp"

// g++ -O2 -std=c++20 -fopenmp gwo_parallel_kmeans_bm.cpp -o parallel_kmeans.exe -I ../eigen-5.0.0

// ===================== MNIST IDX LOADER (embedded) =====================
static inline uint32_t read_be_u32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) throw std::runtime_error("Failed to read u32");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

struct MNIST {
    int n = 0, rows = 0, cols = 0;
    std::vector<float> X;
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

// ===================== KMEANS SSE FITNESS (embedded) =====================
static inline float kmeans_sse_cpu(const float* X, int N, int D,
                                    const float* centroids, int K)
{
    float sse = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float* xi = X + (size_t)i * D;
        float best = 3.4e38f;

        for (int k = 0; k < K; ++k) {
            const float* ck = centroids + (size_t)k * D;
            float dist = 0.0f;
            for (int d = 0; d < D; ++d) {
                float diff = xi[d] - ck[d];
                dist += diff * diff;
            }
            if (dist < best) best = dist;
        }
        sse += best;
    }
    return sse;
}

// ===================== GWO problem =====================
struct KMeansProblemCPU_OMP : public GWO::Problem<float> {
    using GWO::Problem<float>::Problem;

    const float* X = nullptr;
    int Ndata = 0, D = 0, K = 0;

    KMeansProblemCPU_OMP(const GWO::Setup& s, const float* X_, int N_, int D_, int K_)
        : GWO::Problem<float>(s), X(X_), Ndata(N_), D(D_), K(K_) {}

    float fitness(const Eigen::ArrayXf& pos) const override {
        // thread-local buffer to avoid per-call allocation
        thread_local std::vector<float> c;
        c.resize((size_t)K * (size_t)D);

        for (int i = 0; i < K * D; ++i) c[(size_t)i] = (float)pos(i);
        return kmeans_sse_cpu(X, Ndata, D, c.data(), K);
    }
};

int main() {
    // ===================== CONFIG =====================
    const int LIMIT_TRAIN = 5000; 
    const int K = 10;
    const int RUNS = 1;
    const int MAX_ITERS = 100;
    const uint64_t SEED = 123456789ULL;

    std::vector<int> Pop_list = {25, 50, 100};
    std::vector<int> thread_list = {2, 4, 8, 16, 20};

    // ===================== Load MNIST =====================
    MNIST train = load_mnist_images_labels(
        "./mnist/train-images-idx3-ubyte",
        "./mnist/train-labels-idx1-ubyte",
        LIMIT_TRAIN
    );
    const int D = train.dim();
    const int Ndata = train.n;

    std::cout << "Loaded MNIST: N=" << Ndata << ", D=" << D << " ("
              << train.rows << "x" << train.cols << ")\n";

    // ===================== CSV =====================
    std::string filename = "gwo_openmp.csv";
    bool need_header = (!std::filesystem::exists(filename) ||
                        std::filesystem::file_size(filename) == 0);

    std::ofstream csv(filename, std::ios::app);
    if (!csv) {
        std::cerr << "Cannot open csv: " << filename << "\n";
        return 1;
    }
    if (need_header) {
        csv << "impl,limit_train,Ndata,D,K,POP_SIZE,threads,max_iters,avg_ms,best_sse\n";
    }

    for (int POP_SIZE : Pop_list) {
        GWO::Setup setup;
        setup.N = (size_t)(K * D);
        setup.POP_SIZE = (size_t)POP_SIZE;
        setup.minRange = Eigen::ArrayXf::Constant((int)setup.N, 0.0f);
        setup.maxRange = Eigen::ArrayXf::Constant((int)setup.N, 1.0f);

        for (int threads : thread_list) {
            #ifdef _OPENMP
            omp_set_dynamic(0);
            omp_set_num_threads(threads);
            omp_set_nested(0);
            #endif

            long long total_ms = 0;
            float best_last = 0.0f;

            std::cout << "================================\n";
            std::cout << "OpenMP GWO | KMeans(MNIST)"
                      << " | Ndata=" << Ndata
                      << " | D=" << D
                      << " | K=" << K
                      << " | POP=" << POP_SIZE
                      << " | threads=" << threads
                      << " | ITERS=" << MAX_ITERS << "\n";

            for (int r = 1; r <= RUNS; ++r) {
                GWO::global_seed = SEED;
                KMeansProblemCPU_OMP problem(setup, train.X.data(), Ndata, D, K);

                auto st = std::chrono::steady_clock::now();
                auto best = problem.run(MAX_ITERS);
                auto ed = std::chrono::steady_clock::now();

                long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();
                total_ms += ms;
                best_last = best.savedFitness;

                std::cout << "Run " << r << " | " << ms << " ms | best SSE=" << best_last << "\n";
            }

            double avg_ms = total_ms / double(RUNS);
            std::cout << "Avg: " << avg_ms << " ms\n";

            csv << "openmp"
                << "," << LIMIT_TRAIN
                << "," << Ndata << "," << D << "," << K
                << "," << POP_SIZE
                << "," << threads
                << "," << MAX_ITERS
                << "," << avg_ms
                << "," << best_last
                << "\n";
            csv.flush();
        }
    }

    csv.close();
    return 0;
}
