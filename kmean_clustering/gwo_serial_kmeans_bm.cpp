// gwo_serial_kmeans_bm.cpp
#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <cmath>

#include "gwo_serial_analyze.hpp"
// Build (MSYS2):
// g++ -O2 -std=c++20 gwo_serial_kmeans_bm.cpp -o serial_kmeans.exe -I ../eigen-5.0.0

// ===================== MNIST IDX LOADER (embedded) =====================
static inline uint32_t read_be_u32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) throw std::runtime_error("Failed to read u32");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

struct MNIST {
    int n = 0, rows = 0, cols = 0;
    std::vector<float> X;      // normalized [0,1], n x (rows*cols)
    std::vector<uint8_t> y;    // labels
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
static inline double kmeans_sse_cpu(const float* X, int N, int D,
                                    const float* centroids, int K)
{
    double sse = 0.0;
    for (int i = 0; i < N; ++i) {
        const float* xi = X + (size_t)i * D;
        double best = 1e300;

        for (int k = 0; k < K; ++k) {
            const float* ck = centroids + (size_t)k * D;
            double dist = 0.0;
            for (int d = 0; d < D; ++d) {
                double diff = (double)xi[d] - (double)ck[d];
                dist += diff * diff;
            }
            if (dist < best) best = dist;
        }
        sse += best;
    }
    return sse;
}

// ===================== GWO problem: optimize centroids =====================
struct KMeansProblemCPU : public GWO::Problem<double> {
    using GWO::Problem<double>::Problem;

    const float* X = nullptr;
    int Ndata = 0, D = 0, K = 0;

    KMeansProblemCPU(const GWO::Setup& s, const float* X_, int N_, int D_, int K_)
        : GWO::Problem<double>(s), X(X_), Ndata(N_), D(D_), K(K_) {}

    double fitness(const Eigen::ArrayXd& pos) const override {
        std::vector<float> c((size_t)K * (size_t)D);
        for (int i = 0; i < K * D; ++i) c[(size_t)i] = (float)pos(i);
        return kmeans_sse_cpu(X, Ndata, D, c.data(), K);
    }
};

static inline long long ms_since(const std::chrono::steady_clock::time_point& a,
                                 const std::chrono::steady_clock::time_point& b)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

int main() {
    using clock = std::chrono::steady_clock;

    // ===================== CONFIG =====================
    const int LIMIT_TRAIN = 1000;
    const int K = 10;
    const int RUNS = 1;
    const int MAX_ITERS = 100;
    const uint64_t SEED = 123456789ULL;

    std::vector<int> Pop_list = {32, 64, 128};

    // ===================== Load MNIST (timed) =====================
    auto t_load0 = clock::now();
    MNIST train = load_mnist_images_labels(
        "./mnist/train-images-idx3-ubyte",
        "./mnist/train-labels-idx1-ubyte",
        LIMIT_TRAIN
    );
    auto t_load1 = clock::now();
    long long mnist_load_ms = ms_since(t_load0, t_load1);

    const int D = train.dim();
    const int Ndata = train.n;

    std::cout << "Loaded MNIST: N=" << Ndata << ", D=" << D
              << " (" << train.rows << "x" << train.cols << ")"
              << " | load=" << mnist_load_ms << " ms\n";

    // ===================== CSV =====================
    std::string filename = "gwo_serial.csv";
    bool need_header = (!std::filesystem::exists(filename) ||
                        std::filesystem::file_size(filename) == 0);

    for (int POP_SIZE : Pop_list) {
        // setup-only timing (not including Problem ctor)
        auto t_setup0 = clock::now();
        GWO::Setup setup;
        setup.N = (size_t)(K * D);
        setup.POP_SIZE = (size_t)POP_SIZE;
        setup.minRange = Eigen::ArrayXd::Constant((int)setup.N, 0.0);
        setup.maxRange = Eigen::ArrayXd::Constant((int)setup.N, 1.0);
        auto t_setup1 = clock::now();
        long long setup_only_ms = ms_since(t_setup0, t_setup1);

        long long total_run_ms = 0;
        long long total_setup_problem_ms = 0;
        double best_last = 0.0;

        // aggregated exclusive profiling (per RUNS)
        uint64_t sum_update_fitness_excl_ms = 0;
        uint64_t sum_update_pop_excl_ms     = 0;
        uint64_t sum_fitness_batch_excl_ms  = 0;
        uint64_t sum_fitness_scalar_ms      = 0;
        uint64_t sum_fitness_calls          = 0;

        std::cout << "================================\n";
        std::cout << "Serial GWO | KMeans(MNIST)"
                  << " | Ndata=" << Ndata
                  << " | D=" << D
                  << " | K=" << K
                  << " | POP=" << POP_SIZE
                  << " | ITERS=" << MAX_ITERS << "\n";

        for (int r = 1; r <= RUNS; ++r) {
            GWO::global_seed = SEED;
            GWO::Profiler::reset();

            auto t_prob0 = clock::now();
            KMeansProblemCPU problem(setup, train.X.data(), Ndata, D, K);
            auto t_prob1 = clock::now();
            long long setup_problem_ms = ms_since(t_prob0, t_prob1);
            total_setup_problem_ms += setup_problem_ms;

            auto st = clock::now();
            auto best = problem.run(MAX_ITERS);
            auto ed = clock::now();
            long long run_ms = ms_since(st, ed);

            total_run_ms += run_ms;
            best_last = best.savedFitness;

            // collect EXCLUSIVE counters
            sum_update_fitness_excl_ms += GWO::Profiler::t_update_fitness_excl_ms;
            sum_update_pop_excl_ms     += GWO::Profiler::t_update_pop_excl_ms;
            sum_fitness_batch_excl_ms  += GWO::Profiler::t_fitness_batch_excl_ms;
            sum_fitness_scalar_ms      += GWO::Profiler::t_fitness_scalar_ms;
            sum_fitness_calls          += GWO::Profiler::n_fitness_calls;

            std::cout << "Run " << r
                      << " | setup_problem=" << setup_problem_ms << " ms"
                      << " | run=" << run_ms << " ms"
                      << " | best SSE=" << best_last
                      << " | fitness_calls=" << (uint64_t)GWO::Profiler::n_fitness_calls
                      << "\n";
        }

        double avg_run_ms = total_run_ms / double(RUNS);
        double avg_setup_problem_ms = total_setup_problem_ms / double(RUNS);

        double avg_update_fitness_excl_ms = sum_update_fitness_excl_ms / double(RUNS);
        double avg_update_pop_excl_ms     = sum_update_pop_excl_ms     / double(RUNS);
        double avg_fitness_batch_excl_ms  = sum_fitness_batch_excl_ms  / double(RUNS);
        double avg_fitness_scalar_ms      = sum_fitness_scalar_ms      / double(RUNS);
        double avg_fitness_calls          = (double)sum_fitness_calls  / double(RUNS);

        std::cout << "Avg: run=" << avg_run_ms
                  << " ms | setup_problem=" << avg_setup_problem_ms << " ms\n";
        std::cout << "EXCL Profile(avg):\n"
                  << "  update_pop_excl=" << avg_update_pop_excl_ms << " ms\n"
                  << "  update_fitness_excl=" << avg_update_fitness_excl_ms << " ms\n"
                  << "  fitness_batch_excl=" << avg_fitness_batch_excl_ms << " ms\n"
                  << "  fitness_scalar=" << avg_fitness_scalar_ms << " ms\n"
                  << "  fitness_calls=" << (uint64_t)avg_fitness_calls << "\n";

        std::ofstream csv(filename, std::ios::app);
        if (!csv) {
            std::cerr << "Cannot open csv: " << filename << "\n";
            return 1;
        }

        if (need_header) {
            csv
              << "impl,limit_train,Ndata,D,K,POP_SIZE,max_iters,avg_ms,best_sse,"
              << "mnist_load_ms,setup_only_ms,avg_setup_problem_ms,"
              << "avg_update_pop_excl_ms,avg_update_fitness_excl_ms,avg_fitness_batch_excl_ms,avg_fitness_scalar_ms,avg_fitness_calls\n";
            need_header = false;
        }

        csv << "serial"
            << "," << LIMIT_TRAIN
            << "," << Ndata << "," << D << "," << K
            << "," << POP_SIZE
            << "," << MAX_ITERS
            << "," << avg_run_ms
            << "," << best_last
            << "," << mnist_load_ms
            << "," << setup_only_ms
            << "," << avg_setup_problem_ms
            << "," << avg_update_pop_excl_ms
            << "," << avg_update_fitness_excl_ms
            << "," << avg_fitness_batch_excl_ms
            << "," << avg_fitness_scalar_ms
            << "," << (uint64_t)avg_fitness_calls
            << "\n";
        csv.close();
    }

    return 0;
}
