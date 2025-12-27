// gwo_serial_UCI.cpp
// Build:
//   g++ -O2 -std=c++20 -I "../eigen-5.0.0" gwo_serial_UCI.cpp -o serial_UCI.exe
//
// Run from examples/ (working directory = examples/):
//   ./serial_UCI.exe
//
// Expected dataset paths (relative to examples/):
//   ../madelon/MADELON/madelon_train.data
//   ../madelon/MADELON/madelon_train.labels
//   ../madelon/MADELON/madelon_valid.data
//   ../madelon/MADELON/madelon_valid.labels

#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <random>

#include "gwo_serial_sync.hpp"

// -------------------- IO --------------------
std::vector<std::vector<double>> load_data(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Cannot open data file: " + path);
    }

    std::vector<std::vector<double>> X;
    std::string line;
    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        row.reserve(512);
        double val;
        while (ss >> val) row.push_back(val);
        X.push_back(std::move(row));
    }
    return X;
}

std::vector<int> load_labels(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Cannot open label file: " + path);
    }

    std::vector<int> y;
    y.reserve(4096);
    int label;
    while (fin >> label) {
        y.push_back(label);
    }
    return y;
}

// -------------------- Reproducibility helpers --------------------
static int count_selected_features(const Eigen::ArrayXd& pos) {
    int cnt = 0;
    for (int i = 0; i < pos.size(); ++i) {
        if (pos(i) > 0) cnt++;
    }
    return cnt;
}

// Hash “dấu vân tay” theo bit chọn/không chọn feature (ổn định giữa các lần chạy)
static uint64_t hash_solution_signature(const Eigen::ArrayXd& pos) {
    // FNV-1a 64-bit style
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < pos.size(); ++i) {
        uint64_t b = (pos(i) > 0) ? 1ULL : 0ULL;
        h ^= b;
        h *= 1099511628211ULL;
    }
    return h;
}

// -------------------- Problem: Madelon Feature Selection (KNN) --------------------
struct MadelonFSProblem : public GWO::Problem<double>
{
    const std::vector<std::vector<double>>& X_train;
    const std::vector<int>& y_train;
    const std::vector<std::vector<double>>& X_valid;
    const std::vector<int>& y_valid;

    // Subsample indices for valid to speed up fitness during optimization
    std::vector<int> valid_idx;

    double alpha = 0.99;
    double beta  = 0.01;
    int k = 5;

    MadelonFSProblem(
        const GWO::Setup& setup_,
        const std::vector<std::vector<double>>& Xt,
        const std::vector<int>& yt,
        const std::vector<std::vector<double>>& Xv,
        const std::vector<int>& yv,
        std::vector<int> valid_idx_
    )
    : GWO::Problem<double>(setup_),
      X_train(Xt), y_train(yt),
      X_valid(Xv), y_valid(yv),
      valid_idx(std::move(valid_idx_))
    {
        if (X_train.empty() || X_valid.empty())
            throw std::runtime_error("Empty dataset!");

        if (X_train[0].size() != setup_.N || X_valid[0].size() != setup_.N)
            throw std::runtime_error("Feature dimension mismatch with setup.N");

        if (y_train.size() != X_train.size() || y_valid.size() != X_valid.size())
            throw std::runtime_error("Label size mismatch with data size");

        if ((int)X_train.size() < k)
            throw std::runtime_error("Train size < k");

        if (valid_idx.empty()) {
            valid_idx.resize((int)X_valid.size());
            std::iota(valid_idx.begin(), valid_idx.end(), 0);
        }
    }

    double fitness(const Eigen::ArrayXd &pos) const override
    {
        // ---- select features ----
        std::vector<int> feat_idx;
        feat_idx.reserve(pos.size());
        for (int i = 0; i < pos.size(); i++) {
            if (pos(i) > 0) feat_idx.push_back(i);
        }
        if (feat_idx.empty())
            return 1e9;

        const size_t n_train = X_train.size();
        std::vector<std::pair<double,int>> dist(n_train);

        int correct = 0;

        for (int ii : valid_idx) {
            const auto& xv = X_valid[ii];

            // distance to all train points
            for (size_t j = 0; j < n_train; j++) {
                const auto& xt = X_train[j];
                double d = 0.0;
                for (int f : feat_idx) {
                    double diff = xv[f] - xt[f];
                    d += diff * diff;
                }
                dist[j] = {d, y_train[j]};
            }

            // pick k nearest
            std::partial_sort(
                dist.begin(), dist.begin() + k, dist.end(),
                [](const auto& a, const auto& b){ return a.first < b.first; }
            );

            int vote = 0;
            for (int t = 0; t < k; t++) vote += dist[t].second;

            int pred = (vote >= 0 ? 1 : -1);
            if (pred == y_valid[ii]) correct++;
        }

        double acc = double(correct) / double(valid_idx.size());
        double feat_ratio = double(feat_idx.size()) / double(pos.size());

        return alpha * (1.0 - acc) + beta * feat_ratio;
    }
};

int main() {
    // NOTE:
    // - MADELON always has 500 features -> setup.N fixed = 500
    // - The original N_list loop is misleading; removed for correctness.

    std::vector<int> Pop_list = {10, 25, 50, 100}; // tăng dần sau khi chạy ổn

    // Debug-friendly defaults (so program doesn’t “look stuck”)
    const int RUNS = 2;                // chạy lặp để kiểm tra reproducible
    const int MAX_ITERS = 40;          // 1000 với KNN brute-force sẽ rất lâu
    const int VALID_SUBSAMPLE = 200;   // dùng 200/600 valid trong fitness để tăng tốc

    const uint64_t GWO_SEED = 123456789ULL;
    const uint32_t SHUFFLE_SEED = 123; // cố định để subset valid không đổi

    std::string problem_name = "Madelon_FS";
    std::string filename = "gwo_serial_UCI.csv";

    try {
        // ---- load dataset ----
        auto X_train = load_data("../madelon/MADELON/madelon_train.data");
        auto y_train = load_labels("../madelon/MADELON/madelon_train.labels");
        auto X_valid = load_data("../madelon/MADELON/madelon_valid.data");
        auto y_valid = load_labels("../madelon/MADELON/madelon_valid.labels");

        std::cout << "Current path: " << std::filesystem::current_path() << "\n";
        std::cout << "Loaded train=" << X_train.size() << "x" << X_train[0].size()
                  << " | valid=" << X_valid.size() << "x" << X_valid[0].size() << "\n";

        // ---- build fixed subset indices for valid (reproducible) ----
        std::vector<int> valid_idx((int)X_valid.size());
        std::iota(valid_idx.begin(), valid_idx.end(), 0);
        std::shuffle(valid_idx.begin(), valid_idx.end(), std::mt19937(SHUFFLE_SEED));
        if ((int)valid_idx.size() > VALID_SUBSAMPLE) valid_idx.resize(VALID_SUBSAMPLE);

        // ---- setup ----
        GWO::Setup setup;
        setup.N = 500;
        setup.maxRange = Eigen::ArrayXd::Constant(setup.N, 1);
        setup.minRange = Eigen::ArrayXd::Constant(setup.N, -1);

        // ---- run experiments ----
        for (int POP_SIZE : Pop_list) {
            setup.POP_SIZE = POP_SIZE;

            std::cout << "================================\n";
            std::cout << "N=" << setup.N
                      << ", POP_SIZE=" << POP_SIZE
                      << ", MAX_ITERS=" << MAX_ITERS
                      << ", VALID_SUBSAMPLE=" << (int)valid_idx.size()
                      << "\n";

            long long total_ms = 0;
            double best_fitness_last_run = 0.0;
            int selected_features_last_run = 0;
            uint64_t signature_last_run = 0;

            uint64_t signature_run1 = 0;

            for (int r = 1; r <= RUNS; r++) {
                // Fix seed => same initial wolves + same random updates => reproducible
                GWO::global_seed = GWO_SEED;

                MadelonFSProblem problem(setup, X_train, y_train, X_valid, y_valid, valid_idx);

                auto start = std::chrono::steady_clock::now();
                auto best = problem.run(MAX_ITERS);
                auto end = std::chrono::steady_clock::now();

                long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                total_ms += ms;

                best_fitness_last_run = best.savedFitness;
                selected_features_last_run = count_selected_features(best.pos);
                signature_last_run = hash_solution_signature(best.pos);

                std::cout << "Run " << r << ": " << ms << " ms"
                          << " | best fitness = " << best.savedFitness
                          << " | selected_features = " << selected_features_last_run
                          << " | signature = " << signature_last_run
                          << "\n";

                if (r == 1) signature_run1 = signature_last_run;
                if (r > 1 && signature_last_run != signature_run1) {
                    std::cout << "WARNING: signature differs from run1 => not reproducible!\n";
                }
            }

            double avg_ms = total_ms / double(RUNS);

            bool need_header = (!std::filesystem::exists(filename) ||
                                std::filesystem::file_size(filename) == 0);

            std::ofstream csv(filename, std::ios::app);
            if (!csv) {
                std::cerr << "Cannot open " << filename << " for writing!\n";
                return 1;
            }

            if (need_header) {
                csv << "problem,N,POP_SIZE,max_iters,valid_subsample,avg_ms,"
                       "best_fitness_last_run,selected_features,signature\n";
            }

            csv << problem_name
                << "," << setup.N
                << "," << setup.POP_SIZE
                << "," << MAX_ITERS
                << "," << (int)valid_idx.size()
                << "," << avg_ms
                << "," << best_fitness_last_run
                << "," << selected_features_last_run
                << "," << signature_last_run
                << "\n";

            csv.close();

            std::cout << ">>> Average over " << RUNS << " runs: " << avg_ms << " ms\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        std::cerr << "Tip: make sure you run from examples/ so relative paths resolve.\n";
        return 1;
    }

    return 0;
}
