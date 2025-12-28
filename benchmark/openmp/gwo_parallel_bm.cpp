// gwo_parallel_bm.cpp
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "gwo_parallel_sync.hpp"
// g++ -O2 -std=c++20 -I "../../eigen-5.0.0" gwo_parallel_bm.cpp -o parallel_bm.exe

// -------------------- Rastrigin benchmark problem --------------------
struct RastriginProblem : public GWO::Problem<double>
{
    using GWO::Problem<double>::Problem;

    double fitness(const Eigen::ArrayXd &pos) const override {
        const double A = 10.0;
        const double two_pi = 2.0 * M_PI;
        return A * pos.size()
             + (pos.square() - A * (two_pi * pos).cos()).sum();
    }
};

int main() {
    // -------------------- Benchmark parameters --------------------
    std::vector<int> N_list   = {10, 50, 100, 500, 1000};
    std::vector<int> Pop_list = {50, 100, 200, 500, 1000};
    std::vector<int> thread_list = {2, 4, 8, 16, 20};

    const int RUNS      = 5;
    const int MAX_ITERS = 1000;
    const uint64_t GLOBAL_SEED = 123456789ULL;

    std::string problem_name = "Rastrigin";
    std::string filename = "gwo_parallel_rastrigin.csv";

    bool need_header =
        (!std::filesystem::exists(filename) ||
         std::filesystem::file_size(filename) == 0);

    std::ofstream csv(filename, std::ios::app);
    if (need_header) {
        csv << "problem,N,POP_SIZE,threads,max_iters,avg_ms,best_fitness\n";
    }

    for (int N : N_list) {
        GWO::Setup setup;
        setup.N = N;
        setup.minRange = Eigen::ArrayXd::Constant(N, -5.12);
        setup.maxRange = Eigen::ArrayXd::Constant(N,  5.12);

        for (int POP_SIZE : Pop_list) {
            setup.POP_SIZE = POP_SIZE;

            for (int threads : thread_list) {
                #ifdef _OPENMP
                omp_set_num_threads(threads);
                omp_set_nested(0);
                #endif

                long long total_ms = 0;
                double best_fitness_last_run = 0.0;

                std::cout << "================================\n";
                std::cout << "OpenMP GWO | Rastrigin"
                          << " | N=" << N
                          << " | POP_SIZE=" << POP_SIZE
                          << " | threads=" << threads
                          << " | MAX_ITERS=" << MAX_ITERS << "\n";

                for (int r = 1; r <= RUNS; r++) {
                    GWO::global_seed = GLOBAL_SEED;

                    RastriginProblem problem(setup);

                    auto start = std::chrono::steady_clock::now();
                    auto best  = problem.run(MAX_ITERS);
                    auto end   = std::chrono::steady_clock::now();

                    long long ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                    total_ms += ms;
                    best_fitness_last_run = best.savedFitness;

                    std::cout << "Run " << r
                              << " | " << ms << " ms"
                              << " | best fitness = " << best.savedFitness
                              << "\n";
                }

                double avg_ms = total_ms / double(RUNS);

                csv << problem_name
                    << "," << N
                    << "," << POP_SIZE
                    << "," << threads
                    << "," << MAX_ITERS
                    << "," << avg_ms
                    << "," << best_fitness_last_run
                    << "\n";

                std::cout << ">>> Avg time: " << avg_ms << " ms\n";
            }
        }
    }

    csv.close();
    return 0;
}
