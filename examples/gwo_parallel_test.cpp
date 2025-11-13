#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include "gwo_parallel.hpp"

// g++ -O2 -std=c++20 -fopenmp -I "..\eigen-5.0.0" gwo_parallel_test.cpp -o parallel_test.exe

struct SphereProblem : public GWO::Problem<double>
{
    using GWO::Problem<double>::Problem;

    double fitness(const Eigen::ArrayXd &pos) const override {
        return pos.square().sum();
    }
};

struct RastriginProblem : public GWO::Problem<double>
{
    using GWO::Problem<double>::Problem;

    double fitness(const Eigen::ArrayXd &pos) const override {
        const double A = 10.0;
        const double two_pi = 2.0 * M_PI;
        return A * pos.size() + (pos.square() - A * (two_pi * pos).cos()).sum();
    }
};

int main() {
    GWO::Setup setup;
    setup.N = 100;
    setup.POP_SIZE = 500;
    setup.maxRange = Eigen::ArrayXd::Constant(setup.N, 10.0);
    setup.minRange = Eigen::ArrayXd::Constant(setup.N, -10.0);

    
    int thread_tests[] = {1, 2, 4, 8, 16, 20};
    int num_tests = sizeof(thread_tests) / sizeof(thread_tests[0]);

#ifdef _OPENMP
    int max_threads_avail = omp_get_max_threads();
    std::cout << "System max threads available: " << max_threads_avail << "\n";
#endif

    std::cout << "=== Grey Wolf Optimization Parallel Benchmark ===\n";
    std::cout << "N = " << setup.N 
              << ", POP = " << setup.POP_SIZE 
              << ", Iter = 1000\n\n";

    for (int i = 0; i < num_tests; i++)
    {
        int threads = thread_tests[i];

#ifdef _OPENMP
        if (threads > max_threads_avail) {
            std::cout << "Skipping " << threads 
                      << " threads (system only has " 
                      << max_threads_avail << ")\n";
            continue;
        }
        omp_set_num_threads(threads);
#endif

        // Seed để cố định giá trị
        GWO::global_seed  = 123456789ULL;

        RastriginProblem problem(setup);

        auto start = std::chrono::steady_clock::now();
        auto best = problem.run(1000);
        auto end = std::chrono::steady_clock::now();

        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

#ifdef _OPENMP
        int t_used = threads;
#else
        int t_used = 1;
#endif

        std::cout << "=== Threads: " << t_used << " ===\n";
        std::cout << "Time: " << ms << " ms\n";
        std::cout << "Best fitness: " << best.savedFitness << "\n";
        std::cout << "Best pos preview: [";
        for (int k = 0; k < std::min<int>(5, setup.N); k++)
            std::cout << best.pos[k] << (k < 4 ? ", " : "]\n");
        std::cout << "\n";
    }

    return 0;
}
