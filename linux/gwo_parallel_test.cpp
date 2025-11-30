#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>
#include "gwo_parallel.hpp"

// g++ -O2 -std=c++20 -fopenmp -I "../eigen-5.0.0" gwo_parallel_test.cpp -o parallel_test

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
    std::vector<int> N_list = {10, 50, 100, 500, 1000};
    std::vector<int> Pop_list = {50, 100, 200, 500, 1000
                                // , 2000, 5000, 10000, 20000
                                // , 50000, 100000
                                };
    std::vector<int> thread_list = {2, 4, 8, 16, 20};

    const int RUNS = 5;
    std::string problem_name = "Rastrigin";
    std::string filename = "gwo_parallel.csv";

    GWO::Setup setup;
    for (int N : N_list) {
        for (int POP_SIZE : Pop_list) {
            setup.N = N;
            setup.POP_SIZE = POP_SIZE;
            setup.maxRange = Eigen::ArrayXd::Constant(setup.N, 5);
            setup.minRange = Eigen::ArrayXd::Constant(setup.N, -5);
            bool need_header = (!std::filesystem::exists(filename) ||
                                std::filesystem::file_size(filename) == 0);

            std::ofstream csv(filename, std::ios::app);
            if (!csv) {
                std::cerr << "Cannot open gwo_parallel.csv\n";
                return 1;
            }

            if (need_header) {
                csv << "problem,N,POP_SIZE,threads,avg_ms,best_fitness_last_run\n";
            }

            for (int threads : thread_list)
            {
                #ifdef _OPENMP
                omp_set_num_threads(threads);
                #endif

                long long total_ms = 0;
                double best_fitness_last_run = 0.0;

                std::cout<<"================================";
                std::cout<<"\nN=" << N << ", POP_SIZE=" << POP_SIZE << ", Threads = "<<threads<<"\n";

                for (int r = 1; r <= RUNS; r++)
                {
                    GWO::global_seed = 123456789ULL;

                    RastriginProblem problem(setup);

                    auto start = std::chrono::steady_clock::now();
                    auto best = problem.run(1000);
                    auto end = std::chrono::steady_clock::now();

                    long long ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                    total_ms += ms;
                    best_fitness_last_run = best.savedFitness;

                    std::cout << "Run " << r << ": " << ms << " ms"
                            << " | best fitness = " << best.savedFitness << "\n";
                }

                double avg_ms = total_ms / double(RUNS);

                std::cout << ">>> Average time (" << threads << " threads): "
                        << avg_ms << " ms\n";

                csv << problem_name
                    << "," << setup.N
                    << "," << setup.POP_SIZE
                    << "," << threads
                    << "," << avg_ms
                    << "," << best_fitness_last_run
                    << "\n";
            }

            csv.close();
        }
    }
    return 0;
}
