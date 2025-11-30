#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm> // std::max

#include "gwo_serial_analyze.hpp"

// g++ -O2 -std=c++20 -I "../eigen-5.0.0" gwo_serial_analyze.cpp -o serial_analyze

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
    std::vector<int> N_list   = {10, 50, 100, 500, 1000};
    std::vector<int> Pop_list = {50, 100, 200, 500, 1000};
    const int RUNS = 5;  // số lần chạy để lấy thời gian trung bình

    std::string problem_name = "Rastrigin";
    std::string filename     = "gwo_serial_analyze.csv";

    GWO::Setup setup;

    // Chuẩn bị file CSV
    bool need_header = (!std::filesystem::exists(filename) ||
                        std::filesystem::file_size(filename) == 0);

    std::ofstream csv(filename, std::ios::app);
    if (!csv) {
        std::cerr << "Cannot open " << filename << " for writing!\n";
        return 1;
    }

    if (need_header) {
        csv << "problem,N,POP_SIZE,"
            << "avg_wall_ms,"
            << "run_ms,fit_ms,upd_ms,heap_ms,other_ms,"
            << "fit_pct,upd_pct,heap_pct,other_pct\n";
    }

    for (int N : N_list) {
        for (int POP_SIZE : Pop_list) {
            std::cout << "================================\n";
            std::cout << "N=" << N << ", POP_SIZE=" << POP_SIZE << "\n";

            setup.N        = N;
            setup.POP_SIZE = POP_SIZE;
            setup.maxRange = Eigen::ArrayXd::Constant(setup.N, 5);
            setup.minRange = Eigen::ArrayXd::Constant(setup.N, -5);

            long long total_wall_ms = 0;
            double best_fitness_last_run = 0.0;

            // reset profiler cho cấu hình này
            GWO::Profiling::t_fitness_batch_ns = 0;
            GWO::Profiling::t_updatePop_ns     = 0;
            GWO::Profiling::t_updateFH_ns      = 0;
            GWO::Profiling::t_run_ns           = 0;

            for (int r = 1; r <= RUNS; r++)
            {
                // reset RNG để các run giống nhau
                GWO::rng.state = 123456789ULL;

                RastriginProblem problem(setup);

                auto start = std::chrono::steady_clock::now();
                auto best  = problem.run(1000);   // GWO tuần tự
                auto end   = std::chrono::steady_clock::now();

                long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                total_wall_ms += ms;
                best_fitness_last_run = best.savedFitness;
            }

            double avg_wall_ms = total_wall_ms / double(RUNS);

            // ====== Phân tích profiler theo "trung bình 1 run" ======
            auto ns_to_ms = [](double ns) { return ns / 1e6; };

            double total_run_ns = static_cast<double>(GWO::Profiling::t_run_ns) / RUNS;
            double fit_ns       = static_cast<double>(GWO::Profiling::t_fitness_batch_ns) / RUNS;
            double upd_ns       = static_cast<double>(GWO::Profiling::t_updatePop_ns) / RUNS;
            double fh_ns        = static_cast<double>(GWO::Profiling::t_updateFH_ns) / RUNS;

            double fh_exclusive_ns = fh_ns - fit_ns; // phần heap/copy không tính fitness
            if (fh_exclusive_ns < 0) fh_exclusive_ns = 0;

            double run_ms  = ns_to_ms(total_run_ns);
            double fit_ms  = ns_to_ms(fit_ns);
            double upd_ms  = ns_to_ms(upd_ns);
            double heap_ms = ns_to_ms(fh_exclusive_ns);

            double other_ms = ns_to_ms(total_run_ns - (fit_ns + upd_ns + fh_exclusive_ns));

            double p_fit = 0.0, p_upd = 0.0, p_heap = 0.0, p_other = 0.0;
            if (total_run_ns > 0) {
                p_fit  = 100.0 * fit_ns          / total_run_ns;
                p_upd  = 100.0 * upd_ns          / total_run_ns;
                p_heap = 100.0 * fh_exclusive_ns / total_run_ns;
                p_other= 100.0 * other_ns        / total_run_ns;
            }

            // In ra console tóm tắt cho cấu hình này
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Avg wall time over " << RUNS << " runs: "
                      << avg_wall_ms << " ms\n";

            std::cout << "=== Profiling per-run (serial) ===\n";
            std::cout << "  run():  " << run_ms  << " ms (100%)\n";
            std::cout << "    fitness_batch: " << fit_ms  << " ms (" << p_fit  << "%)\n";
            std::cout << "    updatePopulation: " << upd_ms << " ms (" << p_upd  << "%)\n";
            std::cout << "    heap/copy (exclusive): " << heap_ms << " ms (" << p_heap << "%)\n";
            std::cout << "    others: " << other_ms << " ms (" << p_other << "%)\n\n";

            // Ghi CSV: 1 dòng / (N, POP_SIZE)
            csv << problem_name
                << "," << setup.N
                << "," << setup.POP_SIZE
                << "," << avg_wall_ms
                << "," << run_ms
                << "," << fit_ms
                << "," << upd_ms
                << "," << heap_ms
                << "," << other_ms
                << "," << p_fit
                << "," << p_upd
                << "," << p_heap
                << "," << p_other
                << "\n";
        }
    }

    csv.close();
    return 0;
}
