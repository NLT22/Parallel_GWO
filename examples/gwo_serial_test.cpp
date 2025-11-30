#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>      
#include <fstream>      
#include "gwo_serial.hpp"
#include "gwo_pcc.hpp"
#include <filesystem> 

// g++ -O2 -std=c++20 -I "../eigen-5.0.0" gwo_serial_test.cpp -o serial_test.exe

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
    const int RUNS = 5; // Số lần chạy để lấy thời gian trung bình
    std::string problem_name = "Rastrigin";
    std::string filename = "gwo_serial.csv";
    
    GWO::Setup setup;
    
    for (int N : N_list) {
        for (int POP_SIZE : Pop_list) {
            std::cout<<"================================";
            std::cout<<"\nN=" << N << ", POP_SIZE=" << POP_SIZE << "\n";
            setup.N = N;                
            setup.POP_SIZE = POP_SIZE;         
            setup.maxRange = Eigen::ArrayXd::Constant(setup.N, 5);
            setup.minRange = Eigen::ArrayXd::Constant(setup.N, -5);

            long long total_ms = 0;
            double best_fitness_last_run = 0.0;            

            for (int r = 1; r <= RUNS; r++)
            {
                GWO::rng.state = 123456789ULL;  

                RastriginProblem problem(setup);

                auto start = std::chrono::steady_clock::now();
                auto best = problem.run(1000);  
                auto end = std::chrono::steady_clock::now();

                long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                total_ms += ms;
                best_fitness_last_run = best.savedFitness;  

                std::cout << "Run " << r << ": " << ms << " ms"
                        << " | best fitness = " << best.savedFitness << "\n";
            }

            double avg = total_ms / double(RUNS);

            std::cout << "-----------------------------------\n";
            std::cout << "Average time over " << RUNS << " runs: " << avg << " ms\n";

            bool need_header = (!std::filesystem::exists(filename) ||
                                std::filesystem::file_size(filename) == 0);

            std::ofstream csv(filename, std::ios::app);

            if (!csv) {
                std::cerr << "Cannot open gwo_results.csv for writing!\n";
                return 1;
            }

            if (need_header) {
                csv << "problem,N,POP_SIZE,avg_ms,best_fitness_last_run\n";
            }

            csv << problem_name
                << "," << setup.N
                << "," << setup.POP_SIZE
                << "," << avg
                << "," << best_fitness_last_run
                << "\n";

            csv.close();
        }
    }
    return 0;
}
