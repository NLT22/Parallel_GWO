#include <iostream>
#include <cmath>
#include <chrono>
#include "gwo_serial.hpp"

// g++ -O2 -std=c++20 -I "..\eigen-5.0.0" gwo_serial_test.cpp -o serial_test.exe

struct SphereProblem : public GWO::Problem<double>
{
    using GWO::Problem<double>::Problem;
    double fitness(const Eigen::ArrayXd &pos) const override {
        return pos.square().sum();
    }
};

int main() {
    GWO::Setup setup;
    setup.N = 100;                
    setup.POP_SIZE = 500;         
    setup.maxRange = Eigen::ArrayXd::Constant(setup.N, 10.0);
    setup.minRange = Eigen::ArrayXd::Constant(setup.N, -10.0);

    // Seed để cấu hình kết quả
    GWO::rng.state = 123456789ULL;  

    SphereProblem problem(setup);

    auto start = std::chrono::steady_clock::now();
    auto best = problem.run(1000);  // 1000 vòng lặp
    auto end = std::chrono::steady_clock::now();

    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "=== Grey Wolf Optimization (Serial) ===\n";
    std::cout << "Dimensions: " << setup.N << "\n";
    std::cout << "Population: " << setup.POP_SIZE << "\n";
    std::cout << "Iterations: 1000\n";
    std::cout << "Time: " << ms << " ms\n";
    std::cout << "Best fitness: " << best.savedFitness << "\n";
    std::cout << "Best position (first 5 dims): [";
    for (int i = 0; i < std::min<int>(5, setup.N); ++i)
        std::cout << best.pos[i] << (i < 4 ? ", " : "]\n");
}
