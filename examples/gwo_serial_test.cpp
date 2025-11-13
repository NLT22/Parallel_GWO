#include <iostream>
#define _USE_MATH_DEFINES
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

    // Seed để cố định giá trị
    GWO::rng.state = 123456789ULL;  

    RastriginProblem problem(setup);

    auto start = std::chrono::steady_clock::now();
    auto best = problem.run(1000);  
    auto end = std::chrono::steady_clock::now();

    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Time: " << ms << " ms\n";
    std::cout << "Best fitness: " << best.savedFitness << "\n";
    std::cout << "Best position (first 5 dims): [";
    for (int i = 0; i < std::min<int>(5, setup.N); ++i)
        std::cout << best.pos[i] << (i < 4 ? ", " : "]\n");
}
