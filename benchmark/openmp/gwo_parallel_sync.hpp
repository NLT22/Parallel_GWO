// gwo_parallel_sync.hpp
#ifndef GWO_PARALLEL_SYNC_HPP
#define GWO_PARALLEL_SYNC_HPP

#include <iostream>
#include <vector>
#include <concepts>
#include <queue>
#include <Eigen/Dense>
#include <stdexcept>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GWO
{
    inline uint64_t global_seed = 123456789ULL;

    inline uint64_t splitmix64(uint64_t x)
    {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x = x ^ (x >> 31);
        return x;
    }

    inline double indexed_random(int iter, int wid, int dim, int channel)
    {
        uint64_t mix =
            (uint64_t)iter * 0x9E3779B97F4A7C15ULL ^
            (uint64_t)wid  * 0xBF58476D1CE4E5B9ULL ^
            (uint64_t)dim  * 0x94D049BB133111EBULL ^
            (uint64_t)channel * 0x123456789ABCDEFULL ^
            global_seed;

        uint64_t r = splitmix64(mix);
        return (r >> 11) * 0x1.0p-53;
    }

    namespace constants
    {
        inline size_t K = 3;
    }

    struct Setup
    {
        size_t N{};
        size_t POP_SIZE{};
        Eigen::ArrayXd maxRange;
        Eigen::ArrayXd minRange;
    };

    template <std::floating_point T>
    struct Wolf
    {
        Wolf(size_t n) : id(-1), savedFitness{}, pos(n), len(n) {}
        int id;              // stable tie-break id
        T savedFitness{};
        Eigen::ArrayX<T> pos;
        size_t len{};
    };

    template <std::floating_point T>
    class Comparator
    {
    public:
        bool operator()(const Wolf<T> &w1, const Wolf<T> &w2)
        {
            if (w1.savedFitness != w2.savedFitness)
                return w1.savedFitness < w2.savedFitness;
            return w1.id < w2.id; // tie-break
        }
    };

    template <std::floating_point T>
    struct Problem
    {
        virtual Eigen::ArrayX<T> fitness_batch(const Eigen::ArrayXX<T> &population_pos) const
        {
            Eigen::ArrayX<T> fitness_values(population_pos.rows());

            #pragma omp parallel for
            for (int i = 0; i < population_pos.rows(); ++i)
                fitness_values(i) = this->fitness(population_pos.row(i));

            return fitness_values;
        }

        virtual T fitness(const Eigen::ArrayX<T> &pos) const = 0;

        Problem(Setup _setup)
            : nextPos(_setup.N), A(_setup.N), C(_setup.N), setup(std::move(_setup))
        {
            if (setup.N == 0 || setup.POP_SIZE == 0)
                throw std::invalid_argument("N and POP_SIZE must be > 0.");
            if (setup.minRange.size() != (int)setup.N || setup.maxRange.size() != (int)setup.N)
                throw std::invalid_argument("minRange and maxRange must have size N.");
            if ((setup.maxRange < setup.minRange).any())
                throw std::invalid_argument("All elements of maxRange must be >= minRange.");

            population.reserve(setup.POP_SIZE);
            for (size_t i = 0; i < setup.POP_SIZE; i++)
            {
                population.emplace_back(setup.N);
                population.back().id = (int)i;

                for (size_t k = 0; k < setup.N; k++)
                {
                    double r = indexed_random(0, (int)i, (int)k, 77);
                    population.back().pos[(int)k] =
                        setup.minRange[(int)k] + r * (setup.maxRange[(int)k] - setup.minRange[(int)k]);
                }
            }
        }

        Wolf<T> run(int maxIterations)
        {
            update_fitness_and_heap();
            for (int iter = 0; iter < maxIterations; iter++)
            {
                T a = 2 * (1 - T(iter) / T(maxIterations));
                updatePopulation(iter, a);
            }
            return getBestKWolves()[0];
        }

        void update_fitness_and_heap()
        {
            Eigen::ArrayXX<T> positions((int)setup.POP_SIZE, (int)setup.N);
            for (size_t i = 0; i < setup.POP_SIZE; ++i)
                positions.row((int)i) = population[i].pos;

            Eigen::ArrayX<T> fitness_values = this->fitness_batch(positions);

            heap = {};
            for (size_t i = 0; i < setup.POP_SIZE; ++i)
            {
                population[i].savedFitness = fitness_values((int)i);
                heap.push(population[i]);
                if (heap.size() > constants::K)
                    heap.pop();
            }
        }

        void updatePopulation(int iter, T a)
        {
            auto bestWolves = getBestKWolves();

            #pragma omp parallel
            {
                Eigen::ArrayX<T> localA((int)setup.N);
                Eigen::ArrayX<T> localC((int)setup.N);
                Eigen::ArrayX<T> localNext((int)setup.N);

                #pragma omp for
                for (int i = 0; i < (int)setup.POP_SIZE; ++i)
                {
                    localNext.setZero();

                    for (size_t j = 0; j < constants::K; j++)
                    {
                        for (int k = 0; k < (int)setup.N; k++)
                        {
                            double r1 = indexed_random(iter, i, k, (int)(j * 2));
                            double r2 = indexed_random(iter, i, k, (int)(j * 2 + 1));

                            localA(k) = 2 * a * r1 - a;
                            localC(k) = 2 * r2;
                        }

                        auto D = (bestWolves[j].pos * localC - population[i].pos).abs();
                        localNext += bestWolves[j].pos - D * localA;
                    }

                    population[i].pos =
                        (localNext / T(constants::K))
                            .max(setup.minRange.template cast<T>())
                            .min(setup.maxRange.template cast<T>());
                }
            }

            update_fitness_and_heap();
        }

        auto getBestKWolves()
        {
            std::vector<Wolf<T>> best;
            auto copy = heap;
            while (!copy.empty())
            {
                best.push_back(copy.top());
                copy.pop();
            }
            while (best.size() < constants::K)
                best.push_back(best.back());
            return best;
        }

        std::vector<Wolf<T>> population;
        std::priority_queue<Wolf<T>, std::vector<Wolf<T>>, Comparator<T>> heap;
        Eigen::ArrayX<T> nextPos;
        Eigen::ArrayX<T> A;
        Eigen::ArrayX<T> C;
        const Setup setup;
    };
}
#endif
