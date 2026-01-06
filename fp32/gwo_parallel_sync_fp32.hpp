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
#include <algorithm>

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

    inline float indexed_random(int iter, int wid, int dim, int channel)
    {
        uint64_t mix =
            (uint64_t)iter * 0x9E3779B97F4A7C15ULL ^
            (uint64_t)wid  * 0xBF58476D1CE4E5B9ULL ^
            (uint64_t)dim  * 0x94D049BB133111EBULL ^
            (uint64_t)channel * 0x123456789ABCDEFULL ^
            global_seed;

        uint64_t r = splitmix64(mix);
        
        uint32_t u = (uint32_t)(r >> 40);
        return (float)u * (1.0f / 16777216.0f);
    }

    namespace constants
    {
        inline size_t K = 3;
    }

    struct Setup
    {
        size_t N{};
        size_t POP_SIZE{};
        Eigen::ArrayXf maxRange;
        Eigen::ArrayXf minRange;
    };

    template <std::floating_point T>
    struct Wolf
    {
        Wolf(size_t n) : id(-1), savedFitness{}, pos(n), len(n) {}
        int id;
        T savedFitness{};
        Eigen::ArrayX<T> pos;
        size_t len{};
    };

    template <std::floating_point T>
    static inline bool better_min(const Wolf<T>& a, const Wolf<T>& b)
    {
        if (a.savedFitness != b.savedFitness) return a.savedFitness < b.savedFitness;
        return a.id < b.id;
    }

    template <std::floating_point T>
    struct WorseFirst
    {
        bool operator()(const Wolf<T>& a, const Wolf<T>& b) const
        {
            return better_min(a, b); // top becomes worst
        }
    };

    template <std::floating_point T>
    struct Problem
    {
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
                    float r = indexed_random(0, (int)i, (int)k, 77);
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
            auto leaders = getBest3WolvesSorted();
            return leaders[0]; // alpha
        }

        void update_fitness_and_heap()
        {
            Eigen::ArrayX<T> fitness_values((int)setup.POP_SIZE);

            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int i = 0; i < (int)setup.POP_SIZE; ++i)
                fitness_values(i) = this->fitness(population[(size_t)i].pos);

            heap = {};
            for (size_t i = 0; i < setup.POP_SIZE; ++i)
            {
                population[i].savedFitness = fitness_values((int)i);
                heap.push(population[i]);
                if (heap.size() > constants::K) heap.pop(); // pop worst among kept
            }
        }

        void updatePopulation(int iter, T a)
        {
            auto leaders = getBest3WolvesSorted(); // [alpha,beta,delta]

            #ifdef _OPENMP
            #pragma omp parallel
            #endif
            {
                Eigen::ArrayX<T> localA((int)setup.N);
                Eigen::ArrayX<T> localC((int)setup.N);
                Eigen::ArrayX<T> localNext((int)setup.N);

                #ifdef _OPENMP
                #pragma omp for schedule(static)
                #endif
                for (int i = 0; i < (int)setup.POP_SIZE; ++i)
                {
                    localNext.setZero();

                    for (size_t j = 0; j < constants::K; j++)
                    {
                        for (int k = 0; k < (int)setup.N; k++)
                        {
                            float r1 = indexed_random(iter, i, k, (int)(j * 2));
                            float r2 = indexed_random(iter, i, k, (int)(j * 2 + 1));
                            localA(k) = 2 * a * r1 - a;
                            localC(k) = 2 * r2;
                        }

                        auto D = (leaders[j].pos * localC - population[(size_t)i].pos).abs();
                        localNext += leaders[j].pos - D * localA;
                    }

                    population[(size_t)i].pos =
                        (localNext / T(constants::K))
                            .max(setup.minRange.template cast<T>())
                            .min(setup.maxRange.template cast<T>());
                }
            }

            update_fitness_and_heap();
        }

        std::vector<Wolf<T>> getBest3WolvesSorted()
        {
            std::vector<Wolf<T>> best;
            auto copy = heap;
            while (!copy.empty())
            {
                best.push_back(copy.top());
                copy.pop();
            }

            if (best.empty()) throw std::runtime_error("Heap empty");
            while (best.size() < constants::K) best.push_back(best.back());

            std::sort(best.begin(), best.end(), [](const Wolf<T>& a, const Wolf<T>& b){
                return better_min(a, b);
            });

            if (best.size() > constants::K)
                best.erase(best.begin() + (std::ptrdiff_t)constants::K, best.end());

            return best;
        }

        std::vector<Wolf<T>> population;
        std::priority_queue<Wolf<T>, std::vector<Wolf<T>>, WorseFirst<T>> heap;

        Eigen::ArrayX<T> nextPos;
        Eigen::ArrayX<T> A;
        Eigen::ArrayX<T> C;
        const Setup setup;
    };
}

#endif
