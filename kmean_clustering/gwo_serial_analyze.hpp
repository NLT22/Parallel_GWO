// gwo_serial_analyze.hpp
#ifndef GWO_SERIAL_SYNC_HPP
#define GWO_SERIAL_SYNC_HPP

#include <iostream>
#include <vector>
#include <concepts>
#include <queue>
#include <Eigen/Dense>
#include <stdexcept>
#include <cstdint>
#include <chrono>

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

    // ===================== PROFILER (EXCLUSIVE times) =====================
    struct Profiler
    {
        // Exclusive times in ms (nested removed)
        static inline uint64_t t_update_fitness_excl_ms = 0;
        static inline uint64_t t_update_pop_excl_ms     = 0;
        static inline uint64_t t_fitness_batch_excl_ms  = 0;

        // Leaf time (no children we measure under it)
        static inline uint64_t t_fitness_scalar_ms      = 0;

        static inline uint64_t n_fitness_calls          = 0;

        static inline void reset()
        {
            t_update_fitness_excl_ms = 0;
            t_update_pop_excl_ms     = 0;
            t_fitness_batch_excl_ms  = 0;
            t_fitness_scalar_ms      = 0;
            n_fitness_calls          = 0;
        }

        static inline uint64_t ms_since(const std::chrono::steady_clock::time_point& a,
                                        const std::chrono::steady_clock::time_point& b)
        {
            return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
        }
    };

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
        int id;
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
            return w1.id < w2.id;
        }
    };

    template <std::floating_point T>
    struct Problem
    {
        // Fitness batch: exclusive = total - scalar_children
        virtual Eigen::ArrayX<T> fitness_batch(const Eigen::ArrayXX<T> &population_pos) const
        {
            using clock = std::chrono::steady_clock;
            auto t0 = clock::now();

            // snapshot before scalar loop
            uint64_t scalar_before = Profiler::t_fitness_scalar_ms;

            Eigen::ArrayX<T> fitness_values(population_pos.rows());
            for (int i = 0; i < population_pos.rows(); ++i)
            {
                auto s0 = clock::now();
                fitness_values(i) = this->fitness(population_pos.row(i));
                auto s1 = clock::now();

                Profiler::t_fitness_scalar_ms += Profiler::ms_since(s0, s1);
                Profiler::n_fitness_calls += 1;
            }

            uint64_t scalar_delta = Profiler::t_fitness_scalar_ms - scalar_before;

            auto t1 = clock::now();
            uint64_t total = Profiler::ms_since(t0, t1);

            // exclusive: remove scalar time
            Profiler::t_fitness_batch_excl_ms += (total > scalar_delta) ? (total - scalar_delta) : 0ULL;

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

        // update_fitness_and_heap: exclusive = total - fitness_batch_total (child)
        void update_fitness_and_heap()
        {
            using clock = std::chrono::steady_clock;
            auto t0 = clock::now();

            // snapshot child-related counters to compute delta
            uint64_t scalar_before = Profiler::t_fitness_scalar_ms;
            uint64_t batch_excl_before = Profiler::t_fitness_batch_excl_ms;

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

            auto t1 = clock::now();
            uint64_t total = Profiler::ms_since(t0, t1);

            // child total = scalar_delta + batch_excl_delta
            uint64_t scalar_delta = Profiler::t_fitness_scalar_ms - scalar_before;
            uint64_t batch_excl_delta = Profiler::t_fitness_batch_excl_ms - batch_excl_before;
            uint64_t child_total = scalar_delta + batch_excl_delta;

            Profiler::t_update_fitness_excl_ms += (total > child_total) ? (total - child_total) : 0ULL;
        }

        // updatePopulation: exclusive = total - update_fitness_and_heap_total (child)
        void updatePopulation(int iter, T a)
        {
            using clock = std::chrono::steady_clock;
            auto t0 = clock::now();

            auto bestWolves = getBestKWolves();

            for (int wi = 0; wi < (int)setup.POP_SIZE; ++wi)
            {
                nextPos.setZero();

                for (size_t j = 0; j < constants::K; j++)
                {
                    for (int k = 0; k < (int)setup.N; k++)
                    {
                        double r1 = indexed_random(iter, wi, k, (int)(j * 2));
                        double r2 = indexed_random(iter, wi, k, (int)(j * 2 + 1));
                        A(k) = 2 * a * r1 - a;
                        C(k) = 2 * r2;
                    }

                    auto D = (bestWolves[j].pos * C - population[wi].pos).abs();
                    nextPos += bestWolves[j].pos - D * A;
                }

                population[wi].pos =
                    (nextPos / T(constants::K))
                        .max(setup.minRange.template cast<T>())
                        .min(setup.maxRange.template cast<T>());
            }

            // snapshot to compute child's total time
            uint64_t uf_before = Profiler::t_update_fitness_excl_ms;
            uint64_t batch_before = Profiler::t_fitness_batch_excl_ms;
            uint64_t scalar_before = Profiler::t_fitness_scalar_ms;

            update_fitness_and_heap();

            uint64_t uf_delta = Profiler::t_update_fitness_excl_ms - uf_before;
            uint64_t batch_delta = Profiler::t_fitness_batch_excl_ms - batch_before;
            uint64_t scalar_delta = Profiler::t_fitness_scalar_ms - scalar_before;

            // child's total = update_fitness_excl + fitness_batch_excl + scalar
            uint64_t child_total = uf_delta + batch_delta + scalar_delta;

            auto t1 = clock::now();
            uint64_t total = Profiler::ms_since(t0, t1);

            Profiler::t_update_pop_excl_ms += (total > child_total) ? (total - child_total) : 0ULL;
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
