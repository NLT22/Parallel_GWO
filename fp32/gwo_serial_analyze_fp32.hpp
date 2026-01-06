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
#include <algorithm>

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

    // ===================== PROFILER (EXCLUSIVE times) =====================
    struct Profiler
    {
        static inline uint64_t t_update_fitness_excl_ms = 0;
        static inline uint64_t t_update_pop_excl_ms     = 0;
        static inline uint64_t t_fitness_batch_excl_ms  = 0;
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

    // Better = smaller fitness, tie by smaller id
    template <std::floating_point T>
    static inline bool better_min(const Wolf<T>& a, const Wolf<T>& b)
    {
        if (a.savedFitness != b.savedFitness) return a.savedFitness < b.savedFitness;
        return a.id < b.id;
    }

    // For std::priority_queue: we want TOP = WORST among kept (max by better_min)
    template <std::floating_point T>
    struct WorseFirst
    {
        bool operator()(const Wolf<T>& a, const Wolf<T>& b) const
        {
            // return true if a is BETTER than b -> then a has lower priority, so top becomes worst
            return better_min(a, b);
        }
    };

    template <std::floating_point T>
    struct Problem
    {
        virtual Eigen::ArrayX<T> fitness_batch(const Eigen::ArrayXX<T> &population_pos) const
        {
            using clock = std::chrono::steady_clock;
            auto t0 = clock::now();

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
            // return alpha (best)
            auto leaders = getBest3WolvesSorted();
            return leaders[0];
        }

        // update_fitness_and_heap: exclusive = total - fitness_batch_total (child)
        void update_fitness_and_heap()
        {
            using clock = std::chrono::steady_clock;
            auto t0 = clock::now();

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
                    heap.pop(); // pop WORST among kept
            }

            auto t1 = clock::now();
            uint64_t total = Profiler::ms_since(t0, t1);

            uint64_t scalar_delta = Profiler::t_fitness_scalar_ms - scalar_before;
            uint64_t batch_excl_delta = Profiler::t_fitness_batch_excl_ms - batch_excl_before;
            uint64_t child_total = scalar_delta + batch_excl_delta;

            Profiler::t_update_fitness_excl_ms += (total > child_total) ? (total - child_total) : 0ULL;
        }

        void updatePopulation(int iter, T a)
        {
            using clock = std::chrono::steady_clock;
            auto t0 = clock::now();

            auto leaders = getBest3WolvesSorted(); // [alpha,beta,delta]

            for (int wi = 0; wi < (int)setup.POP_SIZE; ++wi)
            {
                nextPos.setZero();

                for (size_t j = 0; j < constants::K; j++)
                {
                    for (int k = 0; k < (int)setup.N; k++)
                    {
                        float r1 = indexed_random(iter, wi, k, (int)(j * 2));
                        float r2 = indexed_random(iter, wi, k, (int)(j * 2 + 1));
                        A(k) = 2 * a * r1 - a;
                        C(k) = 2 * r2;
                    }

                    auto D = (leaders[j].pos * C - population[wi].pos).abs();
                    nextPos += leaders[j].pos - D * A;
                }

                population[wi].pos =
                    (nextPos / T(constants::K))
                        .max(setup.minRange.template cast<T>())
                        .min(setup.maxRange.template cast<T>());
            }

            uint64_t uf_before = Profiler::t_update_fitness_excl_ms;
            uint64_t batch_before = Profiler::t_fitness_batch_excl_ms;
            uint64_t scalar_before = Profiler::t_fitness_scalar_ms;

            update_fitness_and_heap();

            uint64_t uf_delta = Profiler::t_update_fitness_excl_ms - uf_before;
            uint64_t batch_delta = Profiler::t_fitness_batch_excl_ms - batch_before;
            uint64_t scalar_delta = Profiler::t_fitness_scalar_ms - scalar_before;

            uint64_t child_total = uf_delta + batch_delta + scalar_delta;

            auto t1 = clock::now();
            uint64_t total = Profiler::ms_since(t0, t1);

            Profiler::t_update_pop_excl_ms += (total > child_total) ? (total - child_total) : 0ULL;
        }

        // Return [alpha,beta,delta] sorted by (fitness asc, id asc)
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
