#ifndef GWO_HPP
#define GWO_HPP
#include <iostream>
#include <random>
#include <vector>
#include <concepts>
#include <queue>
#include <Eigen/Dense>
#include <stdexcept>

namespace GWO
{
    struct XorShift64
    {
        uint64_t state;
        uint64_t next()
        {
            uint64_t x = state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            return state = x;
        }
    };
    std::random_device rd_seed;
    XorShift64 rng{rd_seed()};

    template <std::floating_point T>
    T random(T min, T max)
    {
        double zero_to_one = (rng.next() >> 11) * 0x1.0p-53;
        return min + zero_to_one * (max - min);
    }
    namespace constants
    {
        size_t K = 3;
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
        Wolf(size_t n) : pos(n), len(n) {}
        void randomize(const Eigen::ArrayXd &min, const Eigen::ArrayXd &max)
        {
            for (size_t i = 0; i < len; i++)
            {
                pos[i] = random(min[i], max[i]);
            }
        }
        T savedFitness{};
        Eigen::ArrayX<T> pos;
        size_t len{};
    };
    template <std::floating_point T>
    std::ostream &operator<<(std::ostream &os, const Wolf<T> &wolf)
    {
        os << "[";
        for (size_t i = 0; i < wolf.len - 1; i++)
        {
            os << wolf.pos[i] << ",";
        }
        os << wolf.pos[wolf.len - 1] << "]";
        return os;
    }
    template <std::floating_point T>
    class Comparator
    {
    public:
        bool operator()(const Wolf<T> &w1, const Wolf<T> &w2)
        {
            return w1.savedFitness < w2.savedFitness;
        }
    };
    template <std::floating_point T>
    struct Problem
    {
        virtual Eigen::ArrayX<T> fitness_batch(const Eigen::ArrayXX<T> &population_pos) const
        {
            Eigen::ArrayX<T> fitness_values(population_pos.rows());
            for (int i = 0; i < population_pos.rows(); ++i)
            {
                fitness_values(i) = this->fitness(population_pos.row(i));
            }
            return fitness_values;
        }

        virtual T fitness(const Eigen::ArrayX<T> &pos) const
        {
            throw std::runtime_error("fitness() not implemented. Did you mean to implement fitness_batch()?");
        };

        Problem(Setup _setup) : nextPos(_setup.N),
                                A(_setup.N), C(_setup.N), setup(std::move(_setup))
        {
            if (setup.N == 0 || setup.POP_SIZE == 0)
                throw std::invalid_argument("N and POP_SIZE must be > 0.");
            if (setup.minRange.size() != setup.N || setup.maxRange.size() != setup.N)
            {
                throw std::invalid_argument("minRange and maxRange must have size N.");
            }
            if ((setup.maxRange < setup.minRange).any())
            {
                throw std::invalid_argument("All elements of maxRange must be >= minRange.");
            }

            for (size_t i = 0; i < setup.POP_SIZE; i++)
            {
                population.emplace_back(setup.N);
                population.back().randomize(setup.minRange, setup.maxRange);
            }
        }

        Wolf<T> run(int maxIterations)
        {
            update_fitness_and_heap();
            for (int i = 0; i < maxIterations; i++)
            {
                T a = 2 * (1 - T(i) / T(maxIterations));
                updatePopulation(a);
            }
            return getBestKWolves()[0];
        }

        void update_fitness_and_heap()
        {
            Eigen::ArrayXX<T> positions(setup.POP_SIZE, setup.N);
            for (size_t i = 0; i < setup.POP_SIZE; ++i)
            {
                positions.row(i) = population[i].pos;
            }
            Eigen::ArrayX<T> fitness_values = this->fitness_batch(positions);
            heap = {};
            for (size_t i = 0; i < setup.POP_SIZE; ++i)
            {
                population[i].savedFitness = fitness_values(i);
                heap.push(population[i]);
                if (heap.size() > constants::K)
                {
                    heap.pop();
                }
            }
        }

        void updatePopulation(T a)
        {
            auto bestWolves = getBestKWolves();
            for (auto &wolf : population)
            {
                nextPos.setZero();
                for (size_t j = 0; j < constants::K; j++)
                {
                    for (size_t k = 0; k < setup.N; k++)
                    {
                        A[k] = 2 * a * random(0.0, 1.0) - a;
                        C[k] = 2 * random(0.0, 1.0);
                    }
                    auto &bestWolf = bestWolves[j];
                    auto D = (bestWolf.pos * C - wolf.pos).abs();
                    nextPos += bestWolf.pos - D * A;
                }

                wolf.pos = (nextPos / T(constants::K))
                               .max(setup.minRange.template cast<T>())
                               .min(setup.maxRange.template cast<T>());
            }
            update_fitness_and_heap();
        }
        auto getBestKWolves()
        {
            std::vector<Wolf<T>> bestWolves;
            auto copy = heap;
            while (!copy.empty())
            {
                bestWolves.push_back(std::move(copy.top()));
                copy.pop();
            }
            return bestWolves;
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