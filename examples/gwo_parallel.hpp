#ifndef GWO_PARALLEL_HPP
#define GWO_PARALLEL_HPP

#include <iostream>
#include <vector>
#include <queue>
#include <stdexcept>
#include <Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
//  DETERMINISTIC RANDOM (KHÔNG PHỤ THUỘC THREAD SCHEDULING)
// ============================================================================

namespace GWO {

inline uint64_t global_seed = 123456789ULL;

// SplitMix64 hash
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

// random ∈ [0..1]
inline double indexed_random(int iter, int wid, int dim, int channel) {
    uint64_t mix =
        (uint64_t)iter * 0x9E3779B97F4A7C15ULL ^
        (uint64_t)wid  * 0xBF58476D1CE4E5B9ULL ^
        (uint64_t)dim  * 0x94D049BB133111EBULL ^
        (uint64_t)channel * 0x123456789ABCDEFULL ^
        GWO::global_seed;

    uint64_t r = splitmix64(mix);
    return (r >> 11) * 0x1.0p-53;   // double 0..1
}

// ============================================================================
//  STRUCT CƠ BẢN
// ============================================================================

namespace constants {
    inline size_t K = 3; // Alpha, Beta, Delta
}

struct Setup {
    size_t N{};
    size_t POP_SIZE{};
    Eigen::ArrayXd maxRange;
    Eigen::ArrayXd minRange;
};

template <typename T>
struct Wolf {
    Wolf(size_t n) : pos(n), len(n) {}
    T savedFitness{};
    Eigen::ArrayX<T> pos;
    size_t len{};
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Wolf<T> &w) {
    os << "[";
    for (size_t i = 0; i < w.len - 1; i++) os << w.pos[i] << ",";
    os << w.pos[w.len - 1] << "]";
    return os;
}

template <typename T>
struct Comparator {
    bool operator()(const Wolf<T> &a, const Wolf<T> &b) const {
        return a.savedFitness < b.savedFitness;
    }
};

// ============================================================================
//  ABSTRACT PROBLEM
// ============================================================================

template <typename T>
class Problem {
public:
    using wolf_t = Wolf<T>;

    Problem(const Setup &setup)
        : N(setup.N), POP_SIZE(setup.POP_SIZE),
          maxRange(setup.maxRange), minRange(setup.minRange)
    {
        if (N == 0 || POP_SIZE == 0)
            throw std::invalid_argument("N and POP_SIZE must be > 0");

        population.reserve(POP_SIZE);

        for (size_t i = 0; i < POP_SIZE; i++) {
            population.emplace_back(N);
            for (size_t k = 0; k < N; k++)
                population.back().pos[k] =
                    minRange[k] + indexed_random(0, i, k, 99) * (maxRange[k] - minRange[k]);
        }
    }

    virtual T fitness(const Eigen::ArrayX<T> &pos) const = 0;

    Eigen::ArrayX<T> fitness_batch(const Eigen::ArrayXX<T> &mat) const {
        Eigen::ArrayX<T> out(mat.rows());

        #pragma omp parallel for
        for (int i = 0; i < (int)mat.rows(); i++)
            out(i) = fitness(mat.row(i));

        return out;
    }

    // ========================================================================
    //  RUN GWO
    // ========================================================================
    wolf_t run(int maxIter) {
        update_fitness_and_heap();

        for (int iter = 0; iter < maxIter; iter++) {
            double a = 2.0 * (1.0 - (double)iter / maxIter);
            update_population(iter, a);
        }

        return get_best_wolves()[0];
    }

private:

    // ========================================================================
    //  UPDATE POPULATION (song song)
    // ========================================================================
    void update_population(int iter, double a) {
        auto best = get_best_wolves();

        #pragma omp parallel
        {
            Eigen::ArrayX<double> A(N), C(N), next(N);

            #pragma omp for
            for (int i = 0; i < (int)POP_SIZE; i++) {

                next.setZero();

                for (size_t bw = 0; bw < constants::K; bw++) {

                    for (size_t k = 0; k < N; k++) {
                        double r1 = indexed_random(iter, i, k, bw*2);
                        double r2 = indexed_random(iter, i, k, bw*2 + 1);

                        A[k] = 2 * a * r1 - a;
                        C[k] = 2 * r2;
                    }

                    auto D = (best[bw].pos * C - population[i].pos).abs();
                    next += best[bw].pos - D * A;
                }

                population[i].pos =
                    (next / (double)constants::K)
                        .max(minRange)
                        .min(maxRange);
            }
        }

        update_fitness_and_heap();
    }

    // ========================================================================
    //  UPDATE FITNESS + HEAP
    // ========================================================================
    void update_fitness_and_heap() {
        Eigen::ArrayXX<T> P(POP_SIZE, N);
        for (size_t i = 0; i < POP_SIZE; i++)
            P.row(i) = population[i].pos;

        auto fit = fitness_batch(P);

        heap = {};
        for (size_t i = 0; i < POP_SIZE; i++) {
            population[i].savedFitness = fit(i);
            heap.push(population[i]);
            if (heap.size() > constants::K)
                heap.pop();
        }
    }

    // ========================================================================
    //  ALWAYS RETURN EXACTLY 3 BEST WOLVES
    // ========================================================================
    std::vector<wolf_t> get_best_wolves() const {
        auto h = heap;
        std::vector<wolf_t> out;

        while (!h.empty() && out.size() < constants::K) {
            out.push_back(h.top());
            h.pop();
        }

        // Nếu thiếu thì replicate cho đủ 3
        while (out.size() < constants::K) {
            out.push_back(out.back());
        }

        return out;
    }

public:
    size_t N, POP_SIZE;
    Eigen::ArrayXd maxRange, minRange;

    std::vector<wolf_t> population;
    std::priority_queue<wolf_t, std::vector<wolf_t>, Comparator<T>> heap;
};

} // namespace GWO

#endif
