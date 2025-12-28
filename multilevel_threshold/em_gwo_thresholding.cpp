// em_gwo_thresholding_opencv.cpp
// EM+GWO for multilevel thresholding via GMM histogram fitting
// - Read JPG/PNG with OpenCV
// - Output segmented image
//
// Build (Linux):
//   g++ -O2 -std=c++20 em_gwo_thresholding_opencv.cpp -o em_gwo_seg `pkg-config --cflags --libs opencv4`
//
// Build (Windows, MinGW - example):
//   g++ -O2 -std=c++20 em_gwo_thresholding_opencv.cpp -o em_gwo_seg -IC:\opencv\build\include -LC:\opencv\build\x64\mingw\lib -lopencv_world4xx
// (tùy bạn cài OpenCV kiểu nào)
//
// Run:
//   ./em_gwo_seg input.jpg output.png 4
//     argv1: input image (jpg/png)
//     argv2: output segmented image (png/jpg)
//     argv3: K classes (default 4) => thresholds = K-1

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ----------------------------- Utilities -----------------------------
static inline double clampd(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

static inline double gauss_pdf(double x, double mu, double sigma) {
    const double s = std::max(sigma, 1e-6);
    const double z = (x - mu) / s;
    const double c = 1.0 / (std::sqrt(2.0 * M_PI) * s);
    return c * std::exp(-0.5 * z * z);
}

// ----------------------------- Histogram -----------------------------
static std::array<double, 256> normalized_histogram_u8(const cv::Mat& gray_u8) {
    if (gray_u8.empty() || gray_u8.type() != CV_8UC1)
        throw std::runtime_error("normalized_histogram_u8 requires CV_8UC1 image");

    std::array<double, 256> h{};
    h.fill(0.0);

    const int rows = gray_u8.rows;
    const int cols = gray_u8.cols;

    for (int r = 0; r < rows; ++r) {
        const uint8_t* p = gray_u8.ptr<uint8_t>(r);
        for (int c = 0; c < cols; ++c) {
            h[p[c]] += 1.0;
        }
    }

    const double n = (double)rows * (double)cols;
    if (n <= 0) throw std::runtime_error("Empty image");
    for (double& v : h) v /= n;
    return h;
}

// ----------------------------- GMM Params -----------------------------
struct GMM {
    int K = 0;
    std::vector<double> mu;     // size K
    std::vector<double> sigma;  // size K
    std::vector<double> pi;     // size K
};

// Pack/unpack vector for GWO: [mu0..muK-1, sigma0..sigmaK-1, pi0..piK-2]
static int dim_for_K(int K) { return 3 * K - 1; }

static std::vector<double> pack(const GMM& g) {
    std::vector<double> x;
    x.reserve((size_t)dim_for_K(g.K));
    for (int k = 0; k < g.K; ++k) x.push_back(g.mu[k]);
    for (int k = 0; k < g.K; ++k) x.push_back(g.sigma[k]);
    for (int k = 0; k < g.K - 1; ++k) x.push_back(g.pi[k]);
    return x;
}

static GMM unpack_and_project(const std::vector<double>& x, int K) {
    if ((int)x.size() != dim_for_K(K)) throw std::runtime_error("Bad vector dim");
    GMM g;
    g.K = K;
    g.mu.resize(K);
    g.sigma.resize(K);
    g.pi.resize(K);

    for (int k = 0; k < K; ++k) g.mu[k] = clampd(x[k], 0.0, 255.0);
    for (int k = 0; k < K; ++k) g.sigma[k] = clampd(x[K + k], 1e-3, 255.0);

    double sum = 0.0;
    for (int k = 0; k < K - 1; ++k) {
        g.pi[k] = std::max(1e-6, x[2 * K + k]);
        sum += g.pi[k];
    }
    g.pi[K - 1] = std::max(1e-6, 1.0 - sum);

    double s = 0.0;
    for (double v : g.pi) s += v;
    for (double& v : g.pi) v /= std::max(s, 1e-12);

    // sort by mu
    std::vector<int> idx(K);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return g.mu[a] < g.mu[b]; });

    GMM sorted = g;
    for (int i = 0; i < K; ++i) {
        sorted.mu[i]    = g.mu[idx[i]];
        sorted.sigma[i] = g.sigma[idx[i]];
        sorted.pi[i]    = g.pi[idx[i]];
    }
    return sorted;
}

// ----------------------------- EM on histogram -----------------------------
static double log_likelihood_hist(const std::array<double,256>& h, const GMM& g) {
    double ll = 0.0;
    for (int i = 0; i < 256; ++i) {
        double p = 0.0;
        for (int k = 0; k < g.K; ++k) {
            p += g.pi[k] * gauss_pdf((double)i, g.mu[k], g.sigma[k]);
        }
        p = std::max(p, 1e-300);
        ll += h[i] * std::log(p);
    }
    return ll;
}

static double rmse_hist(const std::array<double,256>& h, const GMM& g) {
    double sse = 0.0;
    for (int i = 0; i < 256; ++i) {
        double p = 0.0;
        for (int k = 0; k < g.K; ++k) {
            p += g.pi[k] * gauss_pdf((double)i, g.mu[k], g.sigma[k]);
        }
        double d = (p - h[i]);
        sse += d * d;
    }
    return std::sqrt(sse / 256.0);
}

static GMM random_init_gmm(int K, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 255.0);

    GMM g;
    g.K = K;
    g.mu.resize(K);
    g.sigma.resize(K);
    g.pi.resize(K);

    for (int k = 0; k < K; ++k) g.mu[k] = U(rng);
    std::sort(g.mu.begin(), g.mu.end());

    for (int k = 0; k < K; ++k) g.sigma[k] = 10.0 + 40.0 * (double)k / std::max(1, K - 1);
    for (int k = 0; k < K; ++k) g.pi[k] = 1.0 / (double)K;
    return g;
}

static void em_one_iter_hist(const std::array<double,256>& h, GMM& g) {
    const int K = g.K;
    if (K > 16) throw std::runtime_error("K too large for this simple buffer (max 16).");

    std::array<std::array<double, 16>, 256> gamma;
    for (int i = 0; i < 256; ++i) {
        double denom = 0.0;
        std::array<double,16> num{};
        num.fill(0.0);
        for (int k = 0; k < K; ++k) {
            num[k] = g.pi[k] * gauss_pdf((double)i, g.mu[k], g.sigma[k]);
            denom += num[k];
        }
        denom = std::max(denom, 1e-300);
        for (int k = 0; k < K; ++k) gamma[i][k] = num[k] / denom;
    }

    std::vector<double> Nk(K, 0.0), mu_new(K, 0.0), var_new(K, 0.0);

    for (int k = 0; k < K; ++k) {
        double sNk = 0.0, s1 = 0.0;
        for (int i = 0; i < 256; ++i) {
            double w = h[i] * gamma[i][k];
            sNk += w;
            s1  += w * (double)i;
        }
        sNk = std::max(sNk, 1e-12);
        Nk[k] = sNk;
        mu_new[k] = s1 / sNk;
    }

    for (int k = 0; k < K; ++k) {
        double s2 = 0.0;
        for (int i = 0; i < 256; ++i) {
            double w = h[i] * gamma[i][k];
            double d = (double)i - mu_new[k];
            s2 += w * d * d;
        }
        var_new[k] = s2 / std::max(Nk[k], 1e-12);
    }

    for (int k = 0; k < K; ++k) {
        g.mu[k] = clampd(mu_new[k], 0.0, 255.0);
        g.sigma[k] = clampd(std::sqrt(std::max(var_new[k], 1e-9)), 1e-3, 255.0);
        g.pi[k] = Nk[k];
    }

    double s = std::accumulate(g.pi.begin(), g.pi.end(), 0.0);
    s = std::max(s, 1e-12);
    for (double& v : g.pi) v /= s;

    std::vector<int> idx(K);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return g.mu[a] < g.mu[b]; });

    GMM sorted = g;
    for (int i = 0; i < K; ++i) {
        sorted.mu[i]    = g.mu[idx[i]];
        sorted.sigma[i] = g.sigma[idx[i]];
        sorted.pi[i]    = g.pi[idx[i]];
    }
    g = sorted;
}

// ----------------------------- GWO (minimization) -----------------------------
struct Wolf {
    std::vector<double> x;
    double fit = 0.0;
};

struct GWOConfig {
    int pop = 30;
    int iters = 50;
    std::vector<double> lo, hi;
    uint64_t seed = 1234567;
};

template <class FitnessFn>
static std::vector<double> gwo_minimize(const GWOConfig& cfg, FitnessFn fitness) {
    std::mt19937_64 rng(cfg.seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    const int D = (int)cfg.lo.size();
    std::vector<Wolf> wolves(cfg.pop);

    auto rand_in_range = [&](int d) {
        return cfg.lo[d] + (cfg.hi[d] - cfg.lo[d]) * U01(rng);
    };

    for (int i = 0; i < cfg.pop; ++i) {
        wolves[i].x.resize(D);
        for (int d = 0; d < D; ++d) wolves[i].x[d] = rand_in_range(d);
        wolves[i].fit = fitness(wolves[i].x);
    }

    auto select_top3 = [&]() {
        std::array<int,3> idx = {0,1,2};
        auto better = [&](int a, int b){ return wolves[a].fit < wolves[b].fit; };
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return better(a,b); });

        for (int i = 3; i < cfg.pop; ++i) {
            if (better(i, idx[2])) {
                idx[2] = i;
                std::sort(idx.begin(), idx.end(), [&](int a, int b){ return better(a,b); });
            }
        }
        return idx;
    };

    for (int t = 0; t < cfg.iters; ++t) {
        auto top3 = select_top3();
        const auto alpha_x = wolves[top3[0]].x;
        const auto beta_x  = wolves[top3[1]].x;
        const auto delta_x = wolves[top3[2]].x;

        const double a = 2.0 * (1.0 - (double)t / (double)std::max(1, cfg.iters - 1));

        for (int i = 0; i < cfg.pop; ++i) {
            for (int d = 0; d < D; ++d) {
                double r1 = U01(rng), r2 = U01(rng);
                double A1 = 2.0 * a * r1 - a;
                double C1 = 2.0 * r2;

                r1 = U01(rng); r2 = U01(rng);
                double A2 = 2.0 * a * r1 - a;
                double C2 = 2.0 * r2;

                r1 = U01(rng); r2 = U01(rng);
                double A3 = 2.0 * a * r1 - a;
                double C3 = 2.0 * r2;

                double D_alpha = std::fabs(C1 * alpha_x[d] - wolves[i].x[d]);
                double D_beta  = std::fabs(C2 * beta_x[d]  - wolves[i].x[d]);
                double D_delta = std::fabs(C3 * delta_x[d] - wolves[i].x[d]);

                double X1 = alpha_x[d] - A1 * D_alpha;
                double X2 = beta_x[d]  - A2 * D_beta;
                double X3 = delta_x[d] - A3 * D_delta;

                double xnew = (X1 + X2 + X3) / 3.0;
                wolves[i].x[d] = clampd(xnew, cfg.lo[d], cfg.hi[d]);
            }
            wolves[i].fit = fitness(wolves[i].x);
        }
    }

    int best = 0;
    for (int i = 1; i < cfg.pop; ++i) if (wolves[i].fit < wolves[best].fit) best = i;
    return wolves[best].x;
}

// ----------------------------- EM + GWO hybrid -----------------------------
static GMM em_fit_hist_hybrid(const std::array<double,256>& h,
                             int K,
                             int em_max_iter,
                             double ll_limit,
                             int stagnancy_limit,
                             int gwo_interval,
                             int gwo_iters_suggest,
                             int gwo_pop_suggest,
                             uint64_t seed) {
    GMM g = random_init_gmm(K, seed);
    double ll_prev = log_likelihood_hist(h, g);
    int stagnancy = 0;

    const int D = dim_for_K(K);
    std::vector<double> lo(D), hi(D);
    for (int k = 0; k < K; ++k) { lo[k] = 0.0; hi[k] = 255.0; }
    for (int k = 0; k < K; ++k) { lo[K + k] = 1e-3; hi[K + k] = 255.0; }
    for (int k = 0; k < K - 1; ++k) { lo[2*K + k] = 1e-6; hi[2*K + k] = 1.0; }

    for (int iter = 1; iter <= em_max_iter; ++iter) {
        em_one_iter_hist(h, g);
        double ll = log_likelihood_hist(h, g);

        double dll = std::fabs(ll - ll_prev);
        if (dll > ll_limit) stagnancy = 0;
        else stagnancy++;

        bool trigger = (stagnancy >= stagnancy_limit) || (gwo_interval > 0 && (iter % gwo_interval == 0));
        if (trigger) {
            GWOConfig cfg;
            cfg.pop = gwo_pop_suggest;
            cfg.iters = gwo_iters_suggest;
            cfg.lo = lo;
            cfg.hi = hi;
            cfg.seed = seed + 1000ull + (uint64_t)iter;

            auto bestx = gwo_minimize(cfg, [&](const std::vector<double>& x){
                GMM gg = unpack_and_project(x, K);
                return -log_likelihood_hist(h, gg); // maximize LL
            });

            GMM g_suggest = unpack_and_project(bestx, K);
            double ll_suggest = log_likelihood_hist(h, g_suggest);

            if (ll_suggest > ll) {
                g = g_suggest;
                ll = ll_suggest;
                stagnancy = 0;
            }
        }

        ll_prev = ll;
    }

    return g;
}

// ----------------------------- Thresholds from GMM -----------------------------
static double gaussian_intersection(double m1, double s1, double p1,
                                    double m2, double s2, double p2) {
    s1 = std::max(s1, 1e-6);
    s2 = std::max(s2, 1e-6);
    p1 = std::max(p1, 1e-12);
    p2 = std::max(p2, 1e-12);

    const double A = 1.0/(2*s2*s2) - 1.0/(2*s1*s1);
    const double B = m1/(s1*s1) - m2/(s2*s2);
    const double C = (m2*m2)/(2*s2*s2) - (m1*m1)/(2*s1*s1) + std::log((p2*s1)/(p1*s2));

    if (std::fabs(A) < 1e-12) {
        if (std::fabs(B) < 1e-12) return (m1 + m2) / 2.0;
        return -C / B;
    }

    const double disc = B*B - 4*A*C;
    if (disc < 0.0) return (m1 + m2) / 2.0;

    const double r1 = (-B + std::sqrt(disc)) / (2*A);
    const double r2 = (-B - std::sqrt(disc)) / (2*A);

    const double lo = std::min(m1, m2), hi = std::max(m1, m2);
    const bool in1 = (r1 >= lo && r1 <= hi);
    const bool in2 = (r2 >= lo && r2 <= hi);

    if (in1 && in2) {
        double mid = 0.5*(m1+m2);
        return (std::fabs(r1-mid) < std::fabs(r2-mid)) ? r1 : r2;
    }
    if (in1) return r1;
    if (in2) return r2;

    double mid = 0.5*(m1+m2);
    return (std::fabs(r1-mid) < std::fabs(r2-mid)) ? r1 : r2;
}

static std::vector<int> thresholds_from_gmm(const GMM& g) {
    std::vector<int> t;
    t.reserve((size_t)g.K - 1);
    for (int k = 0; k < g.K - 1; ++k) {
        double x = gaussian_intersection(g.mu[k], g.sigma[k], g.pi[k],
                                         g.mu[k+1], g.sigma[k+1], g.pi[k+1]);
        int ti = (int)std::lround(clampd(x, 0.0, 255.0));
        t.push_back(ti);
    }
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    return t;
}

// Assign class based on thresholds (K classes => K-1 thresholds).
static inline int class_of_pixel(uint8_t v, const std::vector<int>& thr) {
    // thr sorted
    int c = 0;
    while (c < (int)thr.size() && v > (uint8_t)thr[c]) c++;
    return c; // 0..K-1
}

// Create segmented grayscale image: map class -> output intensity.
static cv::Mat make_segmented_image(const cv::Mat& gray_u8,
                                    const std::vector<int>& thr,
                                    int K) {
    cv::Mat out(gray_u8.size(), CV_8UC1);

    // map K classes to equally spaced intensities
    std::vector<uint8_t> levels(K);
    if (K == 1) levels[0] = 0;
    else {
        for (int c = 0; c < K; ++c) {
            levels[c] = (uint8_t)std::lround(255.0 * (double)c / (double)(K - 1));
        }
    }

    for (int r = 0; r < gray_u8.rows; ++r) {
        const uint8_t* p = gray_u8.ptr<uint8_t>(r);
        uint8_t* q = out.ptr<uint8_t>(r);
        for (int c = 0; c < gray_u8.cols; ++c) {
            int cls = class_of_pixel(p[c], thr);
            cls = std::max(0, std::min(cls, K - 1));
            q[c] = levels[cls];
        }
    }
    return out;
}

int main(int argc, char** argv) {
    try {
        std::string in_path  = (argc >= 2) ? argv[1] : "input.jpg";
        std::string out_path = (argc >= 3) ? argv[2] : "segmented.png";
        int K = (argc >= 4) ? std::stoi(argv[3]) : 4;
        if (K < 2 || K > 16) throw std::runtime_error("K must be in [2,16]");

        // ---- Load image with OpenCV ----
        cv::Mat bgr = cv::imread(in_path, cv::IMREAD_COLOR);
        if (bgr.empty()) throw std::runtime_error("Cannot read image: " + in_path);

        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY); // CV_8UC1

        auto h = normalized_histogram_u8(gray);

        // ---- Parameters (you can align to paper) ----
        const int EM_MAX_ITER = 100;
        const double LL_LIMIT = 1e-6;
        const int STAGNANCY_LIMIT = 20;
        const int GWO_INTERVAL = 10;

        const int GWO_POP_SUGGEST   = 20;
        const int GWO_ITERS_SUGGEST = 30;

        const int GWO_POP_TUNE   = 30;
        const int GWO_ITERS_TUNE = 100;

        const uint64_t SEED = 123456789ULL;

        std::cout << "Input: " << in_path << " (" << gray.cols << "x" << gray.rows << ")\n";
        std::cout << "K=" << K << " (thresholds=" << (K-1) << ")\n";

        // ---- Step 1: EM + periodic GWO suggestions (LL-based) ----
        GMM g_em = em_fit_hist_hybrid(h, K,
                                     EM_MAX_ITER,
                                     LL_LIMIT,
                                     STAGNANCY_LIMIT,
                                     GWO_INTERVAL,
                                     GWO_ITERS_SUGGEST,
                                     GWO_POP_SUGGEST,
                                     SEED);

        double ll_em = log_likelihood_hist(h, g_em);
        double rmse_em = rmse_hist(h, g_em);

        std::cout << "After EM(+GWO suggest): LL=" << ll_em << " RMSE=" << rmse_em << "\n";

        // ---- Step 2: final tuning by GWO minimizing RMSE ----
        const int D = dim_for_K(K);
        std::vector<double> lo(D), hi(D);
        for (int k = 0; k < K; ++k) { lo[k] = 0.0; hi[k] = 255.0; }
        for (int k = 0; k < K; ++k) { lo[K + k] = 1e-3; hi[K + k] = 255.0; }
        for (int k = 0; k < K - 1; ++k) { lo[2*K + k] = 1e-6; hi[2*K + k] = 1.0; }

        std::vector<double> x_em = pack(g_em);

        // shrink bounds around EM solution for more stable tuning
        std::vector<double> lo2 = lo, hi2 = hi;
        for (int k = 0; k < K; ++k) {
            lo2[k] = clampd(x_em[k] - 30.0, 0.0, 255.0);
            hi2[k] = clampd(x_em[k] + 30.0, 0.0, 255.0);
        }
        for (int k = 0; k < K; ++k) {
            double s = x_em[K + k];
            lo2[K + k] = clampd(s * 0.5, 1e-3, 255.0);
            hi2[K + k] = clampd(s * 1.5, 1e-3, 255.0);
        }
        for (int k = 0; k < K - 1; ++k) {
            double p = x_em[2*K + k];
            lo2[2*K + k] = clampd(p - 0.2, 1e-6, 1.0);
            hi2[2*K + k] = clampd(p + 0.2, 1e-6, 1.0);
        }

        GWOConfig cfg_tune;
        cfg_tune.pop = GWO_POP_TUNE;
        cfg_tune.iters = GWO_ITERS_TUNE;
        cfg_tune.lo = lo2;
        cfg_tune.hi = hi2;
        cfg_tune.seed = SEED + 9999;

        auto bestx = gwo_minimize(cfg_tune, [&](const std::vector<double>& x){
            GMM gg = unpack_and_project(x, K);
            return rmse_hist(h, gg);
        });

        GMM g_final = unpack_and_project(bestx, K);
        double ll_final = log_likelihood_hist(h, g_final);
        double rmse_final = rmse_hist(h, g_final);

        std::cout << "After RMSE tune (GWO):   LL=" << ll_final << " RMSE=" << rmse_final << "\n";

        auto thr = thresholds_from_gmm(g_final);
        std::cout << "Thresholds: ";
        for (size_t i = 0; i < thr.size(); ++i)
            std::cout << thr[i] << (i + 1 < thr.size() ? ", " : "\n");

        // ---- Make segmented image and write ----
        cv::Mat seg = make_segmented_image(gray, thr, K);
        if (!cv::imwrite(out_path, seg))
            throw std::runtime_error("Failed to write output: " + out_path);

        std::cout << "Wrote segmented image: " << out_path << "\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        std::cerr << "Usage: ./em_gwo_seg input.jpg output.png [K]\n";
        return 1;
    }
}
