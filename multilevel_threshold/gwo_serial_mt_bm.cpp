// gwo_serial_mt_bm.cpp
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp>
#include "../benchmark/openmp/gwo_serial_sync.hpp"

// Build (MSYS2 UCRT64/MINGW64):
// g++ -O2 -std=c++20 gwo_serial_mt_bm.cpp -o gwo_serial_mt_bm.exe -I "../eigen-5.0.0" $(pkg-config --cflags --libs opencv4)
//
// Run:
// ./gwo_serial_mt_bm.exe
// Optional args:
// ./gwo_serial_mt_bm.exe ./image ./output 4

namespace fs = std::filesystem;

static std::array<double,256> normalized_histogram_u8(const cv::Mat& gray_u8) {
    std::array<double,256> h{};
    h.fill(0.0);
    const int rows = gray_u8.rows, cols = gray_u8.cols;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* p = gray_u8.ptr<uint8_t>(r);
        for (int c = 0; c < cols; ++c) h[p[c]] += 1.0;
    }
    const double n = (double)rows * (double)cols;
    for (double& v : h) v /= std::max(1.0, n);
    return h;
}

static inline int class_of_pixel(uint8_t v, const std::vector<int>& thr_sorted) {
    int c = 0;
    while (c < (int)thr_sorted.size() && v > (uint8_t)thr_sorted[c]) c++;
    return c;
}

static cv::Mat make_segmented_image(const cv::Mat& gray_u8,
                                    const std::vector<int>& thr_sorted) {
    const int K = (int)thr_sorted.size() + 1;
    std::vector<uint8_t> levels(K);
    if (K == 1) levels[0] = 0;
    else {
        for (int c = 0; c < K; ++c)
            levels[c] = (uint8_t)std::lround(255.0 * (double)c / (double)(K - 1));
    }

    cv::Mat out(gray_u8.size(), CV_8UC1);
    for (int r = 0; r < gray_u8.rows; ++r) {
        const uint8_t* p = gray_u8.ptr<uint8_t>(r);
        uint8_t* q = out.ptr<uint8_t>(r);
        for (int c = 0; c < gray_u8.cols; ++c) {
            int cls = class_of_pixel(p[c], thr_sorted);
            q[c] = levels[std::max(0, std::min(cls, K-1))];
        }
    }
    return out;
}

// -------------------- Otsu multilevel from histogram (maximize sigmaB^2) --------------------
struct OtsuPrecomp {
    // prefix sums for p and i*p
    std::array<double,256> P{};   // cumulative prob
    std::array<double,256> S{};   // cumulative mean sum (i*p)
    double muT = 0.0;
};

static OtsuPrecomp build_otsu_precomp(const std::array<double,256>& p) {
    OtsuPrecomp pc;
    double cumP = 0.0;
    double cumS = 0.0;
    for (int i = 0; i < 256; ++i) {
        cumP += p[i];
        cumS += (double)i * p[i];
        pc.P[i] = cumP;
        pc.S[i] = cumS;
    }
    pc.muT = pc.S[255]; // total mean
    return pc;
}

static inline double range_prob(const OtsuPrecomp& pc, int a, int b) {
    if (a > b) return 0.0;
    if (a <= 0) return pc.P[b];
    return pc.P[b] - pc.P[a - 1];
}

static inline double range_mean_sum(const OtsuPrecomp& pc, int a, int b) {
    if (a > b) return 0.0;
    if (a <= 0) return pc.S[b];
    return pc.S[b] - pc.S[a - 1];
}

// Compute sigma_B^2 for thresholds thr (sorted, size k), classes = k+1
static double otsu_between_class_variance(const OtsuPrecomp& pc,
                                         const std::vector<int>& thr_sorted) {
    const int k = (int)thr_sorted.size();
    const int K = k + 1;

    // Build class intervals:
    // class0: [0, t1], class1: [t1+1, t2], ..., classk: [tk+1, 255]
    double sigmaB2 = 0.0;

    int a = 0;
    for (int c = 0; c < K; ++c) {
        int b = (c < k) ? thr_sorted[c] : 255;
        double w = range_prob(pc, a, b);
        if (w > 1e-15) {
            double ms = range_mean_sum(pc, a, b);
            double m = ms / w;
            double d = (m - pc.muT);
            sigmaB2 += w * d * d;
        } else {
            // empty class -> penalize by making objective small
            // (sigmaB2 unchanged, but later we can penalize)
        }
        a = b + 1;
    }
    return sigmaB2;
}

// -------------------- Multilevel Thresholding Problem (Otsu) --------------------
struct MultiLevelOtsuProblem : public GWO::Problem<double>
{
    using GWO::Problem<double>::Problem;

    const cv::Mat* gray = nullptr;
    std::array<double,256> hist{};
    OtsuPrecomp pc{};

    MultiLevelOtsuProblem(const GWO::Setup& setup_, const cv::Mat& gray_)
        : GWO::Problem<double>(setup_)
    {
        if (gray_.empty() || gray_.type() != CV_8UC1)
            throw std::runtime_error("Input image must be CV_8UC1 grayscale");
        gray = &gray_;
        hist = normalized_histogram_u8(gray_);
        pc = build_otsu_precomp(hist);
    }

    double fitness(const Eigen::ArrayXd &pos) const override {
        const int k = (int)pos.size();

        std::vector<int> thr;
        thr.reserve(k);
        for (int i = 0; i < k; ++i) {
            int t = (int)std::lround(pos(i));
            t = std::max(0, std::min(255, t));
            thr.push_back(t);
        }
        std::sort(thr.begin(), thr.end());
        thr.erase(std::unique(thr.begin(), thr.end()), thr.end());

        // Need exactly k thresholds
        if ((int)thr.size() < k) {
            return 1e9 + 1e6 * (k - (int)thr.size());
        }

        // Also avoid degenerate thresholds too close causing empty classes
        // Quick check: if any interval becomes empty, penalize.
        int prev = -1;
        for (int t : thr) {
            if (t <= prev) return 1e9;
            prev = t;
        }

        double sigmaB2 = otsu_between_class_variance(pc, thr);

        // additional penalty for empty classes (check intervals)
        int a = 0;
        for (int c = 0; c < k + 1; ++c) {
            int b = (c < k) ? thr[c] : 255;
            double w = range_prob(pc, a, b);
            if (w <= 1e-15) {
                return 1e8; // empty class => bad
            }
            a = b + 1;
        }

        // GWO minimizes -> maximize sigmaB2 by returning negative
        return -sigmaB2;
    }
};

// helper: list image files in folder
static std::vector<fs::path> list_images(const fs::path& dir) {
    std::vector<fs::path> files;
    if (!fs::exists(dir)) return files;

    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

static std::vector<int> pos_to_thresholds(const Eigen::ArrayXd& pos) {
    std::vector<int> thr;
    thr.reserve(pos.size());
    for (int i = 0; i < pos.size(); ++i) {
        int t = (int)std::lround(pos(i));
        t = std::max(0, std::min(255, t));
        thr.push_back(t);
    }
    std::sort(thr.begin(), thr.end());
    thr.erase(std::unique(thr.begin(), thr.end()), thr.end());
    return thr;
}

int main(int argc, char** argv) {
    fs::path image_dir  = (argc >= 2) ? fs::path(argv[1]) : fs::path("./image");
    fs::path output_dir = (argc >= 3) ? fs::path(argv[2]) : fs::path("./output");
    int N_thresholds    = (argc >= 4) ? std::stoi(argv[3]) : 4;

    if (N_thresholds < 1 || N_thresholds > 16) {
        std::cerr << "N_thresholds should be in [1,16]\n";
        return 1;
    }

    auto images = list_images(image_dir);
    if (images.empty()) {
        std::cerr << "No images found in: " << image_dir.string() << "\n";
        return 1;
    }

    fs::create_directories(output_dir);

    // ---- benchmark params (you can change) ----
    std::vector<int> Pop_list = {20, 40, 80, 160};
    const int RUNS      = 1;
    const int MAX_ITERS = 300;
    const uint64_t GWO_SEED = 123456789ULL;

    std::string problem_name = "MultiThreshold_Otsu";
    std::string filename = "gwo_serial_mt_bm_otsu.csv";

    // CSV header once
    bool need_header =
        (!fs::exists(filename) || fs::file_size(filename) == 0);

    std::ofstream csv(filename, std::ios::app);
    if (!csv) {
        std::cerr << "Cannot open csv: " << filename << "\n";
        return 1;
    }
    if (need_header) {
        csv << "problem,image,thresholds,POP_SIZE,max_iters,avg_ms,best_fitness(neg_sigmaB2)\n";
    }

    for (const auto& img_path : images) {
        cv::Mat bgr = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (bgr.empty()) {
            std::cerr << "Skip (cannot read): " << img_path.string() << "\n";
            continue;
        }
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        // Setup for this image and fixed N_thresholds
        GWO::Setup setup;
        setup.N = (size_t)N_thresholds;
        setup.minRange = Eigen::ArrayXd::Constant(N_thresholds, 0.0);
        setup.maxRange = Eigen::ArrayXd::Constant(N_thresholds, 255.0);

        for (int POP_SIZE : Pop_list) {
            setup.POP_SIZE = (size_t)POP_SIZE;

            long long total_ms = 0;
            double best_fit_last = 0.0;
            Eigen::ArrayXd best_pos_last;

            std::cout << "================================\n";
            std::cout << "Image: " << img_path.filename().string()
                      << " | Otsu multilevel | N(thr)=" << N_thresholds
                      << " | POP_SIZE=" << POP_SIZE
                      << " | MAX_ITERS=" << MAX_ITERS << "\n";

            for (int r = 1; r <= RUNS; ++r) {
                GWO::global_seed = GWO_SEED;

                MultiLevelOtsuProblem problem(setup, gray);

                auto start = std::chrono::steady_clock::now();
                auto best  = problem.run(MAX_ITERS);
                auto end   = std::chrono::steady_clock::now();

                long long ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                total_ms += ms;
                best_fit_last = best.savedFitness; // negative sigmaB2
                best_pos_last = best.pos;

                std::cout << "Run " << r
                          << " | " << ms << " ms"
                          << " | best fitness = " << best.savedFitness
                          << " (sigmaB2=" << (-best.savedFitness) << ")"
                          << "\n";
            }

            double avg_ms = total_ms / double(RUNS);

            csv << problem_name
                << "," << img_path.filename().string()
                << "," << N_thresholds
                << "," << POP_SIZE
                << "," << MAX_ITERS
                << "," << avg_ms
                << "," << best_fit_last
                << "\n";
            csv.flush();

            // Save segmented image for each (image, POP_SIZE)
            auto thr = pos_to_thresholds(best_pos_last);
            // ensure exactly N_thresholds (if duplicates occurred, skip save)
            if ((int)thr.size() == N_thresholds) {
                cv::Mat seg = make_segmented_image(gray, thr);

                std::string out_name =
                    img_path.stem().string() +
                    "_otsu_k" + std::to_string(N_thresholds) +
                    "_pop" + std::to_string(POP_SIZE) + ".png";

                fs::path out_path = output_dir / out_name;
                cv::imwrite(out_path.string(), seg);

                std::cout << "Saved: " << out_path.string() << "\n";
                std::cout << "Thresholds: ";
                for (size_t i = 0; i < thr.size(); ++i)
                    std::cout << thr[i] << (i + 1 < thr.size() ? ", " : "\n");
            } else {
                std::cout << "Skip save: thresholds collapsed by duplicates.\n";
            }
        }
    }

    csv.close();
    std::cout << "Done. CSV: " << filename << " | Output dir: " << output_dir.string() << "\n";
    return 0;
}
