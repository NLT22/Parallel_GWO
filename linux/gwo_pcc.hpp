#ifndef GWO_PCCGWO_HPP
#define GWO_PCCGWO_HPP

#include "gwo_serial.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GWO
{
    // Hàm tiện ích: chia d chiều thành m đoạn gần bằng nhau
    inline std::vector<std::pair<int,int>> make_blocks(int d, int m)
    {
        if (m <= 0) m = 1;
        if (m > d) m = d;
        std::vector<std::pair<int,int>> blocks;
        blocks.reserve(m);

        int base = d / m;
        int rem  = d % m;
        int cur  = 0;
        for (int j = 0; j < m; ++j)
        {
            int len = base + (j < rem ? 1 : 0);
            blocks.emplace_back(cur, len); // (start, len)
            cur += len;
        }
        return blocks;
    }

    // ===========================
    // PCCGWO: Master–Slave style
    // ===========================
    template <std::floating_point T>
    Wolf<T> run_pccgwo(Problem<T>& problem,
                       int maxIterations,
                       int m_subswarms)
    {
        const int D   = static_cast<int>(problem.setup.N);
        const int POP = static_cast<int>(problem.setup.POP_SIZE);

        if (POP <= 0 || D <= 0)
            throw std::runtime_error("Invalid POP_SIZE or N in Setup.");

        // 1) Chia chiều thành m khối (sub-components) như (11)–(12)
        auto blocks = make_blocks(D, m_subswarms);
        const int M = static_cast<int>(blocks.size());

        // 2) Khởi tạo quần thể riêng cho PCCGWO (Song song hoá mức thuật toán)
        //    (Không dùng population sẵn trong Problem, chỉ dùng fitness())
        std::vector<Wolf<T>> wolves;
        wolves.reserve(POP);
        for (int i = 0; i < POP; ++i)
        {
            wolves.emplace_back(problem.setup.N);
            wolves.back().randomize(problem.setup.minRange,
                                    problem.setup.maxRange);
        }

        // 3) Khởi tạo buffer vector C (context vector) – ghép đại diện từ mỗi sub-swarm
        Eigen::ArrayX<T> C(D);

        // mỗi sub-swarm chọn ngẫu nhiên một cá thể làm đại diện ban đầu
        std::vector<int> repIndex(M); // chỉ số con sói đại diện cho từng sub-swarm
        for (int j = 0; j < M; ++j)
        {
            repIndex[j] = j % POP; // đơn giản: chọn tuần tự
        }

        // ghép C từ các đại diện
        for (int j = 0; j < M; ++j)
        {
            auto [start, len] = blocks[j];
            C.segment(start, len) =
                wolves[repIndex[j]].pos.segment(start, len);
        }

        // ===========================
        // Vòng lặp tiến hoá (Cycle)
        // ===========================
        for (int iter = 0; iter < maxIterations; ++iter)
        {
            // -------------------------------
            // MASTER: gửi C cho tất cả slaves
            // -------------------------------
            // (ở đây C là biến shared, slaves chỉ đọc)

            // Mỗi sub-swarm trả về: best segment + index wolf tương ứng
            std::vector<int> newRepIndex(M);

            // -------------------------------
            // Parallel for j = 1..M (slaves)
            // -------------------------------
            #pragma omp parallel for
            for (int j = 0; j < M; ++j)
            {
                auto [start, len] = blocks[j];

                // 1) Đánh giá fitness cho tất cả cá thể trong sub-swarm S_j
                //    Ở đây đơn giản: S_j = toàn bộ quần thể, nhưng chỉ tối ưu đoạn [start, start+len)
                Eigen::ArrayX<T> fvals(POP);

                for (int i = 0; i < POP; ++i)
                {
                    // xây C_i = C nhưng thay đoạn j bằng vị trí hiện tại của wolf i
                    Eigen::ArrayX<T> Ci = C;
                    Ci.segment(start, len) = wolves[i].pos.segment(start, len);
                    fvals(i) = problem.fitness(Ci);
                }

                // 2) Tìm alpha, beta, delta trong sub-swarm j
                int alpha = 0, beta = 0, delta = 0;
                {
                    // khởi tạo alpha = cá thể tốt nhất
                    alpha = 0;
                    for (int i = 1; i < POP; ++i)
                        if (fvals(i) < fvals(alpha)) alpha = i;

                    // beta = tốt nhì, delta = tốt ba
                    beta  = -1;
                    delta = -1;
                    for (int i = 0; i < POP; ++i)
                    {
                        if (i == alpha) continue;
                        if (beta == -1 || fvals(i) < fvals(beta))
                        {
                            delta = beta;
                            beta  = i;
                        }
                        else if (delta == -1 || fvals(i) < fvals(delta))
                        {
                            delta = i;
                        }
                    }
                    if (beta  == -1) beta  = alpha;
                    if (delta == -1) delta = alpha;
                }

                // 3) Cập nhật vị trí cho block j của mỗi wolf theo GWO (9)–(10)
                T a = 2 * (1 - T(iter) / T(maxIterations));

                for (int i = 0; i < POP; ++i)
                {
                    for (int k = 0; k < len; ++k)
                    {
                        int dim = start + k;

                        T r1 = random<T>(0.0, 1.0);
                        T r2 = random<T>(0.0, 1.0);
                        T A1 = 2 * a * r1 - a;
                        T C1 = 2 * r2;

                        r1 = random<T>(0.0, 1.0);
                        r2 = random<T>(0.0, 1.0);
                        T A2 = 2 * a * r1 - a;
                        T C2 = 2 * r2;

                        r1 = random<T>(0.0, 1.0);
                        r2 = random<T>(0.0, 1.0);
                        T A3 = 2 * a * r1 - a;
                        T C3 = 2 * r2;

                        T X_alpha = wolves[alpha].pos(dim);
                        T X_beta  = wolves[beta].pos(dim);
                        T X_delta = wolves[delta].pos(dim);
                        T X_i     = wolves[i].pos(dim);

                        T D_alpha = std::abs(C1 * X_alpha - X_i);
                        T D_beta  = std::abs(C2 * X_beta  - X_i);
                        T D_delta = std::abs(C3 * X_delta - X_i);

                        T X1 = X_alpha - A1 * D_alpha;
                        T X2 = X_beta  - A2 * D_beta;
                        T X3 = X_delta - A3 * D_delta;

                        T X_new = (X1 + X2 + X3) / T(3);

                        // chặn trong [minRange, maxRange]
                        X_new = std::max(
                            problem.setup.minRange(dim),
                            std::min(problem.setup.maxRange(dim), X_new)
                        );

                        wolves[i].pos(dim) = X_new;
                    }
                }

                // 4) Chọn đại diện mới của S_j (tốt nhất sau cập nhật)
                int bestIdx = 0;
                {
                    // tính lại fitness với block j mới
                    T bestFit;
                    {
                        Eigen::ArrayX<T> Ci = C;
                        Ci.segment(start, len) =
                            wolves[0].pos.segment(start, len);
                        bestFit = problem.fitness(Ci);
                    }

                    for (int i = 1; i < POP; ++i)
                    {
                        Eigen::ArrayX<T> Ci = C;
                        Ci.segment(start, len) =
                            wolves[i].pos.segment(start, len);
                        T fi = problem.fitness(Ci);
                        if (fi < bestFit)
                        {
                            bestFit = fi;
                            bestIdx = i;
                        }
                    }
                }

                newRepIndex[j] = bestIdx;
            } // end parallel for j

            // -------------------------------
            // MASTER: nhận đại diện từ các slaves, cập nhật buffer C
            // -------------------------------
            for (int j = 0; j < M; ++j)
            {
                auto [start, len] = blocks[j];
                int idx = newRepIndex[j];
                repIndex[j] = idx;
                C.segment(start, len) =
                    wolves[idx].pos.segment(start, len);
            }
        } // end for iter

        // Sau khi kết thúc, C là lời giải ghép từ các sub-swarms
        Wolf<T> best(D);
        best.pos = C;
        best.len = D;
        best.savedFitness = problem.fitness(C);
        return best;
    }

} // namespace GWO

#endif // GWO_PCCGWO_HPP
