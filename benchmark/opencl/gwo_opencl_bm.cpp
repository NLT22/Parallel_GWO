// gwo_opencl_bm.cpp
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <limits>

// g++ gwo_opencl_bm.cpp -O2 -o opencl_bm.exe -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/lib/x64" -lOpenCL

#define CHECK(err) \
    if ((err) != CL_SUCCESS) { \
        std::cerr << "OpenCL error " << (err) << " at line " << __LINE__ << "\n"; \
        std::exit(1); \
    }

static std::string load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open file: " << path << "\n";
        std::exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

static std::string get_platform_str(cl_platform_id pid, cl_platform_info param) {
    size_t sz = 0;
    CHECK(clGetPlatformInfo(pid, param, 0, nullptr, &sz));
    std::string s(sz, '\0');
    CHECK(clGetPlatformInfo(pid, param, sz, s.data(), nullptr));
    while (!s.empty() && (s.back() == '\0' || s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

static std::string get_device_str(cl_device_id did, cl_device_info param) {
    size_t sz = 0;
    CHECK(clGetDeviceInfo(did, param, 0, nullptr, &sz));
    std::string s(sz, '\0');
    CHECK(clGetDeviceInfo(did, param, sz, s.data(), nullptr));
    while (!s.empty() && (s.back() == '\0' || s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

static std::string dev_type_str(cl_device_type t) {
    std::string out;
    if (t & CL_DEVICE_TYPE_CPU) out += "CPU";
    if (t & CL_DEVICE_TYPE_GPU) { if (!out.empty()) out += "+"; out += "GPU"; }
    if (t & CL_DEVICE_TYPE_ACCELERATOR) { if (!out.empty()) out += "+"; out += "ACCEL"; }
    if (out.empty()) out = "OTHER";
    return out;
}

static size_t round_up(size_t x, size_t g) {
    if (g == 0) return x;
    return ((x + g - 1) / g) * g;
}

static bool is_pow2(size_t x) { return x && ((x & (x - 1)) == 0); }
static size_t floor_pow2(size_t x) {
    if (x == 0) return 0;
    size_t p = 1;
    while ((p << 1) <= x) p <<= 1;
    return p;
}

static size_t clamp_local_pow2(size_t want, size_t kernel_max_wg) {
    // For reduce kernels: local size should be power-of-two for the tree reduction.
    size_t cap = kernel_max_wg;
    if (cap == 0) return 0;

    size_t L = std::min(want, cap);
    if (!is_pow2(L)) L = floor_pow2(L);
    if (L == 0) L = floor_pow2(cap);
    if (L == 0) L = 1;
    return L;
}

struct TargetDevice {
    std::string platform_substr;
    std::string device_substr;
};

static bool find_device(
    const std::vector<cl_platform_id>& platforms,
    const TargetDevice& t,
    cl_platform_id& out_pid,
    cl_device_id& out_did
) {
    for (auto pid : platforms) {
        std::string pname = get_platform_str(pid, CL_PLATFORM_NAME);
        if (pname.find(t.platform_substr) == std::string::npos) continue;

        cl_uint num_dev = 0;
        cl_int e = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_dev);
        if (e != CL_SUCCESS || num_dev == 0) continue;

        std::vector<cl_device_id> devs(num_dev);
        CHECK(clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, num_dev, devs.data(), nullptr));

        for (auto did : devs) {
            std::string dname = get_device_str(did, CL_DEVICE_NAME);
            if (dname.find(t.device_substr) != std::string::npos) {
                out_pid = pid;
                out_did = did;
                return true;
            }
        }
    }
    return false;
}

static bool device_has_fp64(cl_device_id did) {
    std::string exts = get_device_str(did, CL_DEVICE_EXTENSIONS);
    return (exts.find("cl_khr_fp64") != std::string::npos);
}

static cl_ulong get_global_mem_bytes(cl_device_id did) {
    cl_ulong mem = 0;
    CHECK(clGetDeviceInfo(did, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr));
    return mem;
}

static size_t get_max_wg_size_kernel(cl_kernel k, cl_device_id did) {
    size_t max_wg = 0;
    CHECK(clGetKernelWorkGroupInfo(k, did, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr));
    return max_wg;
}

static void print_header_device(const std::string& platform_name,
                                const std::string& device_name,
                                const std::string& device_type,
                                bool fp64) {
    std::cout << "\n==================================================\n";
    std::cout << "Benchmark on:\n";
    std::cout << "  Platform: " << platform_name << "\n";
    std::cout << "  Device  : " << device_name << " (" << device_type << ")\n";
    std::cout << "  FP64    : " << (fp64 ? "YES" : "NO -> FP32") << "\n";
    std::cout << "==================================================\n";
}

// --------- Main benchmark for one device ---------
// Simpler pipeline:
// update -> reduce1 -> reduce2 -> top3_find_global(1 work-item) -> gather
template <typename Real>
static void run_one_device(
    const std::string& platform_name,
    const std::string& device_name,
    const std::string& device_type,
    cl_device_id did,
    const std::string& kernel_src,
    const std::string& build_opts,
    const std::vector<int>& N_list,
    const std::vector<int>& POP_list,
    int MAX_ITERS,
    int RUNS,
    cl_ulong GLOBAL_SEED,
    size_t WANT_L_UPDATE,
    size_t WANT_L_REDUCE,
    std::ofstream& csv
) {
    cl_int err;

    cl_context context = clCreateContext(nullptr, 1, &did, nullptr, nullptr, &err);
    CHECK(err);

    cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, did, props, &err);
    CHECK(err);

    const char* src_c = kernel_src.c_str();
    size_t src_len = kernel_src.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src_c, &src_len, &err);
    CHECK(err);

    err = clBuildProgram(program, 1, &did, build_opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, did, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, did, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build failed on " << device_name << "\nBuild log:\n" << log << "\n";
        std::exit(1);
    }

    cl_kernel k_update  = clCreateKernel(program, "gwo_update_and_term", &err); CHECK(err);
    cl_kernel k_red1    = clCreateKernel(program, "reduce_stage1",       &err); CHECK(err);
    cl_kernel k_red2    = clCreateKernel(program, "reduce_stage2",       &err); CHECK(err);
    cl_kernel k_top3g   = clCreateKernel(program, "top3_find_global",    &err); CHECK(err);
    cl_kernel k_gather  = clCreateKernel(program, "gather_leaders",      &err); CHECK(err);

    size_t max_wg_update = get_max_wg_size_kernel(k_update, did);
    size_t max_wg_reduce = get_max_wg_size_kernel(k_red1,   did);
    size_t max_wg_gather = get_max_wg_size_kernel(k_gather, did);

    // Update can be any size <= max. Reduce needs pow2 for tree reduction.
    size_t L_UPDATE = std::min(WANT_L_UPDATE, max_wg_update);
    if (L_UPDATE == 0) L_UPDATE = 1;

    size_t L_REDUCE = clamp_local_pow2(WANT_L_REDUCE, max_wg_reduce);
    size_t L_GATHER = std::min(L_UPDATE, max_wg_gather);
    if (L_GATHER == 0) L_GATHER = 1;

    std::cout << "  Kernel max WG size: update=" << max_wg_update
              << ", reduce=" << max_wg_reduce
              << ", gather=" << max_wg_gather << "\n";
    std::cout << "  Locals used       : update=" << L_UPDATE
              << ", reduce=" << L_REDUCE
              << ", gather=" << L_GATHER << "\n";

    const Real minX = (Real)-5.12;
    const Real maxX = (Real) 5.12;

    cl_ulong dev_mem = get_global_mem_bytes(did);

    for (int N : N_list) {
        for (int POP : POP_list) {
            if (POP < 3) continue;

            const size_t totalXN = (size_t)POP * (size_t)N;
            const size_t blocks_per_wolf = (size_t)((N + (int)L_REDUCE - 1) / (int)L_REDUCE);

            // memory estimate (rough)
            size_t bytes =
                sizeof(Real) * (totalXN * 2                    // X + term
                              + (size_t)POP * blocks_per_wolf  // partial
                              + (size_t)POP                    // fitness
                              + (size_t)(3 * (size_t)N))       // leaders
              + sizeof(int) * 3;                                // top3 idx

            if ((cl_ulong)bytes > dev_mem * 9 / 10) {
                std::cout << "  Skip N=" << N << " POP=" << POP
                          << " (memory est " << (bytes / (1024.0*1024.0)) << " MB > ~90% device mem)\n";
                continue;
            }

            // Host init X
            std::vector<Real> X(totalXN);
            for (auto& v : X) {
                double u = rand() / double(RAND_MAX);
                v = (Real)(-5.12 + 10.24 * u);
            }

            // Buffers
            cl_mem bufX       = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Real) * totalXN, nullptr, &err); CHECK(err);
            cl_mem bufTerm    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Real) * totalXN, nullptr, &err); CHECK(err);
            cl_mem bufPart    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Real) * ((size_t)POP * blocks_per_wolf), nullptr, &err); CHECK(err);
            cl_mem bufFit     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Real) * (size_t)POP, nullptr, &err); CHECK(err);

            cl_mem bufA       = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Real) * (size_t)N, nullptr, &err); CHECK(err);
            cl_mem bufB       = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Real) * (size_t)N, nullptr, &err); CHECK(err);
            cl_mem bufD       = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Real) * (size_t)N, nullptr, &err); CHECK(err);

            cl_mem bufTop3Idx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 3, nullptr, &err); CHECK(err);

            long long total_ms = 0;
            Real best_fit_last = (Real)0;
            int best_idx_last[3] = {0,1,2};

            for (int run = 0; run < RUNS; run++) {
                CHECK(clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, sizeof(Real)*totalXN, X.data(), 0, nullptr, nullptr));

                int init_idx[3] = {0,1,2};
                CHECK(clEnqueueWriteBuffer(queue, bufTop3Idx, CL_TRUE, 0, sizeof(int)*3, init_idx, 0, nullptr, nullptr));

                // gather initial leaders
                {
                    CHECK(clSetKernelArg(k_gather, 0, sizeof(cl_mem), &bufX));
                    CHECK(clSetKernelArg(k_gather, 1, sizeof(int), &N));
                    CHECK(clSetKernelArg(k_gather, 2, sizeof(int), &POP));
                    CHECK(clSetKernelArg(k_gather, 3, sizeof(cl_mem), &bufTop3Idx));
                    CHECK(clSetKernelArg(k_gather, 4, sizeof(cl_mem), &bufA));
                    CHECK(clSetKernelArg(k_gather, 5, sizeof(cl_mem), &bufB));
                    CHECK(clSetKernelArg(k_gather, 6, sizeof(cl_mem), &bufD));

                    size_t g = round_up((size_t)N, L_GATHER);
                    size_t l = L_GATHER;
                    CHECK(clEnqueueNDRangeKernel(queue, k_gather, 1, nullptr, &g, &l, 0, nullptr, nullptr));
                }

                clFinish(queue);

                auto t0 = std::chrono::steady_clock::now();

                for (int iter = 0; iter < MAX_ITERS; iter++) {
                    // 1) update + term
                    CHECK(clSetKernelArg(k_update, 0, sizeof(cl_mem), &bufX));
                    CHECK(clSetKernelArg(k_update, 1, sizeof(cl_mem), &bufTerm));
                    CHECK(clSetKernelArg(k_update, 2, sizeof(cl_mem), &bufA));
                    CHECK(clSetKernelArg(k_update, 3, sizeof(cl_mem), &bufB));
                    CHECK(clSetKernelArg(k_update, 4, sizeof(cl_mem), &bufD));
                    CHECK(clSetKernelArg(k_update, 5, sizeof(int), &N));
                    CHECK(clSetKernelArg(k_update, 6, sizeof(int), &POP));
                    CHECK(clSetKernelArg(k_update, 7, sizeof(int), &iter));
                    CHECK(clSetKernelArg(k_update, 8, sizeof(int), &MAX_ITERS));
                    CHECK(clSetKernelArg(k_update, 9, sizeof(cl_ulong), &GLOBAL_SEED));
                    CHECK(clSetKernelArg(k_update,10, sizeof(Real), &minX));
                    CHECK(clSetKernelArg(k_update,11, sizeof(Real), &maxX));

                    size_t global_update = round_up(totalXN, L_UPDATE);
                    size_t local_update  = L_UPDATE;
                    CHECK(clEnqueueNDRangeKernel(queue, k_update, 1, nullptr, &global_update, &local_update, 0, nullptr, nullptr));

                    // 2) reduce stage1
                    {
                        int blocks_i = (int)blocks_per_wolf;

                        CHECK(clSetKernelArg(k_red1, 0, sizeof(cl_mem), &bufTerm));
                        CHECK(clSetKernelArg(k_red1, 1, sizeof(cl_mem), &bufPart));
                        CHECK(clSetKernelArg(k_red1, 2, sizeof(int), &N));
                        CHECK(clSetKernelArg(k_red1, 3, sizeof(int), &POP));
                        CHECK(clSetKernelArg(k_red1, 4, sizeof(int), &blocks_i));
                        // dynamic local memory:
                        CHECK(clSetKernelArg(k_red1, 5, sizeof(Real) * L_REDUCE, nullptr));

                        size_t local  = L_REDUCE;
                        size_t global = (size_t)POP * blocks_per_wolf * local;
                        global = round_up(global, local);

                        CHECK(clEnqueueNDRangeKernel(queue, k_red1, 1, nullptr, &global, &local, 0, nullptr, nullptr));
                    }

                    // 3) reduce stage2
                    {
                        int blocks_i = (int)blocks_per_wolf;

                        CHECK(clSetKernelArg(k_red2, 0, sizeof(cl_mem), &bufPart));
                        CHECK(clSetKernelArg(k_red2, 1, sizeof(cl_mem), &bufFit));
                        CHECK(clSetKernelArg(k_red2, 2, sizeof(int), &POP));
                        CHECK(clSetKernelArg(k_red2, 3, sizeof(int), &blocks_i));
                        CHECK(clSetKernelArg(k_red2, 4, sizeof(int), &N));
                        // dynamic local memory:
                        CHECK(clSetKernelArg(k_red2, 5, sizeof(Real) * L_REDUCE, nullptr));

                        size_t local  = L_REDUCE;
                        size_t global = (size_t)POP * local;
                        global = round_up(global, local);

                        CHECK(clEnqueueNDRangeKernel(queue, k_red2, 1, nullptr, &global, &local, 0, nullptr, nullptr));
                    }

                    // 4) global top3 (single work-item)
                    {
                        CHECK(clSetKernelArg(k_top3g, 0, sizeof(cl_mem), &bufFit));
                        CHECK(clSetKernelArg(k_top3g, 1, sizeof(int), &POP));
                        CHECK(clSetKernelArg(k_top3g, 2, sizeof(cl_mem), &bufTop3Idx));

                        size_t global = 1, local = 1;
                        CHECK(clEnqueueNDRangeKernel(queue, k_top3g, 1, nullptr, &global, &local, 0, nullptr, nullptr));
                    }

                    // 5) gather leaders
                    {
                        CHECK(clSetKernelArg(k_gather, 0, sizeof(cl_mem), &bufX));
                        CHECK(clSetKernelArg(k_gather, 1, sizeof(int), &N));
                        CHECK(clSetKernelArg(k_gather, 2, sizeof(int), &POP));
                        CHECK(clSetKernelArg(k_gather, 3, sizeof(cl_mem), &bufTop3Idx));
                        CHECK(clSetKernelArg(k_gather, 4, sizeof(cl_mem), &bufA));
                        CHECK(clSetKernelArg(k_gather, 5, sizeof(cl_mem), &bufB));
                        CHECK(clSetKernelArg(k_gather, 6, sizeof(cl_mem), &bufD));

                        size_t g = round_up((size_t)N, L_GATHER);
                        size_t l = L_GATHER;
                        CHECK(clEnqueueNDRangeKernel(queue, k_gather, 1, nullptr, &g, &l, 0, nullptr, nullptr));
                    }
                }

                clFinish(queue);

                auto t1 = std::chrono::steady_clock::now();
                total_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

                // read final indices + best fitness
                CHECK(clEnqueueReadBuffer(queue, bufTop3Idx, CL_TRUE, 0, sizeof(int)*3, best_idx_last, 0, nullptr, nullptr));

                int a = best_idx_last[0];
                if (a < 0 || a >= POP) a = 0;

                Real fbest{};
                CHECK(clEnqueueReadBuffer(queue, bufFit, CL_TRUE, sizeof(Real)*(size_t)a, sizeof(Real), &fbest, 0, nullptr, nullptr));
                best_fit_last = fbest;
            }

            double avg_ms = total_ms / double(RUNS);

            csv << platform_name << ","
                << device_name << ","
                << device_type << ","
                << (std::is_same<Real,double>::value ? "FP64" : "FP32") << ","
                << N << ","
                << POP << ","
                << MAX_ITERS << ","
                << (long long)L_UPDATE << ","
                << (long long)L_REDUCE << ","
                << avg_ms << ","
                << (double)best_fit_last << ","
                << best_idx_last[0] << ","
                << best_idx_last[1] << ","
                << best_idx_last[2] << "\n";

            std::cout << "  OpenCL "
                      << (std::is_same<Real,double>::value ? "FP64" : "FP32")
                      << " | N=" << N
                      << " | POP=" << POP
                      << " | iters=" << MAX_ITERS
                      << " | avg=" << avg_ms << " ms"
                      << " | best_fit=" << (double)best_fit_last
                      << " | idx=(" << best_idx_last[0] << "," << best_idx_last[1] << "," << best_idx_last[2] << ")"
                      << "\n";

            clReleaseMemObject(bufX);
            clReleaseMemObject(bufTerm);
            clReleaseMemObject(bufPart);
            clReleaseMemObject(bufFit);
            clReleaseMemObject(bufA);
            clReleaseMemObject(bufB);
            clReleaseMemObject(bufD);
            clReleaseMemObject(bufTop3Idx);
        }
    }

    clReleaseKernel(k_update);
    clReleaseKernel(k_red1);
    clReleaseKernel(k_red2);
    clReleaseKernel(k_top3g);
    clReleaseKernel(k_gather);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main() {
    std::vector<TargetDevice> targets = {
        {"NVIDIA CUDA", "RTX 3050 Ti"},
        {"Intel(R) OpenCL HD Graphics", "Iris"},
        {"Intel(R) OpenCL", "i7-12700H"}
    };

    // You can change freely
    std::vector<int> N_list   = {10, 50, 100, 500, 1000};
    std::vector<int> POP_list = {50, 100, 200, 500, 1000};

    const int MAX_ITERS = 1000;
    const int RUNS = 3;
    const cl_ulong GLOBAL_SEED = 123456789ULL;

    // “Want” locals (will be clamped)
    const size_t WANT_L_UPDATE = 256;
    const size_t WANT_L_REDUCE = 256;

    std::string kernel_src = load_file("gwo_rastrigin.cl");

    cl_uint num_platforms = 0;
    CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    std::vector<cl_platform_id> platforms(num_platforms);
    CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    std::ofstream csv("gwo_opencl.csv");
    csv << "platform,device,device_type,precision,N,POP,iters,L_update,L_reduce,avg_ms,best_fit,alpha,beta,delta\n";

    for (const auto& t : targets) {
        cl_platform_id pid{};
        cl_device_id did{};
        if (!find_device(platforms, t, pid, did)) {
            std::cerr << "WARNING: cannot find device: platform contains [" << t.platform_substr
                      << "], device contains [" << t.device_substr << "]\n";
            continue;
        }

        std::string platform_name = get_platform_str(pid, CL_PLATFORM_NAME);
        std::string device_name   = get_device_str(did, CL_DEVICE_NAME);

        cl_device_type dtype{};
        CHECK(clGetDeviceInfo(did, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr));
        std::string device_type = dev_type_str(dtype);

        bool has_fp64 = device_has_fp64(did);
        print_header_device(platform_name, device_name, device_type, has_fp64);

        // No fast-math for reproducibility
        std::string build_opts = has_fp64 ? "-DREAL_IS_DOUBLE=1" : "-DREAL_IS_DOUBLE=0";

        if (has_fp64) {
            run_one_device<double>(
                platform_name, device_name, device_type,
                did, kernel_src, build_opts,
                N_list, POP_list,
                MAX_ITERS, RUNS, GLOBAL_SEED,
                WANT_L_UPDATE, WANT_L_REDUCE,
                csv
            );
        } else {
            run_one_device<float>(
                platform_name, device_name, device_type,
                did, kernel_src, build_opts,
                N_list, POP_list,
                MAX_ITERS, RUNS, GLOBAL_SEED,
                WANT_L_UPDATE, WANT_L_REDUCE,
                csv
            );
        }
    }

    csv.close();
    std::cout << "\nSaved CSV: gwo_opencl.csv\n";
    return 0;
}
