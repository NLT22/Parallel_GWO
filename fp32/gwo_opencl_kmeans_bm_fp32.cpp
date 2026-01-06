// gwo_opencl_kmeans_bm.cpp
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>

// g++ -O2 -std=c++20 gwo_opencl_kmeans_bm.cpp -o opencl_kmeans.exe -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/lib/x64" -lOpenCL

// g++ -O2 -std=c++20 gwo_opencl_kmeans_bm.cpp -o opencl_kmeans -lOpenCL

// ===================== TARGET FILTER =====================
struct TargetDevice {
    std::string platform_sub;
    std::string device_sub;
};

static bool contains_icase(std::string hay, std::string needle) {
    auto tolow = [](unsigned char c){ return (char)std::tolower(c); };
    std::transform(hay.begin(), hay.end(), hay.begin(), tolow);
    std::transform(needle.begin(), needle.end(), needle.begin(), tolow);
    return hay.find(needle) != std::string::npos;
}

static bool is_target_device(const std::string& plat, const std::string& dev,
                             const std::vector<TargetDevice>& targets) {
    for (const auto& t : targets) {
        if (contains_icase(plat, t.platform_sub) && contains_icase(dev, t.device_sub))
            return true;
    }
    return false;
}

// ===================== MNIST IDX LOADER (embedded) =====================
static inline uint32_t read_be_u32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) throw std::runtime_error("Failed to read u32");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

struct MNIST {
    int n = 0, rows = 0, cols = 0;
    std::vector<float> X;
    std::vector<uint8_t> y;
    int dim() const { return rows * cols; }
};

static inline MNIST load_mnist_images_labels(const std::string& images_path,
                                             const std::string& labels_path,
                                             int limit = -1)
{
    std::ifstream fi(images_path, std::ios::binary);
    std::ifstream fl(labels_path, std::ios::binary);
    if (!fi) throw std::runtime_error("Cannot open images: " + images_path);
    if (!fl) throw std::runtime_error("Cannot open labels: " + labels_path);

    uint32_t magic_i = read_be_u32(fi);
    uint32_t n_i     = read_be_u32(fi);
    uint32_t rows    = read_be_u32(fi);
    uint32_t cols    = read_be_u32(fi);

    uint32_t magic_l = read_be_u32(fl);
    uint32_t n_l     = read_be_u32(fl);

    if (magic_i != 2051) throw std::runtime_error("Bad image magic (expected 2051)");
    if (magic_l != 2049) throw std::runtime_error("Bad label magic (expected 2049)");
    if (n_i != n_l) throw std::runtime_error("Image/label count mismatch");

    int n = (int)n_i;
    if (limit > 0 && limit < n) n = limit;

    MNIST ds;
    ds.n = n;
    ds.rows = (int)rows;
    ds.cols = (int)cols;
    const int D = ds.dim();

    ds.X.resize((size_t)n * (size_t)D);
    ds.y.resize((size_t)n);

    std::vector<uint8_t> buf((size_t)D);

    for (int i = 0; i < n; ++i) {
        fi.read(reinterpret_cast<char*>(buf.data()), D);
        if (!fi) throw std::runtime_error("Failed reading image data");
        for (int j = 0; j < D; ++j) ds.X[(size_t)i * D + j] = buf[j] / 255.0f;

        uint8_t lab;
        fl.read(reinterpret_cast<char*>(&lab), 1);
        if (!fl) throw std::runtime_error("Failed reading label data");
        ds.y[(size_t)i] = lab;
    }
    return ds;
}

// ===================== OpenCL helpers =====================
static void ocl_check(cl_int err, const char* where) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error " << err << " at " << where << "\n";
        throw std::runtime_error("OpenCL failure");
    }
}

static std::string get_platform_str(cl_platform_id p, cl_platform_info what) {
    size_t n = 0;
    clGetPlatformInfo(p, what, 0, nullptr, &n);
    std::string s(n, '\0');
    clGetPlatformInfo(p, what, n, s.data(), nullptr);
    while (!s.empty() && (s.back()=='\0' || s.back()=='\n' || s.back()=='\r')) s.pop_back();
    return s;
}

static std::string get_device_str(cl_device_id d, cl_device_info what) {
    size_t n = 0;
    clGetDeviceInfo(d, what, 0, nullptr, &n);
    std::string s(n, '\0');
    clGetDeviceInfo(d, what, n, s.data(), nullptr);
    while (!s.empty() && (s.back()=='\0' || s.back()=='\n' || s.back()=='\r')) s.pop_back();
    return s;
}

static bool device_supports_fp64(cl_device_id dev) {
    std::string ext = get_device_str(dev, CL_DEVICE_EXTENSIONS);
    return (ext.find("cl_khr_fp64") != std::string::npos);
}

static std::string read_text_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open kernel file: " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::string build_log(cl_program prog, cl_device_id dev) {
    size_t n = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
    std::string log(n, '\0');
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    return log;
}

static size_t round_up(size_t x, size_t m) {
    return ((x + m - 1) / m) * m;
}

// ===================== MINIMAL FIX #1: queue fallback (2.0 -> 1.2) =====================
static cl_command_queue create_queue_any(cl_context ctx, cl_device_id dev) {
    cl_int err = CL_SUCCESS;

    // Try OpenCL 2.0+ API first
    {
        const cl_queue_properties props[] = { 0 };
        cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
        if (q && err == CL_SUCCESS) return q;
    }

    // Fallback OpenCL 1.2 API
    err = CL_SUCCESS;
    cl_command_queue q12 = clCreateCommandQueue(ctx, dev, 0, &err);
    ocl_check(err, "clCreateCommandQueue (OpenCL 1.2 fallback)");
    return q12;
}

// ===================== platform/device listing =====================
static void list_platforms_devices() {
    cl_uint nplat = 0;
    ocl_check(clGetPlatformIDs(0, nullptr, &nplat), "clGetPlatformIDs(count)");
    std::vector<cl_platform_id> plats(nplat);
    ocl_check(clGetPlatformIDs(nplat, plats.data(), nullptr), "clGetPlatformIDs(list)");

    std::cout << "==== OpenCL Platforms/Devices ====\n";
    for (cl_uint pi = 0; pi < nplat; ++pi) {
        std::string plat_name = get_platform_str(plats[pi], CL_PLATFORM_NAME);
        std::string plat_ver  = get_platform_str(plats[pi], CL_PLATFORM_VERSION);
        std::cout << "[" << pi << "] Platform: " << plat_name << " | " << plat_ver << "\n";

        cl_uint ndev = 0;
        cl_int err = clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev);
        if (err != CL_SUCCESS || ndev == 0) {
            std::cout << "    (no devices)\n";
            continue;
        }
        std::vector<cl_device_id> devs(ndev);
        ocl_check(clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr), "clGetDeviceIDs(list)");

        for (cl_uint di = 0; di < ndev; ++di) {
            std::string dev_name = get_device_str(devs[di], CL_DEVICE_NAME);
            cl_device_type dtype{};
            clGetDeviceInfo(devs[di], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
            const char* t = (dtype & CL_DEVICE_TYPE_GPU) ? "GPU" :
                            (dtype & CL_DEVICE_TYPE_CPU) ? "CPU" : "OTHER";
            std::cout << "    (" << di << ") Device: " << dev_name << " [" << t << "]\n";
        }
    }
    std::cout << "=================================\n";
}

static bool pick_by_index(int platform_idx, int device_idx,
                          cl_platform_id& outPlat, cl_device_id& outDev,
                          std::string& outPlatName, std::string& outDevName, std::string& outDevVer)
{
    cl_uint nplat = 0;
    ocl_check(clGetPlatformIDs(0, nullptr, &nplat), "clGetPlatformIDs(count)");
    if (nplat == 0) return false;

    std::vector<cl_platform_id> plats(nplat);
    ocl_check(clGetPlatformIDs(nplat, plats.data(), nullptr), "clGetPlatformIDs(list)");

    if (platform_idx < 0 || platform_idx >= (int)nplat) return false;
    cl_platform_id plat = plats[(size_t)platform_idx];

    cl_uint ndev = 0;
    cl_int err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev);
    if (err != CL_SUCCESS || ndev == 0) return false;

    std::vector<cl_device_id> devs(ndev);
    ocl_check(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr), "clGetDeviceIDs(list)");

    if (device_idx < 0 || device_idx >= (int)ndev) return false;

    cl_device_id dev = devs[(size_t)device_idx];
    outPlat = plat;
    outDev  = dev;

    outPlatName = get_platform_str(plat, CL_PLATFORM_NAME);
    outDevName  = get_device_str(dev,  CL_DEVICE_NAME);
    outDevVer   = get_device_str(dev,  CL_DEVICE_VERSION);
    return true;
}

// ===================== main =====================
int main(int argc, char** argv) {
    try {
        // ===================== CLI =====================
        bool do_list = false;
        int platform_idx = -1;
        int device_idx   = -1;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--list") == 0) {
                do_list = true;
            } else if (std::strcmp(argv[i], "--platform") == 0 && i + 1 < argc) {
                platform_idx = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
                device_idx = std::atoi(argv[++i]);
            }
        }

        if (do_list) {
            list_platforms_devices();
            return 0;
        }

        // ===================== CONFIG =====================
        const int LIMIT_TRAIN = 5000;
        const int K = 10;
        const int RUNS = 1;
        const int MAX_ITERS = 100;
        const uint64_t SEED = 123456789ULL;

        std::vector<int> Pop_list = {25, 50, 100};

        // Fallback target filter (optional)
        std::vector<TargetDevice> targets = {
            {"NVIDIA CUDA", "RTX"},
            {"Intel(R) OpenCL Graphics", "Iris"},
            {"Portable Computing Language", "cpu"} // PoCL CPU
        };

        MNIST train = load_mnist_images_labels(
            "./mnist/train-images-idx3-ubyte",
            "./mnist/train-labels-idx1-ubyte",
            LIMIT_TRAIN
        );
        const int D = train.dim();
        const int Ndata = train.n;
        const int NDIM = K * D;

        std::cout << "Loaded MNIST: N=" << Ndata << ", D=" << D
                  << " (" << train.rows << "x" << train.cols
                  << "), NDIM=" << NDIM << "\n";

        std::string kernel_src = read_text_file("gwo_kmeans_fp32.cl");

        // ===================== CSV =====================
        std::string csv_name = "gwo_opencl.csv";
        bool need_header = (!std::filesystem::exists(csv_name) ||
                            std::filesystem::file_size(csv_name) == 0);

        std::ofstream csv(csv_name, std::ios::app);
        if (!csv) throw std::runtime_error("Cannot open csv: " + csv_name);

        if (need_header) {
            csv << "impl,limit_train,Ndata,D,K,POP_SIZE,max_iters,runs,avg_ms,best_sse,"
                   "platform,device,ocl_version,fp_mode,local_x,nGroupsX,platform_idx,device_idx\n";
        }

        // enumerate platforms
        cl_uint nplat = 0;
        ocl_check(clGetPlatformIDs(0, nullptr, &nplat), "clGetPlatformIDs(count)");
        std::vector<cl_platform_id> plats(nplat);
        ocl_check(clGetPlatformIDs(nplat, plats.data(), nullptr), "clGetPlatformIDs(list)");

        int ran_count = 0;

        auto run_one_device = [&](cl_platform_id plat, cl_device_id dev,
                                  const std::string& plat_name,
                                  const std::string& dev_name,
                                  const std::string& dev_ocl_ver,
                                  int pi_for_csv, int di_for_csv)
        {
            std::cout << "\n====================================================\n";
            std::cout << "Platform: " << plat_name << "\n";
            std::cout << "Device  : " << dev_name << "\n";
            std::cout << "OCL Ver : " << dev_ocl_ver << "\n";

            bool supports_fp64 = device_supports_fp64(dev);

            // Keep your policy: Iris -> FP32
            bool force_fp32 = contains_icase(plat_name, "Intel(R) OpenCL Graphics") && contains_icase(dev_name, "Iris");
            bool use_fp64 = false; // FP32-only benchmark

            std::cout << "FP64 ext : " << (supports_fp64 ? "YES" : "NO")
                      << " | use_fp64=" << (use_fp64 ? 1 : 0) << "\n";

            cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0 };
            cl_int ectx = 0;
            cl_context ctx = clCreateContext(props, 1, &dev, nullptr, nullptr, &ectx);
            if (!ctx || ectx != CL_SUCCESS) {
                std::cout << "Skip: cannot create context\n";
                return;
            }

            // MINIMAL FIX #1 applied here
            cl_command_queue q = create_queue_any(ctx, dev);

            const char* srcp = kernel_src.c_str();
            size_t srclen = kernel_src.size();
            cl_int eprg = 0;
            cl_program prog = clCreateProgramWithSource(ctx, 1, &srcp, &srclen, &eprg);
            if (!prog || eprg != CL_SUCCESS) {
                std::cout << "Skip: cannot create program\n";
                clReleaseCommandQueue(q);
                clReleaseContext(ctx);
                return;
            }

            // ===================== MINIMAL FIX #2: remove -cl-std=CL2.0 =====================
            // Just don't force OpenCL C version. PoCL will use OpenCL C 1.2 and succeed.
            std::string options = std::string("-D K_CONST=10 -D USE_FP64=") + (use_fp64 ? "1" : "0");

            cl_int eb = clBuildProgram(prog, 1, &dev, options.c_str(), nullptr, nullptr);
            if (eb != CL_SUCCESS) {
                std::cout << "Build failed:\n" << build_log(prog, dev) << "\n";
                clReleaseProgram(prog);
                clReleaseCommandQueue(q);
                clReleaseContext(ctx);
                return;
            }

            // kernels
            cl_int ek = 0;
            cl_kernel k_init = clCreateKernel(prog, "init_pos", &ek); ocl_check(ek, "create init_pos");
            cl_kernel k_ssep = clCreateKernel(prog, "sse_partial", &ek); ocl_check(ek, "create sse_partial");
            cl_kernel k_sser = clCreateKernel(prog, "sse_reduce", &ek); ocl_check(ek, "create sse_reduce");
            cl_kernel k_r1   = clCreateKernel(prog, "reduce_top3_stage1", &ek); ocl_check(ek, "create reduce1");
            cl_kernel k_r2   = clCreateKernel(prog, "reduce_top3_stage2", &ek); ocl_check(ek, "create reduce2");
            cl_kernel k_gath = clCreateKernel(prog, "gather_leaders", &ek); ocl_check(ek, "create gather");
            cl_kernel k_upd  = clCreateKernel(prog, "gwo_update", &ek); ocl_check(ek, "create update");

            // choose LOCAL_X safely
            size_t maxWg = 0;
            clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWg, nullptr);

            size_t LOCAL_X = 64;
            if (maxWg > 0) LOCAL_X = std::min(LOCAL_X, maxWg);
            LOCAL_X = std::max((size_t)16, LOCAL_X);

            int nGroupsX = (Ndata + (int)LOCAL_X - 1) / (int)LOCAL_X;

            const size_t realSize = use_fp64 ? sizeof(double) : sizeof(float);

            // X buffer
            cl_int em = 0;
            cl_mem dX = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * (size_t)Ndata * (size_t)D,
                (void*)train.X.data(), &em);
            ocl_check(em, "create dX");

            for (int POP : Pop_list) {
                std::cout << "--------------------------------------------\n";
                std::cout << "OpenCL FULL | POP=" << POP
                          << " | ITERS=" << MAX_ITERS
                          << " | nGroupsX=" << nGroupsX
                          << " | FP=" << (use_fp64 ? "FP64" : "FP32")
                          << " | LOCAL_X=" << LOCAL_X
                          << " | RUNS=" << RUNS << "\n";

                const size_t posBytes = realSize * (size_t)POP * (size_t)NDIM;

                cl_mem dPos = clCreateBuffer(ctx, CL_MEM_READ_WRITE, posBytes, nullptr, &em);
                ocl_check(em, "create dPos");

                cl_mem dPartial = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                    realSize * (size_t)POP * (size_t)nGroupsX, nullptr, &em);
                ocl_check(em, "create dPartial");

                cl_mem dSSE = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                    realSize * (size_t)POP, nullptr, &em);
                ocl_check(em, "create dSSE");

                size_t r1_global = round_up((size_t)POP, LOCAL_X);
                size_t r1_groups = r1_global / LOCAL_X;
                int N3 = (int)(r1_groups * 3);

                cl_mem dCand = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                    (size_t)N3 * realSize * 2, nullptr, &em);
                ocl_check(em, "create dCand");

                cl_mem dBest3 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                    sizeof(int) * 3, nullptr, &em);
                ocl_check(em, "create dBest3");

                cl_mem dL0 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, realSize * (size_t)NDIM, nullptr, &em);
                cl_mem dL1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, realSize * (size_t)NDIM, nullptr, &em);
                cl_mem dL2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, realSize * (size_t)NDIM, nullptr, &em);
                ocl_check(em, "create leaders");

                long long total_ms = 0;

                for (int r = 1; r <= RUNS; ++r) {
                    // init_pos
                    {
                        cl_ulong seed = (cl_ulong)SEED;
                        ocl_check(clSetKernelArg(k_init, 0, sizeof(cl_mem), &dPos), "arg init 0");
                        ocl_check(clSetKernelArg(k_init, 1, sizeof(int), &POP),   "arg init 1");
                        ocl_check(clSetKernelArg(k_init, 2, sizeof(int), &NDIM),  "arg init 2");
                        ocl_check(clSetKernelArg(k_init, 3, sizeof(cl_ulong), &seed), "arg init 3");

                        size_t g = (size_t)POP * (size_t)NDIM;
                        ocl_check(clEnqueueNDRangeKernel(q, k_init, 1, nullptr, &g, nullptr, 0, nullptr, nullptr),
                                  "enqueue init_pos");
                        ocl_check(clFinish(q), "finish init_pos");
                    }

                    auto t0 = std::chrono::steady_clock::now();

                    for (int iter = 0; iter < MAX_ITERS; ++iter) {
                        // sse_partial
                        {
                            ocl_check(clSetKernelArg(k_ssep, 0, sizeof(cl_mem), &dX), "arg ssep 0");
                            ocl_check(clSetKernelArg(k_ssep, 1, sizeof(int), &Ndata), "arg ssep 1");
                            ocl_check(clSetKernelArg(k_ssep, 2, sizeof(int), &D), "arg ssep 2");
                            ocl_check(clSetKernelArg(k_ssep, 3, sizeof(cl_mem), &dPos), "arg ssep 3");
                            ocl_check(clSetKernelArg(k_ssep, 4, sizeof(int), &K), "arg ssep 4");
                            ocl_check(clSetKernelArg(k_ssep, 5, sizeof(int), &POP), "arg ssep 5");
                            ocl_check(clSetKernelArg(k_ssep, 6, sizeof(int), &nGroupsX), "arg ssep 6");
                            ocl_check(clSetKernelArg(k_ssep, 7, sizeof(cl_mem), &dPartial), "arg ssep 7");
                            ocl_check(clSetKernelArg(k_ssep, 8, realSize * LOCAL_X, nullptr), "arg ssep 8");

                            size_t global[2] = { (size_t)nGroupsX * LOCAL_X, (size_t)POP };
                            size_t local[2]  = { LOCAL_X, 1 };
                            ocl_check(clEnqueueNDRangeKernel(q, k_ssep, 2, nullptr, global, local, 0, nullptr, nullptr),
                                      "enqueue sse_partial");
                        }

                        // sse_reduce
                        {
                            ocl_check(clSetKernelArg(k_sser, 0, sizeof(cl_mem), &dPartial), "arg sser 0");
                            ocl_check(clSetKernelArg(k_sser, 1, sizeof(int), &POP), "arg sser 1");
                            ocl_check(clSetKernelArg(k_sser, 2, sizeof(int), &nGroupsX), "arg sser 2");
                            ocl_check(clSetKernelArg(k_sser, 3, sizeof(cl_mem), &dSSE), "arg sser 3");

                            size_t g = (size_t)POP;
                            ocl_check(clEnqueueNDRangeKernel(q, k_sser, 1, nullptr, &g, nullptr, 0, nullptr, nullptr),
                                      "enqueue sse_reduce");
                        }

                        // reduce1
                        {
                            ocl_check(clSetKernelArg(k_r1, 0, sizeof(cl_mem), &dSSE), "arg r1 0");
                            ocl_check(clSetKernelArg(k_r1, 1, sizeof(int), &POP), "arg r1 1");
                            ocl_check(clSetKernelArg(k_r1, 2, sizeof(cl_mem), &dCand), "arg r1 2");

                            size_t localBytes = realSize * 2 * LOCAL_X;
                            ocl_check(clSetKernelArg(k_r1, 3, localBytes, nullptr), "arg r1 3");
                            ocl_check(clSetKernelArg(k_r1, 4, localBytes, nullptr), "arg r1 4");
                            ocl_check(clSetKernelArg(k_r1, 5, localBytes, nullptr), "arg r1 5");

                            size_t g = r1_global;
                            size_t l = LOCAL_X;
                            ocl_check(clEnqueueNDRangeKernel(q, k_r1, 1, nullptr, &g, &l, 0, nullptr, nullptr),
                                      "enqueue reduce1");
                        }

                        // reduce2
                        {
                            ocl_check(clSetKernelArg(k_r2, 0, sizeof(cl_mem), &dCand), "arg r2 0");
                            ocl_check(clSetKernelArg(k_r2, 1, sizeof(int), &N3), "arg r2 1");
                            ocl_check(clSetKernelArg(k_r2, 2, sizeof(cl_mem), &dBest3), "arg r2 2");

                            size_t localBytes = realSize * 2 * LOCAL_X;
                            ocl_check(clSetKernelArg(k_r2, 3, localBytes, nullptr), "arg r2 3");
                            ocl_check(clSetKernelArg(k_r2, 4, localBytes, nullptr), "arg r2 4");
                            ocl_check(clSetKernelArg(k_r2, 5, localBytes, nullptr), "arg r2 5");

                            size_t g = LOCAL_X;
                            size_t l = LOCAL_X;
                            ocl_check(clEnqueueNDRangeKernel(q, k_r2, 1, nullptr, &g, &l, 0, nullptr, nullptr),
                                      "enqueue reduce2");
                        }

                        // gather
                        {
                            ocl_check(clSetKernelArg(k_gath, 0, sizeof(cl_mem), &dPos), "arg gath 0");
                            ocl_check(clSetKernelArg(k_gath, 1, sizeof(int), &NDIM), "arg gath 1");
                            ocl_check(clSetKernelArg(k_gath, 2, sizeof(cl_mem), &dBest3), "arg gath 2");
                            ocl_check(clSetKernelArg(k_gath, 3, sizeof(cl_mem), &dL0), "arg gath 3");
                            ocl_check(clSetKernelArg(k_gath, 4, sizeof(cl_mem), &dL1), "arg gath 4");
                            ocl_check(clSetKernelArg(k_gath, 5, sizeof(cl_mem), &dL2), "arg gath 5");

                            size_t g = (size_t)NDIM;
                            ocl_check(clEnqueueNDRangeKernel(q, k_gath, 1, nullptr, &g, nullptr, 0, nullptr, nullptr),
                                      "enqueue gather");
                        }

                        // update
                        {
                            float a_host = 2.0f * (1.0f - (float)iter / (float)MAX_ITERS);
                            cl_ulong seed = (cl_ulong)SEED;

                            ocl_check(clSetKernelArg(k_upd, 0, sizeof(cl_mem), &dPos), "arg upd 0");
                            ocl_check(clSetKernelArg(k_upd, 1, sizeof(int), &POP), "arg upd 1");
                            ocl_check(clSetKernelArg(k_upd, 2, sizeof(int), &NDIM), "arg upd 2");
                            ocl_check(clSetKernelArg(k_upd, 3, sizeof(cl_mem), &dL0), "arg upd 3");
                            ocl_check(clSetKernelArg(k_upd, 4, sizeof(cl_mem), &dL1), "arg upd 4");
                            ocl_check(clSetKernelArg(k_upd, 5, sizeof(cl_mem), &dL2), "arg upd 5");

                            if (use_fp64) {
                                float a = a_host;
                                ocl_check(clSetKernelArg(k_upd, 6, sizeof(float), &a), "arg upd 6 fp32");
                            } else {
                                float a = (float)a_host;
                                ocl_check(clSetKernelArg(k_upd, 6, sizeof(float), &a), "arg upd 6 fp32");
                            }

                            ocl_check(clSetKernelArg(k_upd, 7, sizeof(int), &iter), "arg upd 7");
                            ocl_check(clSetKernelArg(k_upd, 8, sizeof(cl_ulong), &seed), "arg upd 8");

                            size_t global[2] = { (size_t)POP, (size_t)NDIM };
                            ocl_check(clEnqueueNDRangeKernel(q, k_upd, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                                      "enqueue update");
                        }
                    }

                    ocl_check(clFinish(q), "finish all iters");

                    auto t1 = std::chrono::steady_clock::now();
                    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                    total_ms += ms;

                    int best3[3] = {-1,-1,-1};
                    ocl_check(clEnqueueReadBuffer(q, dBest3, CL_TRUE, 0, sizeof(best3), best3, 0, nullptr, nullptr),
                              "read best3");

                    float bestSSE = 0.0f;
                    if (use_fp64) {
                        float v = 0.0f;
                        ocl_check(clEnqueueReadBuffer(q, dSSE, CL_TRUE,
                            (size_t)best3[0] * sizeof(float), sizeof(float), &v, 0, nullptr, nullptr),
                            "read sse(best) fp64");
                        bestSSE = v;
                    } else {
                        float v = 0.0f;
                        ocl_check(clEnqueueReadBuffer(q, dSSE, CL_TRUE,
                            (size_t)best3[0] * sizeof(float), sizeof(float), &v, 0, nullptr, nullptr),
                            "read sse(best) fp32");
                        bestSSE = v;
                    }

                    std::cout << "Run " << r
                              << " | " << ms << " ms"
                              << " | best SSE=" << bestSSE
                              << " | idx0=" << best3[0] << "\n";

                    double avg_ms = total_ms / double(RUNS);

                    csv << "opencl"
                        << "," << LIMIT_TRAIN
                        << "," << Ndata
                        << "," << D
                        << "," << K
                        << "," << POP
                        << "," << MAX_ITERS
                        << "," << RUNS
                        << "," << avg_ms
                        << "," << bestSSE
                        << "," << '"' << plat_name << '"'
                        << "," << '"' << dev_name << '"'
                        << "," << '"' << dev_ocl_ver << '"'
                        << "," << (use_fp64 ? "FP64" : "FP32")
                        << "," << LOCAL_X
                        << "," << nGroupsX
                        << "," << pi_for_csv
                        << "," << di_for_csv
                        << "\n";
                    csv.flush();
                }

                clReleaseMemObject(dL0); clReleaseMemObject(dL1); clReleaseMemObject(dL2);
                clReleaseMemObject(dBest3);
                clReleaseMemObject(dCand);
                clReleaseMemObject(dSSE);
                clReleaseMemObject(dPartial);
                clReleaseMemObject(dPos);
            }

            clReleaseMemObject(dX);

            clReleaseKernel(k_init);
            clReleaseKernel(k_ssep);
            clReleaseKernel(k_sser);
            clReleaseKernel(k_r1);
            clReleaseKernel(k_r2);
            clReleaseKernel(k_gath);
            clReleaseKernel(k_upd);

            clReleaseProgram(prog);
            clReleaseCommandQueue(q);
            clReleaseContext(ctx);

            ran_count++;
        };

        // OPTION 1: explicit --platform --device
        if (platform_idx >= 0 && device_idx >= 0) {
            cl_platform_id plat = nullptr;
            cl_device_id dev = nullptr;
            std::string plat_name, dev_name, dev_ocl_ver;

            if (!pick_by_index(platform_idx, device_idx, plat, dev, plat_name, dev_name, dev_ocl_ver)) {
                throw std::runtime_error("Invalid --platform/--device index. Use --list to see valid indices.");
            }

            run_one_device(plat, dev, plat_name, dev_name, dev_ocl_ver, platform_idx, device_idx);

            if (ran_count == 0) {
                std::cout << "\nSelected device not runnable.\n";
            }
            csv.close();
            return 0;
        }

        // OPTION 2: your target filter behavior
        for (cl_uint pi = 0; pi < nplat; ++pi) {
            cl_platform_id plat = plats[pi];

            cl_uint ndev = 0;
            cl_int err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev);
            if (err != CL_SUCCESS || ndev == 0) continue;

            std::vector<cl_device_id> devs(ndev);
            ocl_check(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr),
                      "clGetDeviceIDs(list)");

            std::string plat_name = get_platform_str(plat, CL_PLATFORM_NAME);

            for (cl_uint di = 0; di < ndev; ++di) {
                cl_device_id dev = devs[di];
                std::string dev_name = get_device_str(dev, CL_DEVICE_NAME);

                if (!is_target_device(plat_name, dev_name, targets))
                    continue;

                std::string dev_ocl_ver = get_device_str(dev, CL_DEVICE_VERSION);

                run_one_device(plat, dev, plat_name, dev_name, dev_ocl_ver, (int)pi, (int)di);
            }
        }

        if (ran_count == 0) {
            std::cout << "\nNo target devices found.\n";
            std::cout << "Tip: run with --list to see what platforms/devices exist.\n";
        }

        csv.close();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}

