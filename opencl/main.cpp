// g++ main.cpp -o main.exe -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/lib/x64" -lOpenCL
#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>

std::string loadKernel(const char* filename) {
    std::ifstream file(filename);
    return std::string(std::istreambuf_iterator<char>(file), {});
}

int main() {
    // Data
    const int N = 5;
    float a[N] = {1,2,3,4,5};
    float b[N] = {10,20,30,40,50};
    float c[N];

    cl_uint platformCount;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, &platformCount);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue_properties props[] = { 0 }; // hoặc danh sách properties, kết thúc bằng 0
    cl_int err;
    
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);

    // Load kernel
    std::string kernelSource = loadKernel("kernel.cl");
    const char* kernelStr = kernelSource.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernelStr, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "add", nullptr);

    // Buffers
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(a), a, nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(b), b, nullptr);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(c), nullptr, nullptr);

    // Set args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    size_t globalSize = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(c), c, 0, nullptr, nullptr);

    // Output
    for (int i = 0; i < N; i++)
        std::cout << c[i] << " ";

    std::cout << "\n";
}

