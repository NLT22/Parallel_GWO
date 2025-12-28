# Makefile (Linux/Ubuntu)
CXX      := g++
NVCC     := nvcc

CXXFLAGS := -O2 -std=c++20
EIGEN_INC:= ../eigen-5.0.0

# OpenCL (Linux): thường chỉ cần -lOpenCL
OCL_LIBS := -lOpenCL

# OpenMP
OMPFLAGS := -fopenmp

# CUDA arch (đổi nếu GPU khác)
CUDA_ARCH := -gencode arch=compute_86,code=sm_86

BIN_DIR := bin

OPENCL_SRC  := gwo_opencl_kmeans_bm.cpp
PAR_SRC     := gwo_parallel_kmeans_bm.cpp
SER_SRC     := gwo_serial_kmeans_bm.cpp
CUDA_SRC    := gwo_cuda_kmeans_bm.cu

OPENCL_BIN  := $(BIN_DIR)/opencl_kmeans
PAR_BIN     := $(BIN_DIR)/parallel_kmeans
SER_BIN     := $(BIN_DIR)/serial_kmeans
CUDA_BIN    := $(BIN_DIR)/cuda_kmeans

.PHONY: all clean run list help

all: $(OPENCL_BIN) $(PAR_BIN) $(SER_BIN) $(CUDA_BIN)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(OPENCL_BIN): $(OPENCL_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OCL_LIBS)

$(PAR_BIN): $(PAR_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ -I $(EIGEN_INC)

$(SER_BIN): $(SER_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ -I $(EIGEN_INC)

$(CUDA_BIN): $(CUDA_SRC) | $(BIN_DIR)
	$(NVCC) -O2 -std=c++20 $(CUDA_ARCH) -o $@ $<

clean:
	rm -rf $(BIN_DIR)

# list device OpenCL
list: $(OPENCL_BIN)
	$(OPENCL_BIN) --list

# chạy auto (tùy ý sửa args)
run: all
	@echo "=== SERIAL ==="
	$(SER_BIN)
	@echo "=== OPENMP ==="
	$(PAR_BIN)
	@echo "=== OPENCL list ==="
	$(OPENCL_BIN) --list
	@echo "=== OPENCL run NVIDIA (vd platform=1 device=0) ==="
	$(OPENCL_BIN) --platform 1 --device 0
	@echo "=== OPENCL run PoCL CPU (vd platform=2 device=0) ==="
	$(OPENCL_BIN) --platform 2 --device 0
	@echo "=== CUDA ==="
	$(CUDA_BIN)

help:
	@echo "make all     : build all binaries"
	@echo "make run     : build + run everything"
	@echo "make list    : OpenCL --list"
	@echo "make clean   : remove bin/"

