#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

// GPU Programming: Tiling - Demonstrating how CUDA naive kernel vs Tiling approach differs in computational overhead
// for matrix multiplication, by reducing the global memory workload. Typically, the naive model calls the first row
// row of matrix A and the first colunm of matrix B, and loads them into rgiesters, This produces a sinlge output
// for matrix C at the relevant co-ordinate i.e. C(0, 0). When repeating this process for C(0, 1), the naive kernel
// reads the entire set of matrix A+1 and B+1 again, loads them into the registers, and output the sinlge Ci value -
// this read/write process repeats for the entirety of C with the number of access being = to the length of the side of the matrix. 
// By utililizing shared memory, the Tiling apprach removes this overhead by segmenting the exisintg A & B matrices
// into blocks (a mini matrix of matrix), loading a portion of those matrices into shared memeory and procedding to draw on the cordinate
// from shared memeory for the out of C. While time compleixty of both approaches remain O(n^3), Tiling reduces global memory complexity
// from O(n^3) to O(n^3/tile_size). 
//  

// ########################## EXPERIMENT SETTINGS ############################

// Tile width/height used by the shared-memory tiled kernel.
#define TILE_SIZE 16


// ############################ CUDA ERROR CHECK #############################

// Wrap every CUDA runtime call so failures stop immediately with line info.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ########################## KERNEL: NAIVE MATMUL ############################

// Each thread computes one output element C[row, col].
// Direct global memory reads in the inner K loop - no shared memeory:
__global__ void MatrixMulNaive(const float* d_a, const float* d_b, float* d_c, int width) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= width || col >= width) {
        return;
    }

    float val = 0.0f;
    for (int k = 0; k < width; k++) {
        val += d_a[row * width + k] * d_b[k * width + col];
    }
    d_c[row * width + col] = val;
}

// ########################## KERNEL: TILED MATMUL ############################

// Each block computes the TILE_SIZE x TILE_SIZE output tile.
// A and B sub-tiles/min matrices are loaded into shared memory then reused, discaredm, and replaced by the next segment of the mini matrix:
__global__ void MatrixMulTiled(const float* d_a, const float* d_b, float* d_c, int width) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    const int num_tiles = (width + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        const int tiled_col = t * TILE_SIZE + threadIdx.x;
        const int tiled_row = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < width && tiled_col < width) ? d_a[row * width + tiled_col] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (tiled_row < width && col < width) ? d_b[tiled_row * width + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            val += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        d_c[row * width + col] = val;
    }
}

// ######################### BENCHMARK: NAIVE KERNEL ##############################

// Launch the naive kernel - runs times and return average kernel time (ms).
static float benchmark_naive(const float* d_a,
                             const float* d_b,
                             float* d_c,
                             int width,
                             int runs,
                             dim3 grid,
                             dim3 block) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        MatrixMulNaive<<<grid, block>>>(d_a, d_b, d_c, width);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(runs);
}


// ######################### BENCHMARK: TILED KERNEL ##############################

// Launch the tiled kernel - runs times and return average kernel time (ms).
static float benchmark_tiled(const float* d_a,
                             const float* d_b,
                             float* d_c,
                             int width,
                             int runs,
                             dim3 grid,
                             dim3 block) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        MatrixMulTiled<<<grid, block>>>(d_a, d_b, d_c, width);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(runs);
}

// ################################## MAIN ####################################

int main(int argc, char** argv) {
    // Defaults:
    //   width  = matrix is width x width
    //   runs   = number of benchmark launches for each kernel
    int width = 1024;
    int runs = 20;

    // ------------------------ Parse CLI arguments -------------------------
    if (argc > 1) {
        width = std::atoi(argv[1]);
    }
    if (argc > 2) {
        runs = std::atoi(argv[2]);
    }
    if (width <= 0 || runs <= 0) {
        std::cerr << "Usage: ./matmul_benchmark_cuda [width] [runs]" << std::endl;
        return 1;
    }

    // ---------------------- Host memory allocation ------------------------
    const std::size_t elem_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(width);
    const std::size_t bytes = elem_count * sizeof(float);

    std::vector<float> h_a(elem_count);
    std::vector<float> h_b(elem_count);

    // Random input initialization (deterministic seed for reproducibility).
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < elem_count; i++) {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    // ---------------------- Device memory allocation ----------------------
    float *d_a = nullptr, *d_b = nullptr, *d_c_naive = nullptr, *d_c_tiled = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_naive, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_tiled, bytes));

    // Copy inputs to GPU.
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // 2D launch setup:
    // - block: TILE_SIZE x TILE_SIZE threads
    // - grid: enough blocks to cover width x width output elements
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (width + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up launches (helps reduce first-launch overhead in timing).
    MatrixMulNaive<<<grid, block>>>(d_a, d_b, d_c_naive, width);
    MatrixMulTiled<<<grid, block>>>(d_a, d_b, d_c_tiled, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------------- Timed benchmark section -----------------------
    const float naive_ms = benchmark_naive(d_a, d_b, d_c_naive, width, runs, grid, block);
    const float tiled_ms = benchmark_tiled(d_a, d_b, d_c_tiled, width, runs, grid, block);

    // Throughput metric:
    // FLOPs for NxN GEMM = 2 * N^3
    const double flops = 2.0 * static_cast<double>(width) * width * width;
    const double naive_gflops = (flops / (naive_ms / 1000.0)) / 1e9;
    const double tiled_gflops = (flops / (tiled_ms / 1000.0)) / 1e9;

    // --------------------------- Print report -----------------------------
    std::cout << "Width: " << width << " x " << width << "\n";
    std::cout << "Tile size: " << TILE_SIZE << " x " << TILE_SIZE << "\n";
    std::cout << "Runs: " << runs << "\n";
    std::cout << "Naive avg kernel time: " << naive_ms << " ms\n";
    std::cout << "Naive throughput: " << naive_gflops << " GFLOP/s\n";
    std::cout << "Tiled avg kernel time: " << tiled_ms << " ms\n";
    std::cout << "Tiled throughput: " << tiled_gflops << " GFLOP/s\n";
    if (tiled_ms > 0.0f) {
        std::cout << "Speedup (naive / tiled): " << (naive_ms / tiled_ms) << "x\n";
    }

    // ------------------------------ Cleanup -------------------------------
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c_naive));
    CUDA_CHECK(cudaFree(d_c_tiled));

    return 0;
}
