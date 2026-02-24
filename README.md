# CUDA Matrix Multiplication: Naive vs Tiled.

GPU Programming: Tiling - Demonstrating how CUDA naive kernel vs Tiling approach differs in computational overhead
for matrix multiplication, by reducing global memory workload. Typically, the naive model calls the first
row of matrix A and the first colunm of matrix B, loads them into the register, and produces a sinlge output
for matrix C at the relevant co-ordinate i.e. C(0, 0). When repeating this process for C(0, 1), the naive kernel
reads the entire set of matrix A+1 and B+1 again, loads them into the register, and output the sinlge Ci value -
this read/write process repeats for the entirety of C across both A & B, with the number of read/write accesses being equal to the length of the side of the A/B matrix.

When utilizing shared memory, the tiling implementation removes this overhead by segmenting the existing A and B matrices into blocks, loading portions of those matrices into shared memory, and drawing from shared memory to compute the corresponding output of C. While the time complexity of both approaches remains O(n^3), tiling reduces global memory access complexity from O(n^3) to O(n^3 / tile_size). 

## What This Project Does

This project benchmarks two CUDA kernels for square matrix multiplication:

- `MatrixMulNaive`: each thread computes one output value and repeatedly reads from global memory.
- `MatrixMulTiled`: each block uses shared memory tiles to reuse data and reduce global memory traffic.

The executable reports:

- average kernel time for both kernels
- throughput in GFLOP/s
- speedup ratio (`naive / tiled`)

## Results

Across increasing matrix sizes (128 → 2048), the tiled kernel consistently outperforms the naive implementation once the workload becomes memory-bound. For small matrices the difference is negligible and can even slightly favor the naive version due to shared memory overhead, but from 256 onward the tiled approach stabilizes at roughly 1.3x speedup. Kernel time scales as expected with O(n^3) growth for both methods. However, the tiled kernel maintains higher sustained throughput, reaching ~1200+ GFLOP/s compared to ~950 GFLOP/s for the naive version at larger sizes. 

<img width="2400" height="720" alt="cuda_matmul_plot" src="https://github.com/user-attachments/assets/9114cc47-9cb3-470c-a635-3c7fc9d3f4ba" />

The performance gap reflects reduced global memory pressure and improved data reuse inside shared memory.

## Future work

Next week I will extend the tiling comparison beyond matmul by applying the same approach to grid-based numerical computation. The objective is to evaluate whether shared memory produces measurable improvement if there is overlap between neighboring grid threads accessing the same data.

## Build And Run

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 matmul_benchmark.cu -o matmul_benchmark_cuda
./matmul_benchmark_cuda 1024 20
```

Arguments:

- first arg: matrix width (`N` for `N x N`)
- second arg: number of timed runs

## Plotting Performance

Use the helper script to benchmark multiple sizes and generate a PNG and CSV:

```bash
python3 plot_cuda_benchmark.py --runs 20
```

## VS Code Tasks (`.vscode/tasks.json`)

The task file contains:

- `Build CUDA Benchmark`: compiles `matmul_benchmark.cu` with `nvcc`
- `Run CUDA Benchmark`: builds, then runs one benchmark config
- `Plot CUDA Benchmark`: builds, then runs `plot_cuda_benchmark.py`
