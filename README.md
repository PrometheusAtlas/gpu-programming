# CUDA Matrix Multiplication: Naive vs Tiled

GPU Programming: Tiling - Demonstrating how CUDA naive kernel vs Tiling approach differs in computational overhead
for matrix multiplication, by reducing the global memory workload. Typically, the naive model calls the first row
row of matrix A and the first colunm of matrix B, and loads them into rgiesters, This produces a sinlge output
for matrix C at the relevant co-ordinate i.e. C(0, 0). When repeating this process for C(0, 1), the naive kernel
reads the entire set of matrix A+1 and B+1 again, loads them into the registers, and output the sinlge Ci value -
this read/write process repeats for the entirety of C with the number of access being = to the length of the side of the matrix.
By utililizing shared memory, the Tiling apprach removes this overhead by segmenting the exisintg A & B matrices
into blocks (a mini matrix of matrix), loading a portion of those matrices into shared memeory and procedding to draw on the cordinate
from shared memeory for the out of C. While time compleixty of both approaches remain O(n^3), Tiling reduces global memory complexity
from O(n^3) to O(n^3/tile_size).

## What This Project Does

This project benchmarks two CUDA kernels for square matrix multiplication:

- `MatrixMulNaive`: each thread computes one output value and repeatedly reads from global memory.
- `MatrixMulTiled`: each block uses shared memory tiles to reuse data and reduce global memory traffic.

The executable reports:

- average kernel time for both kernels
- throughput in GFLOP/s
- speedup ratio (`naive / tiled`)

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
