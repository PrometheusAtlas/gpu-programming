#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NAIVE_MS_RE = re.compile(r"Naive avg kernel time:\s*([0-9.eE+-]+)\s*ms")
TILED_MS_RE = re.compile(r"Tiled avg kernel time:\s*([0-9.eE+-]+)\s*ms")
NAIVE_GFLOPS_RE = re.compile(r"Naive throughput:\s*([0-9.eE+-]+)\s*GFLOP/s")
TILED_GFLOPS_RE = re.compile(r"Tiled throughput:\s*([0-9.eE+-]+)\s*GFLOP/s")
SPEEDUP_RE = re.compile(r"Speedup \(naive / tiled\):\s*([0-9.eE+-]+)x")


def parse_value(pattern: re.Pattern[str], text: str, label: str) -> float:
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not parse {label} from benchmark output.")
    return float(match.group(1))


def run_one(binary: Path, size: int, runs: int) -> dict:
    cmd = [str(binary), str(size), str(runs)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed for size={size}, runs={runs}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    output = proc.stdout
    return {
        "size": size,
        "runs": runs,
        "naive_ms": parse_value(NAIVE_MS_RE, output, "naive_ms"),
        "tiled_ms": parse_value(TILED_MS_RE, output, "tiled_ms"),
        "naive_gflops": parse_value(NAIVE_GFLOPS_RE, output, "naive_gflops"),
        "tiled_gflops": parse_value(TILED_GFLOPS_RE, output, "tiled_gflops"),
        "speedup": parse_value(SPEEDUP_RE, output, "speedup"),
    }


def write_csv(results: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "size",
        "runs",
        "naive_ms",
        "tiled_ms",
        "naive_gflops",
        "tiled_gflops",
        "speedup",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def write_plot(results: list[dict], png_path: Path) -> None:
    sizes = [r["size"] for r in results]
    naive_ms = [r["naive_ms"] for r in results]
    tiled_ms = [r["tiled_ms"] for r in results]
    naive_gflops = [r["naive_gflops"] for r in results]
    tiled_gflops = [r["tiled_gflops"] for r in results]
    speedups = [r["speedup"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(sizes, naive_ms, marker="o", linewidth=2, label="Naive")
    axes[0].plot(sizes, tiled_ms, marker="o", linewidth=2, label="Tiled")
    axes[0].set_title("Kernel Time (Lower is Better)")
    axes[0].set_xlabel("Matrix Size (N for NxN)")
    axes[0].set_ylabel("Time (ms)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(sizes, naive_gflops, marker="o", linewidth=2, label="Naive")
    axes[1].plot(sizes, tiled_gflops, marker="o", linewidth=2, label="Tiled")
    axes[1].set_title("Throughput (Higher is Better)")
    axes[1].set_xlabel("Matrix Size (N for NxN)")
    axes[1].set_ylabel("GFLOP/s")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(sizes, speedups, marker="o", linewidth=2, color="tab:green")
    axes[2].axhline(1.0, linestyle="--", linewidth=1, color="gray")
    axes[2].set_title("Speedup (Naive / Tiled)")
    axes[2].set_xlabel("Matrix Size (N for NxN)")
    axes[2].set_ylabel("x")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("CUDA Matmul: Naive vs Tiled")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def parse_sizes(raw: str) -> list[int]:
    sizes = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        val = int(item)
        if val <= 0:
            raise ValueError(f"Invalid size: {item}")
        sizes.append(val)
    if not sizes:
        raise ValueError("No valid sizes provided.")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CUDA matmul benchmark across sizes and plot naive vs tiled."
    )
    parser.add_argument(
        "--binary",
        default="./matmul_benchmark_cuda",
        help="Path to benchmark binary (default: ./matmul_benchmark_cuda)",
    )
    parser.add_argument(
        "--sizes",
        default="128,256,512,1024,1536,2048",
        help="Comma-separated matrix sizes (default: 128,256,512,1024,1536,2048)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Kernel launches averaged inside each benchmark run (default: 20)",
    )
    parser.add_argument(
        "--csv-out",
        default="cuda_matmul_results.csv",
        help="Output CSV path (default: cuda_matmul_results.csv)",
    )
    parser.add_argument(
        "--plot-out",
        default="cuda_matmul_plot.png",
        help="Output PNG path (default: cuda_matmul_plot.png)",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be > 0")

    binary = Path(args.binary).resolve()
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

    sizes = parse_sizes(args.sizes)
    results = []
    for size in sizes:
        row = run_one(binary, size, args.runs)
        results.append(row)
        print(
            f"size={size}: naive={row['naive_ms']:.4f} ms, "
            f"tiled={row['tiled_ms']:.4f} ms, speedup={row['speedup']:.3f}x"
        )

    csv_path = Path(args.csv_out).resolve()
    png_path = Path(args.plot_out).resolve()
    write_csv(results, csv_path)
    write_plot(results, png_path)

    print(f"CSV saved: {csv_path}")
    print(f"Plot saved: {png_path}")


if __name__ == "__main__":
    main()
