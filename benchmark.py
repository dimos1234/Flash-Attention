import torch
import math
import time
import os
from torch.utils.cpp_extension import load

print("Compiling CUDA kernels... (This might take a minute)")

# Load Standard Attention Kernel
std_attn = load(
    name="standard_attn_cuda",
    sources=["standard_attention.cu"],
    extra_cuda_cflags=["-O3"]
)

# Load Flash Attention Kernel
flash_attn = load(
    name="flash_attn_cuda",
    sources=["flash_attn_fwd.cu"],
    extra_cuda_cflags=["-O3"]
)

print("Compilation successful! Let the race begin.\n")

def run_benchmark():
    seq_lengths = [128, 256, 512, 1024, 2048]
    d = 64
    device = torch.device("cuda")
    
    warmup = 20
    repeats = 100

    # Header
    print(f"{'Seq Len':<10}{'Naive (ms)':<15}{'PyTorch (ms)':<15}{'Flash (ms)':<15}{'Naive Mem (MB)':<15}{'Flash Mem (MB)':<15}")
    print("-" * 85)

    # We will save results to write to our results file later
    results_rows = []

    for N in seq_lengths:
        scale = 1.0 / math.sqrt(d)
        
        Q = torch.randn(N, d, device=device)
        K = torch.randn(N, d, device=device)
        V = torch.randn(N, d, device=device)
        
        O_naive = torch.zeros(N, d, device=device)
        O_flash = torch.zeros(N, d, device=device)

        Q_4d = Q.unsqueeze(0).unsqueeze(0)
        K_4d = K.unsqueeze(0).unsqueeze(0)
        V_4d = V.unsqueeze(0).unsqueeze(0)

        # ---------------------------------------------------------
        # 1. Benchmark Naive CUDA Kernel
        # ---------------------------------------------------------
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        for _ in range(warmup):
            std_attn.standard_attention(Q, K, V, O_naive, N, d)
            
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        start_evt.record()
        for _ in range(repeats):
            std_attn.standard_attention(Q, K, V, O_naive, N, d)
        end_evt.record()
        torch.cuda.synchronize()
        
        naive_time = start_evt.elapsed_time(end_evt) / repeats
        # Get peak memory allocated during this step
        naive_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # ---------------------------------------------------------
        # 2. Benchmark PyTorch Native
        # ---------------------------------------------------------
        for _ in range(warmup):
            _ = torch.nn.functional.scaled_dot_product_attention(Q_4d, K_4d, V_4d)
            
        start_evt.record()
        for _ in range(repeats):
            _ = torch.nn.functional.scaled_dot_product_attention(Q_4d, K_4d, V_4d)
        end_evt.record()
        torch.cuda.synchronize()
        
        pt_time = start_evt.elapsed_time(end_evt) / repeats

        # ---------------------------------------------------------
        # 3. Benchmark Custom Flash Attention
        # ---------------------------------------------------------
        Br, Bc = 16, 16
        Tr, Tc = (N + Br - 1) // Br, (N + Bc - 1) // Bc
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        for _ in range(warmup):
            flash_attn.flash_attn_fwd_kernel(Q, K, V, O_flash, N, d, Tr, Tc, scale)
            
        start_evt.record()
        for _ in range(repeats):
            flash_attn.flash_attn_fwd_kernel(Q, K, V, O_flash, N, d, Tr, Tc, scale)
        end_evt.record()
        torch.cuda.synchronize()
        
        flash_time = start_evt.elapsed_time(end_evt) / repeats
        flash_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Print metrics
        print(f"{N:<10}{naive_time:<15.4f}{pt_time:<15.4f}{flash_time:<15.4f}{naive_mem:<15.2f}{flash_mem:<15.2f}")
        results_rows.append((N, naive_time, pt_time, flash_time, naive_mem, flash_mem))

    # Write results table to a markdown file inside results/ directory
    os.makedirs("results", exist_ok=True)
    with open("results/results_table.md", "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write("| Seq Len | Naive Attention (ms) | PyTorch (ms) | Flash Attention (ms) | Naive Memory (MB) | Flash Memory (MB) | Speedup |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results_rows:
            speedup = r[1] / r[3]
            f.write(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.2f} | {r[5]:.2f} | {speedup:.2f}x |\n")

    headline_speedup = results_rows[-1][1] / results_rows[-1][3]
    print(f"\n⚡ Done! Your Custom Flash Attention is {headline_speedup:.2f}x faster than the Naive baseline at Sequence Length 2048!")

if __name__ == "__main__":
    run_benchmark()
