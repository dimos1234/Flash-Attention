import os
import matplotlib.pyplot as plt

# 1. Create directories
os.makedirs("results/charts", exist_ok=True)

# Data collected from your benchmark run
seq_lens = [128, 256, 512, 1024, 2048]
naive_lat = [0.3402, 0.9791, 1.0507, 2.6513, 8.0829]
pt_lat = [0.0634, 0.1148, 0.0922, 0.1631, 0.4272]
flash_lat = [0.5436, 0.4722, 0.8562, 2.0219, 5.8839]

naive_mem = [0.16, 0.31, 0.62, 1.25, 2.50]
flash_mem = [0.19, 0.38, 0.75, 1.50, 3.00]

# ---------------------------------------------------------
# Chart 1: Latency vs Sequence Length
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(seq_lens, naive_lat, marker='o', label='Naive CUDA Baseline', color='#e74c3c', linewidth=2)
plt.plot(seq_lens, flash_lat, marker='s', label='Flash Attention (This Project)', color='#2ecc71', linewidth=2)
plt.plot(seq_lens, pt_lat, marker='^', label='PyTorch F.scaled_dot_product_attention', color='#3498db', linewidth=2)

plt.title("Execution Latency vs. Sequence Length (FP32, Head Dim 64)", fontsize=12, fontweight='bold')
plt.xlabel("Sequence Length (N)", fontsize=10)
plt.ylabel("Execution Time (ms)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("results/charts/latency_vs_seqlen.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Chart 2: Peak Memory vs Sequence Length
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(seq_lens, naive_mem, marker='o', label='Naive CUDA Baseline', color='#e74c3c', linewidth=2)
plt.plot(seq_lens, flash_mem, marker='s', label='Flash Attention (Tiled)', color='#2ecc71', linewidth=2)

plt.title("Peak Memory Usage vs. Sequence Length", fontsize=12, fontweight='bold')
plt.xlabel("Sequence Length (N)", fontsize=10)
plt.ylabel("Peak Allocated Memory (MB)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("results/charts/memory_vs_seqlen.png", dpi=300)
plt.close()

print("✅ Charts generated in results/charts/")

# ---------------------------------------------------------
# Methodology Note
# ---------------------------------------------------------
with open("results/methodology.md", "w") as f:
    f.write("""# Benchmark Methodology

- **Hardware:** NVIDIA Tesla T4 GPU (Google Colab Environment)
- **Data Type:** Single-Precision Floating Point (FP32)
- **Batch Size:** 1
- **Head Dimension:** 64
- **Warmup Runs:** 20 iterations
- **Timed Runs:** 100 iterations per configuration
- **Timing Protocol:** `torch.cuda.Event` synchronized timing
""")

# ---------------------------------------------------------
# Final README.md
# ---------------------------------------------------------
readme_content = """# Flash Attention Forward Pass in CUDA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**Headline Result:** Our custom Flash Attention forward pass CUDA kernel achieves a **1.37× speedup** over the naive CUDA baseline at a sequence length of 2048 in FP32 precision.

---

## Why Flash Attention is Faster

Standard attention computes $S = QK^T / \\sqrt{d}$, applies row-wise softmax to produce $P$, and finally computes $O = PV$. In a naive CUDA implementation, each of these stages requires materializing intermediate $N \\times N$ matrices ($S$ and $P$) in GPU global memory (HBM). Because global memory bandwidth is significantly slower than GPU compute, standard attention is heavily **memory-bandwidth-bound**.

Flash Attention restructures the computation by tiling the input matrices $Q, K, V$ into blocks small enough to fit inside on-chip **Shared Memory (SRAM)**. By combining tiling with the **Online Softmax** algorithm, the intermediate $N \\times N$ attention matrix is computed and accumulated incrementally inside fast SRAM without ever being written to global memory. This dramatically reduces memory transfers between HBM and the GPU cores.

---

## Benchmark Results

| Sequence Length | Naive CUDA Baseline (ms) | PyTorch Native (ms) | Flash Attention (ms) | Speedup vs Naive |
|---|---|---|---|---|
| 128 | 0.3402 | 0.0634 | 0.5436 | 0.63x |
| 256 | 0.9791 | 0.1148 | 0.4722 | 2.07x |
| 512 | 1.0507 | 0.0922 | 0.8562 | 1.23x |
| 1024 | 2.6513 | 0.1631 | 2.0219 | 1.31x |
| 2048 | 8.0829 | 0.4272 | 5.8839 | **1.37x** |

---

## Performance Charts

### Latency Comparison
![Latency vs Sequence Length](results/charts/latency_vs_seqlen.png)

### Memory Comparison
![Memory vs Sequence Length](results/charts/memory_vs_seqlen.png)

---

## Repository Structure

.
├── attn_standard.cu     # Naive CUDA baseline implementation
├── flash_attn_fwd.cu    # Tiled Flash Attention forward pass kernel
├── online_softmax.cu    # Standalone online softmax verification
├── validate.py          # Correctness test vs PyTorch (1e-3 tolerance)
├── benchmark.py         # Performance profiling script
├── results/
│   ├── methodology.md   # Hardware & benchmark setup specifications
│   ├── results_table.md # Detailed raw numbers
│   └── charts/          # Generated latency & memory plots
└── README.md            # Project overview & documentation

"""

with open("README.md", "w") as f:
    f.write(readme_content)

print("✅ README.md & methodology.md written successfully!")
