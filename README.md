# Flash Attention Forward Pass in CUDA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15wga9jwDSat06dbjX5823tR8sLNecm61)

**Headline Result:** Our custom Flash Attention forward pass CUDA kernel achieves a **1.26× speedup** and an **11.5× reduction in peak GPU memory usage** over the naive CUDA baseline at sequence length 2048 in FP32 precision.

---

## Why Flash Attention is Faster

Standard attention computes $S = QK^T / \sqrt{d}$, applies row-wise softmax to produce $P$, and finally computes $O = PV$. In a naive CUDA implementation, each of these stages requires materializing intermediate $N \times N$ matrices ($S$ and $P$) in GPU global memory (HBM). Because global memory bandwidth is significantly slower than GPU compute, standard attention is heavily **memory-bandwidth-bound** and scales quadratically in memory footprint $\mathcal{O}(N^2)$.

Flash Attention restructures the computation by tiling the input matrices $Q, K, V$ into blocks small enough to fit inside on-chip **Shared Memory (SRAM)**. By combining tiling with the **Online Softmax** algorithm, the intermediate $N \times N$ attention matrix is computed and accumulated incrementally inside fast SRAM without ever being written to global memory. This dramatically reduces memory transfers between HBM and GPU cores while scaling memory linearly $\mathcal{O}(N)$.

---

## Benchmark Results

| Sequence Length | Naive CUDA Baseline (ms) | PyTorch Native (ms) | Flash Attention (ms) | Naive Peak Mem | Flash Peak Mem | Speedup vs Naive |
|:---|:---|:---|:---|:---|:---|:---|
| **128** | 0.2111 | 0.0634 | 0.5544 | 0.28 MB | 0.19 MB | 0.38× |
| **256** | 0.8347 | 0.1177 | 0.5218 | 0.81 MB | 0.38 MB | 1.60× |
| **512** | 0.8751 | 0.0913 | 0.8639 | 2.62 MB | 0.75 MB | 1.01× |
| **1024** | 2.3009 | 0.1633 | 2.0182 | 9.25 MB | 1.50 MB | 1.14× |
| **2048** | 7.3930 | 0.4200 | 5.8680 | **34.50 MB** | **3.00 MB** | **1.26×** |

---

## Performance Charts

### Latency Comparison
![Latency vs Sequence Length](results/charts/latency_vs_seqlen.png)

### Memory Comparison
![Memory vs Sequence Length](results/charts/memory_vs_seqlen.png)

---

## Repository Structure

```text
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
