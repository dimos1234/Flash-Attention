# Benchmark Methodology

- **Hardware:** NVIDIA Tesla T4 GPU (Google Colab Environment)
- **Data Type:** Single-Precision Floating Point (FP32)
- **Batch Size:** 1
- **Head Dimension:** 64
- **Warmup Runs:** 20 iterations
- **Timed Runs:** 100 iterations per configuration
- **Timing Protocol:** `torch.cuda.Event` synchronized timing
