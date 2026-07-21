import torch
from torch.utils.cpp_extension import load
import math

# =========================================================================
# Step 5: JIT Compile and Wrap the CUDA Kernel into a Python Interface
# =========================================================================
print("Compiling CUDA kernel via PyTorch JIT Extension Loader...")
flash_attn_module = load(
    name="flash_attn_cuda",
    sources=["flash_attn_fwd.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)
print("Compilation successful!")

# =========================================================================
# Step 4: Validate Correctness against PyTorch Reference
# =========================================================================
def run_validation():
    # Test configurations required by the project spec
    seq_lengths = [128, 256, 512, 1024]
    head_dims = [32, 64] # Keeping max at 64 to match our kernel's static shared memory allocation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning correctness validation on device: {device}\n")
    print(f"{'Seq Len':<10}{'Head Dim':<10}{'Max Abs Error':<15}{'Status':<10}")
    print("-" * 48)

    all_passed = True

    for N in seq_lengths:
        for d in head_dims:
            # Create random test inputs (Batch=1, Head=1, SeqLen=N, Dim=d)
            Q = torch.randn(N, d, device=device, dtype=torch.float32)
            K = torch.randn(N, d, device=device, dtype=torch.float32)
            V = torch.randn(N, d, device=device, dtype=torch.float32)
            O_custom = torch.zeros(N, d, device=device, dtype=torch.float32)

            # 1. Get PyTorch Reference Output
            # Reshape to (Batch, Head, SeqLen, Dim) because PyTorch expects 4D tensors
            Q_4d = Q.unsqueeze(0).unsqueeze(0)
            K_4d = K.unsqueeze(0).unsqueeze(0)
            V_4d = V.unsqueeze(0).unsqueeze(0)

            O_ref = torch.nn.functional.scaled_dot_product_attention(Q_4d, K_4d, V_4d)
            O_ref = O_ref.squeeze(0).squeeze(0) # Squeeze back to 2D (N, d)

            # 2. Run Our Custom Flash Attention Kernel
            scale = 1.0 / math.sqrt(d)
            Br = 16
            Bc = 16
            Tr = (N + Br - 1) // Br
            Tc = (N + Bc - 1) // Bc

            # Launch the JIT loaded function
            # Under the hood, PyTorch matches this call directly to the entry point in our .cu file
            flash_attn_module.flash_attn_fwd_kernel(Q, K, V, O_custom, N, d, Tr, Tc, scale)

            # 3. Calculate Error
            max_error = torch.max(torch.abs(O_custom - O_ref)).item()

            # Specs require error to be within 1e-3
            passed = max_error <= 1e-3
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False

            print(f"{N:<10}{d:<10}{max_error:<15.6f}{status:<10}")

    if all_passed:
        print("\n🎉 SUCCESS: All configurations passed within the 1e-3 error tolerance limit!")
    else:
        print("\n❌ FAILURE: Some configurations mismatched PyTorch reference outputs.")

if __name__ == "__main__":
    run_validation()
