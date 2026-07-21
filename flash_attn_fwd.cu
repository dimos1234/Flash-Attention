#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define Bc 16
#define Br 16

__global__ void flash_attn_fwd_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, int Tr, int Tc, float scale
) {
    // Each CUDA block handles one row-tile of Q
    int tx = threadIdx.x; // Thread ID within the block
    int row_tile_idx = blockIdx.x;

    // Allocate Shared Memory (SRAM) for the current tiles
    __shared__ float s_Q[Br][64]; // Assuming max head dim 'd' is 64 for simplicity
    __shared__ float s_K[Bc][64];
    __shared__ float s_V[Bc][64];

    // Local registers to keep track of online softmax stats for this block's rows
    // Since each thread can handle one row of the tile:
    float m_running = -INFINITY;
    float d_running = 0.0f;
    float r_O[64] = {0.0f}; // Accumulator for local output row

    // 1. Load Q tile into Shared Memory
    if (tx < Br && row_tile_idx * Br + tx < N) {
        for (int i = 0; i < d; ++i) {
            s_Q[tx][i] = Q[(row_tile_idx * Br + tx) * d + i];
        }
    }
    __syncthreads();

    // 2. Inner Loop over all column tiles of K and V
    for (int j = 0; j < Tc; ++j) {
        // Load K and V tiles into Shared Memory
        if (tx < Bc && j * Bc + tx < N) {
            for (int i = 0; i < d; ++i) {
                s_K[tx][i] = K[(j * Bc + tx) * d + i];
                s_V[tx][i] = V[(j * Bc + tx) * d + i];
            }
        }
        __syncthreads();

        // If this thread represents a valid row inside our Q tile
        int row_idx = row_tile_idx * Br + tx;
        if (row_idx < N && tx < Br) {
            float scores[Bc] = {0.0f};
            float m_local = -INFINITY;

            // Compute local scores: S_local = Q_tile * K_tile^T
            for (int c = 0; c < Bc; ++c) {
                if (j * Bc + c < N) {
                    float sum = 0.0f;
                    for (int i = 0; i < d; ++i) {
                        sum += s_Q[tx][i] * s_K[c][i];
                    }
                    scores[c] = sum * scale;
                    m_local = fmaxf(m_local, scores[c]);
                }
            }

            // Compute local denominator sum
            float d_local = 0.0f;
            for (int c = 0; c < Bc; ++c) {
                if (j * Bc + c < N) {
                    scores[c] = expf(scores[c] - m_local);
                    d_local += scores[c];
                }
            }

            // --- ONLINE SOFTMAX UPDATE & OUTPUT ACCUMULATION ---
            float m_next = fmaxf(m_running, m_local);
            float alpha = expf(m_running - m_next);
            float beta = expf(m_local - m_next);

            float d_next = d_running * alpha + d_local * beta;

            // Rescale previous output values and add new contribution
            for (int i = 0; i < d; ++i) {
                float pv = 0.0f;
                for (int c = 0; c < Bc; ++c) {
                    if (j * Bc + c < N) {
                        pv += scores[c] * s_V[c][i];
                    }
                }
                r_O[i] = (r_O[i] * d_running * alpha + pv * beta) / d_next;
            }

            // Update running statistics
            m_running = m_next;
            d_running = d_next;
        }
        __syncthreads();
    }

    // 3. Write final output tile back to Global Memory (HBM)
    int row_idx = row_tile_idx * Br + tx;
    if (row_idx < N && tx < Br) {
        for (int i = 0; i < d; ++i) {
            O[row_idx * d + i] = r_O[i];
        }
    }
}

#include <torch/extension.h>

void launch_flash_attn_fwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    int N, int d, int Tr, int Tc, float scale
) {
    // Launch the kernel with 1 block per Row Tile of Q
    flash_attn_fwd_kernel<<<Tr, Br>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N, d, Tr, Tc, scale
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fwd_kernel", &launch_flash_attn_fwd, "Flash Attention Forward Pass Kernel");
}
