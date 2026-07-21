#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm>

// 1. Vanilla Softmax: Processes the entire row at once (Step 1 baseline style)
__global__ void vanilla_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int row = blockIdx.x;
    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    // Pass 1: Find global max
    float max_val = -INFINITY;
    for (int i = 0; i < N; ++i) {
        max_val = fmaxf(max_val, row_in[i]);
    }

    // Pass 2: Compute global sum
    float sum_exp = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum_exp += expf(row_in[i] - max_val);
    }

    // Pass 3: Divide and write output
    for (int i = 0; i < N; ++i) {
        row_out[i] = expf(row_in[i] - max_val) / sum_exp;
    }
}

// 2. Online Softmax: Processes elements in chunks (simulating Flash Attention tiles)
__global__ void online_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int N, int tile_size) {
    int row = blockIdx.x;
    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    float m_running = -INFINITY;
    float d_running = 0.0f;

    // Loop over the row in sequential blocks/tiles
    for (int tile_idx = 0; tile_idx < N; tile_idx += tile_size) {
        float m_local = -INFINITY;

        // Find local max for this specific tile
        for (int i = 0; i < tile_size && (tile_idx + i) < N; ++i) {
            m_local = fmaxf(m_local, row_in[tile_idx + i]);
        }

        // Compute local denominator sum for this specific tile
        float d_local = 0.0f;
        for (int i = 0; i < tile_size && (tile_idx + i) < N; ++i) {
            d_local += expf(row_in[tile_idx + i] - m_local);
        }

        // --- THE ONLINE CORRECTION STEP ---
        float m_next = fmaxf(m_running, m_local);

        // Rescale the old running sum and add the rescaled local sum
        d_running = d_running * expf(m_running - m_next) + d_local * expf(m_local - m_next);
        m_running = m_next;
    }

    // Final pass: Since we now have the true global m and d, write out the results
    for (int i = 0; i < N; ++i) {
        row_out[i] = expf(row_in[i] - m_running) / d_running;
    }
}

int main() {
    const int ROWS = 4;
    const int N = 16;
    const int TILE_SIZE = 4; // Break a row of 16 elements into 4 tiles of size 4
    size_t bytes = ROWS * N * sizeof(float);

    // Initialize random host inputs
    float h_in[ROWS * N];
    for(int i = 0; i < ROWS * N; ++i) h_in[i] = static_cast<float>(rand() % 10);

    // Allocate GPU memory
    float *d_in, *d_out_vanilla, *d_out_online;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out_vanilla, bytes);
    cudaMalloc(&d_out_online, bytes);

    // Copy inputs to GPU
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Run both approaches
    vanilla_softmax_kernel<<<ROWS, 1>>>(d_in, d_out_vanilla, N);
    online_softmax_kernel<<<ROWS, 1>>>(d_in, d_out_online, N, TILE_SIZE);

    // Copy results back to CPU
    float h_out_vanilla[ROWS * N];
    float h_out_online[ROWS * N];
    cudaMemcpy(h_out_vanilla, d_out_vanilla, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_online, d_out_online, bytes, cudaMemcpyDeviceToHost);

    // Verify correctness
    bool match = true;
    for(int i = 0; i < ROWS * N; ++i) {
        if (std::abs(h_out_vanilla[i] - h_out_online[i]) > 1e-5) {
            match = false;
            std::cout << "Mismatch at index " << i << ": Vanilla=" << h_out_vanilla[i] << ", Online=" << h_out_online[i] << "\n";
        }
    }

    if(match) {
        std::cout << "SUCCESS: Online Softmax matches Vanilla Softmax perfectly!\n";
    }

    // Clean up
    cudaFree(d_in); cudaFree(d_out_vanilla); cudaFree(d_out_online);
    return 0;
}
