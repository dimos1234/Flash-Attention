#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// ==========================================
// KERNEL 1: Calculate Scores (S = Q * K^T / sqrt(d))
// ==========================================
__global__ void compute_scores_kernel(const float* Q, const float* K, float* S, int N, int d) {
    // Each thread handles one cell in the N x N grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float score = 0.0f;
        // Dot product between Row of Q and Column of K^T (Row of K)
        for (int i = 0; i < d; ++i) {
            score += Q[row * d + i] * K[col * d + i]; 
        }
        // Scale by 1 / sqrt(d) and save to global memory matrix S
        S[row * N + col] = score / sqrtf(static_cast<float>(d));
    }
}

// ==========================================
// KERNEL 2: Standard Row-wise Softmax (S -> P)
// ==========================================
__global__ void softmax_rows_kernel(const float* S, float* P, int N) {
    // Each thread handles one entire row of the N x N grid
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N) {
        const float* row_S = S + row * N;
        float* row_P = P + row * N;

        // 1. Find the maximum value in this row
        float max_val = -INFINITY;
        for (int j = 0; j < N; ++j) {
            max_val = fmaxf(max_val, row_S[j]);
        }

        // 2. Compute the sum of exponentials (denominator)
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += expf(row_S[j] - max_val);
        }

        // 3. Compute final probability and save to global memory matrix P
        for (int j = 0; j < N; ++j) {
            row_P[j] = expf(row_S[j] - max_val) / sum;
        }
    }
}

// ==========================================
// KERNEL 3: Compute Output (O = P * V)
// ==========================================
__global__ void compute_output_kernel(const float* P, const float* V, float* O, int N, int d) {
    // Each thread handles one cell in the final N x d matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < d) {
        float out_val = 0.0f;
        // Matrix multiplication: Row of P multiplied by Column of V
        for (int i = 0; i < N; ++i) {
            out_val += P[row * N + i] * V[i * d + col];
        }
        // Save final result to global memory matrix O
        O[row * d + col] = out_val;
    }
}

// ==========================================
// The Main Orchestration Pipeline
// ==========================================
void standard_attention(const float* d_Q, const float* d_K, const float* d_V, float* d_O, int N, int d) {
    // Allocate temporary space in GPU memory for the huge intermediate N x N matrices
    float *d_S, *d_P;
    cudaMalloc(&d_S, N * N * sizeof(float));
    cudaMalloc(&d_P, N * N * sizeof(float));

    // Setup 2D blocks of threads (16x16 threads per block)
    dim3 threadsPerBlock2D(16, 16);
    dim3 gridS((N + 15) / 16, (N + 15) / 16);
    dim3 gridO((d + 15) / 16, (N + 15) / 16);
    
    // Run Step 1: Computes S and saves it to global GPU memory
    compute_scores_kernel<<<gridS, threadsPerBlock2D>>>(d_Q, d_K, d_S, N, d);
    
    // Run Step 2: Reads S from memory, computes P, saves P back to memory
    softmax_rows_kernel<<<(N + 255) / 256, 256>>>(d_S, d_P, N);
    
    // Run Step 3: Reads P from memory, computes final O, saves O to memory
    compute_output_kernel<<<gridO, threadsPerBlock2D>>>(d_P, d_V, d_O, N, d);

    // Free the temporary N x N matrices
    cudaFree(d_S);
    cudaFree(d_P);
}

int main() {
    std::cout << "Standard Attention Baseline successfully compiled!\n";
    return 0;
}
