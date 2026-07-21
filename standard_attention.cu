#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <torch/extension.h>

// KERNEL 1: S = Q * K^T / sqrt(d)
__global__ void compute_scores_kernel(const float* Q, const float* K, float* S, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float score = 0.0f;
        for (int i = 0; i < d; ++i) {
            score += Q[row * d + i] * K[col * d + i];
        }
        S[row * N + col] = score / sqrtf(static_cast<float>(d));
    }
}

// KERNEL 2: P = softmax(S)
__global__ void softmax_rows_kernel(const float* S, float* P, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        const float* row_S = S + row * N;
        float* row_P = P + row * N;

        float max_val = -INFINITY;
        for (int j = 0; j < N; ++j) {
            max_val = fmaxf(max_val, row_S[j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += expf(row_S[j] - max_val);
        }

        for (int j = 0; j < N; ++j) {
            row_P[j] = expf(row_S[j] - max_val) / sum;
        }
    }
}

// KERNEL 3: O = P * V
__global__ void compute_output_kernel(const float* P, const float* V, float* O, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < d) {
        float out_val = 0.0f;
        for (int i = 0; i < N; ++i) {
            out_val += P[row * N + i] * V[i * d + col];
        }
        O[row * d + col] = out_val;
    }
}

// The PyTorch Launcher
void launch_standard_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, int N, int d) {
    float *d_S, *d_P;
    cudaMalloc(&d_S, N * N * sizeof(float));
    cudaMalloc(&d_P, N * N * sizeof(float));

    dim3 threadsPerBlock2D(16, 16);
    dim3 gridS((N + 15) / 16, (N + 15) / 16);
    dim3 gridO((d + 15) / 16, (N + 15) / 16);

    compute_scores_kernel<<<gridS, threadsPerBlock2D>>>(Q.data_ptr<float>(), K.data_ptr<float>(), d_S, N, d);
    softmax_rows_kernel<<<(N + 255) / 256, 256>>>(d_S, d_P, N);
    compute_output_kernel<<<gridO, threadsPerBlock2D>>>(d_P, V.data_ptr<float>(), O.data_ptr<float>(), N, d);

    cudaFree(d_S);
    cudaFree(d_P);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("standard_attention", &launch_standard_attention, "Standard Attention Baseline Kernel");
}
