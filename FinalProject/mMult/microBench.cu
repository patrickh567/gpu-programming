#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include "time.h"
#include "matLib.h"
#include "../../HW3/cuda_error.h"

#define M 8
#define N 512
#define SHRD_MEM_SIZE N * M

void launchRegisterBankConflict(const float* h_A, const float* h_B, float* h_C, 
        const unsigned int blockSize);

__global__ void registerBankConflicts(const float* A, const float* B, float* C) {

    __shared__ float shr_A[SHRD_MEM_SIZE];
    __shared__ float shr_B[SHRD_MEM_SIZE];

    float reg_A[M], reg_B[M], reg_C[M];

    for(int k = 0; k < N; k++) {
        // Load row of A and col of B into registers
        for(int i = 0; i < M; i++) {
            reg_A[i] = shr_A[i+k*M];
            reg_B[i] = shr_B[i+k*M];
        }
        // Perform dot product
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < M; j++) {
                reg_C[i * M + j] += reg_A[i] * reg_B[j];
            }
        }
    }

}

int main(void) {

    const unsigned int maxElementSize = 10;

    float* A = allocateMatrix(M, N);
    float* B = allocateMatrix(M, N);
    float* C = allocateMatrix(N, N);

    initializeMatrix(A, M, N, maxElementSize);
    initializeMatrix(B, M, N, maxElementSize);
    initializeMatrix(C, N, N, maxElementSize);

    launchRegisterBankConflict(A, B, C, 32);

    return 0;
}

void launchRegisterBankConflict(const float* h_A, const float* h_B, float* h_C, 
        const unsigned int blockSize) {

    float* d_A;
    float* d_B;
    float* d_C;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));
    
    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCuda(cudaMalloc((void**)&d_A, M * N * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_B, M * N * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_C, N * N * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, M * N * sizeof(float), cudaMemcpyHostToDevice));
     
    // Launch kernel
    dim3 DimGrid((N-1)/blockSize+1, (N-1)/blockSize+1);
    dim3 DimBlock(blockSize, blockSize, 1);

    registerBankConflicts<<<DimGrid, DimBlock>>>(d_A, d_B, d_C);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());
    
    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

}
