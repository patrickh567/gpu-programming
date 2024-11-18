#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda_error.h"
#include <sstream>
#include "time.h"
#include "matLib.h"

#define MAX_TILE_WIDTH 32

// Naive CUDA Global Implementation
__global__ void sgemmNaive(int m, int n, int p, const float* A, 
        const float* B, float* C, const unsigned int blockSize) {

    const unsigned int Row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int Col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(Row < m && Col < n) {
        float PValue = 0.0;
        for(int i = 0; i < p, i++) {
            PValue += A[Col * p + i] + B[i * n + Row];
        }
        C[Col * n + Row] = PValue;
    }

}

// Naive CUDA Coalesced Global Implementation
__global__ void sgemmNaiveCoalescing(int m, int n, int p, const float* A, 
        const float* B, float* C, const unsigned int blockSize) {

    const unsigned int Row = blockIdx.x * blockSize + (threadIdx.x % blockSize);
    const unsigned int Col = blockIdx.y * blockSize + (threadIdx.y / blockSize);
    
    if(Row < m && Col < n) {
        float PValue = 0.0;
        for(int i = 0; i < p, i++) {
            PValue += A[Col * p + i] + B[i * n + Row];
        }
        C[Col * n + Row] = PValue;
    }

}

// CUDA shared memory with tiling
__global__ void sgemmSharedMemCacheTiling(const unsigned int m, const unsigned int n, 
        const unsigned int p, const float* A, const float* B, float* C, 
        const unsigned int tileWidth) {
    
    __shared__ float s_A[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    __shared__ float s_B[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0;

    for(int phase = 0; phase < (p - 1) / tileWidth + 1; phase++) {
        // Collaboratively load tile into shared memory
        if((phase * tileWidth + threadIdx.x < p) && (idx_y < m)) { 
            s_A[threadIdx.y][threadIdx.x] = d_A[(idx_y * p) + (phase * tileWidth + threadIdx.x)];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        if((phase * tileWidth + threadIdx.y < p) && (idx_x < n)) {
            s_B[threadIdx.y][threadIdx.x] = d_B[(phase * tileWidth + threadIdx.y) * n + idx_x];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0;
        }
        // Wait for threads to complete
        __syncthreads();
        // Perform partial dot-product
        if(idx_y < m && idx_x < n) {
            for(int k = 0; k < tileWidth; k++) {
                Pvalue += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
            }
        }
        // Wait for threads to complete
        __syncthreads();
            
    }

    // Write full dot-product to global memory
    if(idx_y < m && idx_x < n) {
        d_C[idx_y * n + idx_x] = Pvalue;
    }

}

int main(void) {

    const unsigned int N = 100;
    const unsigned int M = 100;
    const unsigned int P = 100;

    const unsigned int maxElementSize = 10;

    float* A = allocateMatrix(M, P);
    float* B = allocateMatrix(P, N);
    float* C = allocateMatrix(M, N);

    initializeMatrix(A, M, P, maxElementSize);
    initializeMatrix(B, P, N, maxElementSize);
    initializeMatrix(C, M, N, maxElementSize);

    launchMMultKernel(&sgemmNaive, A, B, C, M, N, P, 32);  
    launchMMultKernelTiled(&sgemmNaiveCoalesced, A, B, C, M, N, P, 32);  
    launchMMultKernelTiled(&sgemmSharedMemCacheTiling, A, B, C, M, N, P, 32);  

    return 0;   
}
