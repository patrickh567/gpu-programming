#include "../../HW3/cuda_error.h"

// Naive CPU Implementation
void CpuMatMul(int M, int N, int K, const float* A, const float* B, float* C) {
    
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            float Pvalue = 0.0;
            for(int k = 0; k < N; k++) {
                Pvalue += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = Pvalue;
        }
    }

}

void verifyCudaMatrix(const float*  matAddCpu, const float* matAddCuda, 
        const unsigned int N, const unsigned int M) {

    const float tol = 1e-7; 
    bool error = false;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            if(abs(matAddCpu[i * M + j] - matAddCuda[i * M + j]) > tol) {
                error = true;
                printf("CUDA matrix addition computation failed at index (%i, %i)\n", i, j);
                printf("CPU: %f, GPU: %f\n", matAddCpu[i * M + j], matAddCuda[i * M + j]);
                break;
            }
        }
    }
    if(error == false) {
        printf("CUDA matrix addition verification check passed.\n");
    }

}

float* allocateMatrix(const unsigned int M, const unsigned int N) {

    float* x = (float*)malloc(sizeof(float*)*N*M); 
    if(x == NULL) {
        printf("Malloc failed.\n");
        exit(-1);
    }

    return x;

}

void initializeMatrix(float* x, const unsigned int N, const unsigned int M, 
        const unsigned int maxElementSize) {

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            x[i * M + j] = (float)rand()/((float)RAND_MAX/maxElementSize); 
        }
    }

}

void launchMMultKernel(void (*kernel)(const float*, const float*, float*, 
            const unsigned int, const unsigned int, const unsigned int), 
        const float* h_A, const float* h_B, float* h_C, const unsigned int m, 
        const unsigned int n, const unsigned int p, const unsigned int blockSize) {

    float* d_A;
    float* d_B;
    float* d_C;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));
    
    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCuda(cudaMalloc((void**)&d_A, m * p * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_B, p * n * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_C, m * n * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(d_A, h_A, m * p * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, p * n * sizeof(float), cudaMemcpyHostToDevice));
     
    // Launch kernel
    dim3 DimGrid((n-1)/blockSize+1, (m-1)/blockSize+1);
    dim3 DimBlock(blockSize, blockSize, 1);

    (*kernel)<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, m, n, p);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());
    
    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

}

void launchMMultKernelTiled(void (*kernel)(const float*, const float*, float*, 
            const unsigned int, const unsigned int, const unsigned int, const unsigned int), 
        const float* h_A, const float* h_B, float* h_C, const unsigned int m, 
        const unsigned int n, const unsigned int p, const unsigned int tileWidth) {

    float* d_A;
    float* d_B;
    float* d_C;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));
    
    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCuda(cudaMalloc((void**)&d_A, m * p * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_B, p * n * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_C, m * n * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(d_A, h_A, m * p * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, p * n * sizeof(float), cudaMemcpyHostToDevice));
     
    // Launch kernel
    dim3 DimGrid((n-1)/tileWidth+1, (m-1)/tileWidth+1);
    dim3 DimBlock(tileWidth, tileWidth, 1);

    (*kernel)<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, m, n, p, tileWidth);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());
    
    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

}

void printVector(const float* vec, const unsigned int size) {
    printf("{");
    for(int i = 0; i < size-1; i++) {
        printf("%.2f, ", vec[i]); 
    }
    printf("%.2f}", vec[size-1]);
}

void printMatrix(const float* c, const unsigned int M, const unsigned int N) {

    printf("{"); 
    for(int i = 0; i < M-1; i++) {
        printVector(&c[i*N], N);
        printf(",\n");
    }
    printVector(&c[(M-1)*N], N);
    printf("}\n");

}

void printResultsMatrix(const float*  c, const float* a, const float* b, 
        const unsigned int M, const unsigned int N, const unsigned int P) {
    printMatrix(a, M, P);
    printf("X\n");
    printMatrix(b, P, N);
    printf("=\n");
    printMatrix(c, M, N);

}
