#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error.h"
#include "saxpyKernel.h"

#include <stdio.h>

__global__ void saxpyKernel(const float a, float* c, const float *y, 
        const float *x, const int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        c[tid] = a * y[tid] + x[tid]; 
    }
}

int main(int argc, char* argv[]) {

    bool verify = false;
    bool print = false;
    unsigned int numThreads = 10;
    unsigned int arraySize = 10;
    const float maxElementSize = 10.0;
    const float a = 10.0;

    if(argc > 7) {
        printf("Wrong number of args.\n");
        exit(-1);
    }

    if(argc > 1) {
        for(int i = 1; i < argc; i++) {
            if(strcmp(argv[i], "--verify") == 0) {
                verify = true;
            } else if(strcmp(argv[i], "--print_results") == 0) {
                print = true;
            } else if(strcmp(argv[i], "--num_threads") == 0) {
                numThreads = atoi(argv[i+1]);
                i++;
            } else if(strcmp(argv[i], "--array_size") == 0) {
                arraySize = atoi(argv[i+1]);
                i++;
            } else {
                printf("Unrecognized arg. Aborting\n"); 
                exit(-1);
            }
        }
    }
    
    // Cant assign more threads than there are elements in the arrays
    assert(numThreads <= arraySize);

    float* x = (float*)malloc(arraySize*sizeof(float));
    if(x == NULL) {
        printf("Malloc failed for array x.\n");
        exit(-1);
    }

    float* y = (float*)malloc(arraySize*sizeof(float));
    if(y == NULL) {
        printf("Malloc failed for array y.\n");
        exit(-1);
    }
    
    float* c = (float*)malloc(arraySize*sizeof(float));
    if(c == NULL) {
        printf("Malloc failed for array c.\n");
        exit(-1);
    }

    for(int i = 0; i < arraySize; i++) {
        x[i] = (float)rand()/((float)RAND_MAX/maxElementSize); 
        y[i] = (float)rand()/((float)RAND_MAX/maxElementSize); 
    }

    saxpyWithCuda(c, x, y, a, arraySize, verify, numThreads);
    
    if(print) {
        printResults(c, x, y, a, arraySize);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCuda(cudaDeviceReset());

    free(x);
    free(y);
    
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void saxpyWithCuda(float* c, const float *x, const float *y, const float a,  
        const unsigned int size, const bool verify, const unsigned int numThreads) {
    float* dev_y = 0;
    float* dev_x = 0;
    float* dev_c = 0;
    //cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCuda(cudaSetDevice(0));

    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCuda(cudaMalloc((void**)&dev_y, size * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_x, size * sizeof(float)));
    checkCuda(cudaMalloc((void**)&dev_c, size * sizeof(float)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch a kernel on the GPU with one thread for each element.
    saxpyKernel<<<ceil(size/numThreads), numThreads>>>(a, dev_c, dev_y, dev_x, size);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());
    
    // Run CPU Saxpy, overlapping latency with CUDA saxpy computation
    float* cpu_c;
    if(verify) {
        cpu_c = (float*)malloc(size * sizeof(float));
        saxpyWithCpu(cpu_c, x, y, a, size); 
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify CUDA computation is correct
    if(verify) {
        verifyCuda(cpu_c, c, size);
        free(cpu_c);
    }

    checkCuda(cudaFree(dev_y));
    checkCuda(cudaFree(dev_x));
    checkCuda(cudaFree(dev_c));
}

void saxpyWithCpu(float* c, const float *x, const float *y, const float a, 
        const unsigned int size) {
    for(int i = 0; i < size; i++) {
        c[i] = a * y[i] + x[i]; 
    }
    //printResults(c, x, y, a, size);
}

void verifyCuda(const float* saxpyCpu, const float* saxpyCuda, const unsigned int size) {
    const float tol = 1e-4; 
    bool error = false;
    for(int i = 0; i < size; i++) {
        if(abs(saxpyCpu[i] - saxpyCuda[i]) > tol) {
            error = true;
            printf("CUDA saxpy computation failed at index %i\n", i);
            printf("CPU: %f, GPU: %f\n", saxpyCpu[i], saxpyCuda[i]);
            break;
        }
    }
    if(error == false) {
        printf("CUDA saxpy verification check passed.\n");
    }
}

void printResults(const float* c, const float* x, const float* y, const float a, const unsigned int size) {
    printf("{");
    for(int i = 0; i < size-1; i++) {
        printf("%.2f, ", y[i]); 
    }
    printf("%.2f} * %.2f + \n{", y[size-1], a);
    for(int i = 0; i < size-1; i++) {
        printf("%.2f, ", x[i]); 
    }
    printf("%.2f} = \n{", x[size-1]);
    for(int i = 0; i < size-1; i++) {
        printf("%.2f, ", c[i]); 
    }
    printf("%.2f}\n", c[size-1]);
}
