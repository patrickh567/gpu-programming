void printVector(const float* vec, const unsigned int size);

void printMatrix(const float* const* c, const unsigned int N, const unsigned int M);

void printResultsMatrix(const float* c, const float* x, const float* y, 
        const unsigned int M, const unsigned int N);

void verifyCudaMatrix(const float* matAddCpu, const float* matAddCuda, 
        const unsigned int N, const unsigned int M);

float* allocateMatrix(const unsigned int M, const unsigned int N);

void initializeMatrix(float* x, const unsigned int N, const unsigned int M, 
        const unsigned int maxElementSize);

void launchMMultKernel(void (*kernel)(const float*, const float*, float*, 
            const unsigned int, const unsigned int, const unsigned int, const unsigned int), 
        const float* h_A, const float* h_B, float* h_C, const unsigned int m, 
        const unsigned int n, const unsigned int p, const unsigned int blockSize);

void launchMMultKernelTiled(void (*kernel)(const float*, const float*, float*, const unsigned int, 
            const unsigned int, const unsigned int, const unsigned int), 
        const float* h_A, const float* h_B, float* h_C, const unsigned int m, 
        const unsigned int n, const unsigned int p, const unsigned int tileWidth);
