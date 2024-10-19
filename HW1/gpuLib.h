void gridAddWithCuda(float* c, const float*  x, const float*  y, 
        const unsigned int N, const unsigned int M, const unsigned int P,
        const bool verify, const unsigned int numThreads_x, const unsigned int numThreads_y, 
        const unsigned int numThreads_z);

void gridAddWithCpu(float* c, const float* x, const float*  y, 
        const unsigned int N, const unsigned int M, const unsigned int P);

void verifyCudaGrid(const float* gridAddCpu, const float* gridAddCuda, const unsigned int N, 
        const unsigned int M, const unsigned int P);

float* allocateGrid(const unsigned int M, const unsigned int N, const unsigned int P);

void initializeGrid(float* x, const unsigned int N, const unsigned int M, 
        const unsigned int P, const unsigned int maxElementSize);

void matAddWithCuda(float* c, const float* x, const float* y, const unsigned int N, 
        const unsigned int M, const bool verify, const unsigned int numThreads_x, 
        const unsigned int numThreads_y);

void printMatrix(const float* const* c, const unsigned int N, const unsigned int M);

void printResultsMatrix(const float* c, const float* x, const float* y, 
        const unsigned int M, const unsigned int N);

void matAddWithCpu(float* c, const float* x, const float* y, const unsigned int N, 
        const unsigned int M);

void verifyCudaMatrix(const float* matAddCpu, const float* matAddCuda, 
        const unsigned int N, const unsigned int M);

float* allocateMatrix(const unsigned int M, const unsigned int N);

void initializeMatrix(float* x, const unsigned int N, const unsigned int M, 
        const unsigned int maxElementSize);

void saxpyWithCuda(float* c, const float* x, const float* y, const float a,
        const unsigned int size, const bool verify, const unsigned int numThreads);

void saxpyWithCpu(float* c, const float* x, const float* y, const float a, 
        const unsigned int size);

void verifyCudaVector(const float* saxpyCpu, const float* saxpyCuda, const unsigned int size);

void printResultsVector(const float* c, const float* x, const float* y, const float a, 
        const unsigned int size);

void printVector(const float* vec, const unsigned int size);

typedef struct args {
    bool verify = false;
    bool print = false;
    unsigned int numThreads_x = 8;
    unsigned int numThreads_y = 8;
    unsigned int numThreads_z = 8;
    unsigned int M = 32;
    unsigned int N = 32;
    unsigned int P = 32;
} args_t;

void parsArgs(args_t* args, int argc, char* argv[]);
