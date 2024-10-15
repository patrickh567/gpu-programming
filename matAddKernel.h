void matAddWithCuda(float* c, const float* x, const float* y, const unsigned int N, 
        const unsigned int M, const bool verify, const unsigned int numThreads_x, 
        const unsigned int numThreads_y);

void printMatrix(const float* const* c, const unsigned int N, const unsigned int M);

void printResults(const float* c, const float* x, const float* y, 
        const unsigned int M, const unsigned int N);

void matAddWithCpu(float* c, const float* x, const float* y, const unsigned int N, 
        const unsigned int M);

void verifyCuda(const float* matAddCpu, const float* matAddCuda, 
        const unsigned int N, const unsigned int M);

float* allocateMatrix(const unsigned int M, const unsigned int N);

void initializeMatrix(float* x, const unsigned int N, const unsigned int M, 
        const unsigned int maxElementSize);

typedef struct args {
    bool verify = false;
    bool print = false;
    unsigned int numThreads_x = 32;
    unsigned int numThreads_y = 32;
    unsigned int numThreads_z = 32;
    unsigned int M = 3024;
    unsigned int N = 4032;
    unsigned int P = 500;
} args_t;

void parsArgs(args_t* args, int argc, char* argv[]);
