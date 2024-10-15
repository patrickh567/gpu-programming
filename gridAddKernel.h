void matAddWithCuda(float* c, const float* x, const float* y, const unsigned int N, 
        const unsigned int M, const unsigned int P, const bool verify, 
        const unsigned int numThreads_x, const unsigned int numThreads_y, 
        const unsigned int numThreads_z);

//void printMatrix(const float* const* c, const unsigned int N, const unsigned int M,
  //      const unsigned int P);

//void printResults(const float* c, const float* x, const float* y, 
 //       const unsigned int M, const unsigned int N, const unsigned int P);

void matAddWithCpu(float* c, const float* x, const float* y, const unsigned int N, 
        const unsigned int M, const unsigned int P);

void verifyCuda(const float* matAddCpu, const float* matAddCuda, 
        const unsigned int N, const unsigned int M, const unsigned int P);

float* allocateMatrix(const unsigned int M, const unsigned int N, const unsigned int P);

void initializeMatrix(float* x, const unsigned int N, const unsigned int M,
        const unsigned int P, const unsigned int maxElementSize);

typedef struct args {
    bool verify = false;
    bool print = false;
    unsigned int numThreads_x = 8;
    unsigned int numThreads_y = 8;
    unsigned int numThreads_z = 8;
    unsigned int M = 1000;
    unsigned int N = 1000;
    unsigned int P = 1000;
} args_t;

void parsArgs(args_t* args, int argc, char* argv[]);
