#pragma once

void saxpyWithCuda(float* c, const float* x, const float* y, const float a,
        const unsigned int size, const bool verify, const unsigned int numThreads);
void saxpyWithCpu(float* c, const float* x, const float* y, const float a, 
        const unsigned int size);
void verifyCuda(const float* saxpyCpu, const float* saxpyCuda, const unsigned int size);
void printResults(const float* c, const float* x, const float* y, const float a, 
        const unsigned int size);

