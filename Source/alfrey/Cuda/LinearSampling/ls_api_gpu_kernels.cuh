// This file contains declaration of GPU-functions, which operates with GPU data

#ifndef __LINEAR_SAMPLING_KERNELS
#define __LINEAR_SAMPLING_KERNELS

#include "ls_api.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

#include "common.h"

template<class T> 
__device__ void kSwap(T* p, int a, int b) {
    T tmp = p[a];
    p[a] = p[b];
    p[b] = tmp;
}

template<class T> inline
__device__ T min(T a, T b) 
{
    return (a < b) ? a : b;
}

// Performs min reduction with a special comparison function:
// [i < j] == (value[i] < value[j]) OR ((value[i] == value[j]) && tear[i] < tear[j])
template<class T>
__device__ void kMinReductionWithTears(T value, T tear, int value_id, T *min, int *min_id)
{
    __shared__ T values[BLOCK_SIZE];
    __shared__ T tears[BLOCK_SIZE];
    __shared__ int values_ids[BLOCK_SIZE];

    values[threadIdx.x] = value;
    tears[threadIdx.x] = tear;
    values_ids[threadIdx.x] = value_id;

    __syncthreads();

    int nTotalThreads = BLOCK_SIZE;
    while(nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1);    // divide by two
        // only the first half of the threads will be active.
 
        if (threadIdx.x < halfPoint) {
            int thread2 = threadIdx.x + halfPoint;
 
            // Get the shared value stored by another thread
            T temp2 = values[thread2];
            if ((temp2 < values[threadIdx.x]) || ((temp2 == values[threadIdx.x]) && (tears[thread2] < tears[threadIdx.x]))) {
                values[threadIdx.x] = temp2;
                tears[threadIdx.x] = tears[thread2];
                values_ids[threadIdx.x] = values_ids[thread2];
            }
        }
            
        __syncthreads();
 
        // Reducing the binary tree size by two:
        nTotalThreads = halfPoint;
    }

    if (threadIdx.x == 0) {
        *min = values[0];
        *min_id = values_ids[0];
    }
}

template<class T>
__device__ void kSumReduction(T value, T *sum) 
{
    __shared__ T values[BLOCK_SIZE];

    values[threadIdx.x] = value;
    __syncthreads();

    int nTotalThreads = BLOCK_SIZE;
    while(nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1);    // divide by two
        // only the first half of the threads will be active.
 
        if (threadIdx.x < halfPoint) {
            int thread2 = threadIdx.x + halfPoint;
            values[threadIdx.x] += values[thread2];
        }
            
        __syncthreads();
 
        // Reducing the binary tree size by two:
        nTotalThreads = halfPoint;
    }

    if (threadIdx.x == 0) {
        *sum = values[0];
    }
}

__global__ void kTransposeMatrix(unsigned char *odata, unsigned char *idata, int width, int height);

// Usage: kFindNeighbors<<< nTargets, BLOCK_SIZE >>>
// Searches for adjecent cells of multiple starting points in multiple walking dirrections.
// Input:
// d_X, d_R, d_XR --- standard things (object-feature matrix, rays of random dirrections, and their matrix product)
// d_w0 -- matrix of size (nW * nFeatures). Defines the list of potential starting points.
// d_Xw0 -- matrix product of w0 and X
// d_rayId, d_wId -- arrays of the length nTargets, that define the actual starting point and the actual ray dirrection
// d_t -- output array of length nTargets.
__global__ void kFindNeighbors( 
    const float *d_X, 
    const float *d_R, 
    const float *d_XR, 
    const float *d_w0,  // of size nW * nFeatures
    const float *d_Xw0, // of size nItems * nW
    const int *d_rayId, // index of the ray to be used by blockIdx.x
    const int *d_wId,   // index of the vector W to be used by blockIdx.x
    int nItems, 
    int nFeatures, 
    int nTargets, 
    int nW,
    float *d_t );

__global__ void kCalcAlgs(
    const float* scores, 
    const int* target, 
    int nItems, 
    int nAlgs, 
    int* errorsCount, 
    unsigned int* hashes);

// Usage: kCalcEV<<<nAlgs, BLOCK_SIZE>>>
__global__ void kCalcEV(const float* scores, const int *target, int nItems, int nAlgs, unsigned char *errorVectors);

__device__ void kGenerateBooleanPermutation(char* perm_shared, curandState *curand_state, int n);

// nItemsPadded = nItems rounded up to the closes multiple of 2 * BLOCK_SIZE. This is what should be sued for perm_shared size.
__global__ void kPerformCrossValidation(
    const char* errorVectors, 
    const int* nTotalErrorsPerAlg, 
    int nItems,
    int nTrainItems,
    int nAlgs, 
    int nIters, 
    int nItemsPadded, 
    int randomSeed,
    int* trainEC,
    int* testEC);

__global__ void kCompareAlgsToSources(
    unsigned char* evAlgs,
    unsigned char* evSources,
    int nAlgs,
    int nSources,
    int nItems,
    int* ecAlgsNotSources,
    int* ecSourcesNotAlgs);

// Usage: kCalcAlgSourceQEps<<<dim3(nAlgs, nSources), BLOCK_SIZE, shared_size>>>, where shared_size is at least sizeof(float) * (2*L + 1).
__global__ void kCalcAlgSourceQEps(
    const int* algsEC,                    // nAlgs, errors count 
    const int* algsConnectivity,        // nAlgs, errors upper connectivity
    const int* ecAlgsNotSources,        // nAlgs * nSources, number of proper errors of algs comparing to sources
    const int* ecSourcesNotAlgs,        // nAlgs * nSources, number of proper errors of sources comparing to algs
    const float* logFactorial_global,    // nItems + 1, table of e-based logarithms of factorials. ln(n!) = logFactorisl[n].
    const float* epsValues,
    int nItems,    
    int nTrainItems,
    int nAlgs, 
    int nSources, 
    int nEpsValues,
    float* QEps);                        // nAlgs * nEpsValues. QEps(iAlg, iEps) is contribution of iAlg to QEps with eps = epsValues[iEps].

#endif  // __LINEAR_SAMPLING_KERNELS 
