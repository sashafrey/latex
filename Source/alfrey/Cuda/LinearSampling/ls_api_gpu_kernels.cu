// This file contains implementation of GPU-functions, which operates with GPU data.

#include "ls_api_gpu_kernels.cuh"
#include "stdio.h"
#include <float.h>

__global__ void kTransposeMatrix(unsigned char *odata, unsigned char *idata, int width, int height) {
    __shared__ unsigned char block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < width) && (yIndex < height)) {
        unsigned int index_in = yIndex * width + xIndex;

        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < height) && (yIndex < width)) {
        unsigned int index_out = yIndex * height + xIndex;

        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

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
    float *d_t ) 
{
    const int iRay = d_rayId[blockIdx.x];
    const int iW = d_wId[blockIdx.x];

    const int iThread = threadIdx.x;
    const int nThreads = BLOCK_SIZE; // must be the same as blockDim.x;

    const int nItemsPerThread = (int) ceil((float)nItems / (float)nThreads);
    const int iFirstItem = iThread * nItemsPerThread;
    const int iLastItem = iFirstItem + nItemsPerThread;

    __shared__ float min[BLOCK_SIZE];
    float result = 0.0f;
    float ignore = FLT_MAX;
    
    for (int iIter = 0; iIter < 2; iIter++) {
        // Step1. Calculate current T.
        float minT = FLT_MAX;
        for (int iItem = iFirstItem; iItem < iLastItem; ++iItem) {
            if (iItem < nItems) {
                float curT = - d_Xw0[IDX2C(iItem, iW, nItems)] / d_XR[IDX2C(iItem, iRay, nItems)];
                if ((curT > 0) && (curT < minT) && (curT != ignore)) {
                    minT = curT;
                }
            }
        }

        // Step2. Search the minimum.
        min[iThread] = minT;

        __syncthreads();
        int nTotalThreads = BLOCK_SIZE;
        int thread2;
        float temp2;

        while(nTotalThreads > 1)
        {
            int halfPoint = (nTotalThreads >> 1);    // divide by two
            // only the first half of the threads will be active.
 
            if (iThread < halfPoint) {
                thread2 = iThread + halfPoint;
 
                // Get the shared value stored by another thread
                temp2 = min[thread2];
                if (temp2 < min[iThread]) {
                    min[iThread] = temp2;
                }
            }
            
            __syncthreads();
 
            // Reducing the binary tree size by two:
            nTotalThreads = halfPoint;
        }

        ignore = min[0];
        if (ignore != FLT_MAX) {
            result = result + ignore / 2.0f;
        } else {
            result = 0.0f / 0.0f;
        }
    }

    if (iThread == 0) 
    {
        d_t[blockIdx.x] = result;
    }
}

__global__ void kCalcAlgs(
    const float* scores, 
    const int* target, 
    int nItems, 
    int nAlgs, 
    int* errorsCount, 
    unsigned int* hashes)
{
    const int iAlg = blockIdx.x;
    const int iThread = threadIdx.x;
    const int nThreads = BLOCK_SIZE; // must be the same as blockDim.x;

    const int nItemsPerThread = (int) ceil((float)nItems / (float)nThreads);
    const int iFirstItem = iThread * nItemsPerThread;
    const int iLastItem = iFirstItem + nItemsPerThread;

    int curErrors = 0;
    unsigned int curHash = 17;

    for (int iItem = iFirstItem; iItem < iLastItem; ++iItem) {
        if (iItem < nItems) {
            int predictedTarget = PREDICTED_TARGET(scores[IDX2C(iItem, iAlg, nItems)]);
            curHash = (curHash * 23 + predictedTarget);

            if (predictedTarget != target[iItem])
            {
                curErrors++;
            }
        }
    }

    __shared__ int sum[BLOCK_SIZE];
    __shared__ unsigned int hash[BLOCK_SIZE];
    sum[iThread] = curErrors;
    hash[iThread] = curHash;

    __syncthreads();
    int nTotalThreads = BLOCK_SIZE;
    int thread2;

    while(nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1);    // divide by two
        // only the first half of the threads will be active.
 
        if (iThread < halfPoint) {
            thread2 = iThread + halfPoint;
 
            // Get the shared value stored by another thread
            sum[iThread] += sum[thread2];
            hash[iThread] = (hash[iThread] * 23 + hash[thread2]);
        }
            
        __syncthreads();
 
        // Reducing the binary tree size by two:
        nTotalThreads = halfPoint;
    }

    if (iThread == 0)
    {
        if (errorsCount != NULL) {
            errorsCount[iAlg] = sum[0];
        }

        if (hashes != NULL) {
            hashes[iAlg] = hash[0];
        }
    }
}

// Usage: kCalcEV<<<nAlgs, BLOCK_SIZE>>>
__global__ void kCalcEV(const float* scores, const int *target, int nItems, int nAlgs, unsigned char *errorVectors) {
    const int iAlg = blockIdx.x;
    const int iThread = threadIdx.x;
    const int nThreads = BLOCK_SIZE; // must be the same as blockDim.x;

    const int nItemsPerThread = (int) ceil((float)nItems / (float)nThreads);
    const int iFirstItem = iThread * nItemsPerThread;
    const int iLastItem = iFirstItem + nItemsPerThread;

    int index;
    for (int iItem = iFirstItem; iItem < iLastItem; ++iItem) {
        if (iItem < nItems) {
            int currentTarget = target[iItem];
            index = IDX2C(iItem, iAlg, nItems);
            int predictedTarget = PREDICTED_TARGET(scores[index]);
            errorVectors[index] = (predictedTarget != currentTarget);
        }
    }
}

// Generates random bit vector of length "n" with "k" bits set to 1 and rest set to 0.
__device__ void kGenerateBooleanPermutation(char* perm_shared, curandState *curand_state, int n, int k) 
{
    // ToDo: replace with this:
    // http://stackoverflow.com/questions/12653995/how-to-generate-random-permutations-with-cuda
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; ++i) {
            perm_shared[i] = (i < k) ? 1 : 0;
        }

        for (int i = (n - 1); i > 0; --i) {
            kSwap(perm_shared, i, curand(curand_state) % (i + 1));
        }
    }

    __syncthreads();
}

// nItemsPadded = nItems rounded up to the closes multiple of 2 * BLOCK_SIZE. This is what should be used for perm_shared size.
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
    int* testEC) 
{
    extern __shared__ char perm_shared[];

    curandState curand_state;
    int sequence = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    curand_init(randomSeed, sequence, 0, &curand_state);

    kGenerateBooleanPermutation(perm_shared, &curand_state, nItems, nTrainItems);

    int id = threadIdx.x;
    int nAlgsPerThread = (int) ceil((float)nAlgs / (float)BLOCK_SIZE);
    int iFirst = id * nAlgsPerThread;
    int iLast  = iFirst + nAlgsPerThread;
    
    const int Uninitialized = -1;
    int bestId = Uninitialized;
    int nMinTrainErrors = nItems + 1;
    int nMaxTotalErrors = Uninitialized;
    for (int iAlg = iFirst; iAlg < iLast; ++iAlg) {
        if (iAlg < nAlgs) {
            int nTrainErrors = 0;
            for (int iItem = 0; iItem < nItems; ++iItem) {
                nTrainErrors += errorVectors[IDX2C(iItem, iAlg, nItems)] * perm_shared[iItem];
            }

            int nTotalErrors = nTotalErrorsPerAlg[iAlg];
            if ((nTrainErrors < nMinTrainErrors) || ((nTrainErrors == nMinTrainErrors) && (nTotalErrors >= nMaxTotalErrors))) {
                nMinTrainErrors = nTrainErrors;
                nMaxTotalErrors = nTotalErrors;
                bestId = iAlg;
            }
        }
    }

    kMinReductionWithTears(nMinTrainErrors, -nMaxTotalErrors, bestId, &nMinTrainErrors, &bestId);

    if (threadIdx.x == 0) {
        trainEC[blockIdx.x] = nMinTrainErrors;
        testEC[blockIdx.x] = nTotalErrorsPerAlg[bestId] - nMinTrainErrors;
    }        
}

// For each algorithm and source calculates the number of items where a) alg is better b) source is better
// Input:
//    - evAlgs - matrix of algs errors, nItems * nAlgs,
//    - evSources - matrix of source errors, nItems * nSources
// Output:
//    - ecAlgsNotSources - matrix of size nAlgs * nSources, containing the number of errors of the alg which are not errors of the source
//  - ecSourcesNotAlgs - matrix of size nAlgs * nSources, containing the number of errors of the source which are not errors of the alg
// Usage: kCompareAlgsToSources<<<dim3(nAlgs, nSources), BLOCK_SIZE>>>
__global__ void kCompareAlgsToSources(
    unsigned char* evAlgs,
    unsigned char* evSources,
    int nAlgs,
    int nSources,
    int nItems,
    int* ecAlgsNotSources,
    int* ecSourcesNotAlgs)
{
    int iAlg = blockIdx.x;
    int iSource = blockIdx.y;

    int algNotSource = 0;
    int sourceNotAlg = 0;
    for (int iItem = threadIdx.x; iItem < nItems; iItem += BLOCK_SIZE) {
        unsigned char eAlg = evAlgs[IDX2C(iItem, iAlg, nItems)];
        unsigned char eSource = evAlgs[IDX2C(iItem, iSource, nItems)];
        if (eAlg > eSource) algNotSource++;
        if (eSource > eAlg) sourceNotAlg++;
    }

    kSumReduction(algNotSource, &ecAlgsNotSources[IDX2C(iAlg, iSource, nAlgs)]);
    kSumReduction(sourceNotAlg, &ecSourcesNotAlgs[IDX2C(iAlg, iSource, nAlgs)]);
}

__device__ double BinCoeff(int n, int k, float* logFactorial) {
    if ((n < 0) || (k < 0) || (k > n)) return 0.0;
    return exp((double)(logFactorial[n] - logFactorial[k] - logFactorial[n - k]));
}

// Usage: kCalcAlgSourceQEps<<<dim3(nAlgs, nSources), BLOCK_SIZE, shared_size>>>, where shared_size is at least sizeof(float) * (2*L + 1).
__global__ void kCalcAlgSourceQEps(
    const int* algsEC,            // nAlgs, errors count 
    const int* algsUpperCon,            // nAlgs, errors upper connectivity
    const int* ecAlgsNotSources,        // nAlgs * nSources, number of proper errors of algs comparing to sources
    const int* ecSourcesNotAlgs,        // nAlgs * nSources, number of proper errors of sources comparing to algs
    const float* logFactorial_global,    // nItems + 1, table of e-based logarithms of factorials. ln(n!) = logFactorisl[n].
    const float* epsValues,
    int nItems,    
    int nItemsTrain,
    int nAlgs, 
    int nSources, 
    int nEpsValues,
    float* QEps)                        // nAlgs * epsValues. QEps(iAlg, iEps) is contribution of iAlg to QEps with eps = epsValues(iEps)
{
    const int iAlg = blockIdx.x;
    const int iSource = blockIdx.y;

    const int m = algsEC[iAlg];
    const int u = algsUpperCon[iAlg];
    const int q = ecAlgsNotSources[IDX2C(iAlg, iSource, nAlgs)];
    const int T = min<int>(q, ecSourcesNotAlgs[IDX2C(iAlg, iSource, nAlgs)]);
    const int L = nItems;
    const int ell = L / 2;
    const int k = L - ell;

    extern __shared__ float shared_data[];
    float* logfact = shared_data + 0;          // log factorial data, length = (L + 1);
    float* qeps_sums = shared_data + (L + 1);  // qeps_sums, length = L;

    // read logFactorial table to warp.    
    for (int j = threadIdx.x; j < (L + 1); j += BLOCK_SIZE) 
    {
        logfact[j] = logFactorial_global[j];
    }

    __syncthreads();

    for (int j = threadIdx.x; j < L; j += BLOCK_SIZE) {
        qeps_sums[j] = 0.0f;
    }

    __syncthreads();

    double CLl = BinCoeff(L, ell, logfact);    
    int j_max = ell * m / L;
    for (int j_ = threadIdx.x; j_ <= j_max; j_ += BLOCK_SIZE) {
        double sum = 0.0f;
        for (int t = 0; t <= T; ++t) {
            int L_ = L - u - q;        
            int ell_ = ell - u - t;
            int m_ = m - q;
            int j = j_ - t;
            sum += BinCoeff(q, t, logfact) * BinCoeff(m_, j, logfact) * BinCoeff(L_ - m_, ell_ - j, logfact);
        }

        sum /= CLl;
        qeps_sums[j_] = (float)sum;
    }

    __syncthreads();

    // calc partial sums in qeps.
    // ToDo: replace this with smart partial-summ algorithm! (similar to reduction)
    if (threadIdx.x == 0) {
        for (int j = 1; j <= j_max; ++j) {
            qeps_sums[j] += qeps_sums[j - 1];
        }
    }

    __syncthreads();

    for (int iEps = threadIdx.x; iEps < nEpsValues; iEps += BLOCK_SIZE) {
        float* targetLocation = &QEps[IDX2C(iAlg, iEps, nAlgs)];
        float s_max = ell * (m - epsValues[iEps] * k) / L;
        float potentialValue = (s_max >= 0) ? qeps_sums[(int)s_max] : 0.0f;

        // atomics require the following project config: <CodeGeneration>compute_20,sm_21</CodeGeneration>
        // http://stackoverflow.com/questions/14411435/how-to-set-cuda-compiler-flags-in-visual-studio-2010
        atomicMin((int*)targetLocation, *(int*)(&potentialValue));
    }
}