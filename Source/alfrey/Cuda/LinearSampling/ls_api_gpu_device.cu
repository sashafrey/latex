// This file contains implementation of CPU-functions, which operates with GPU data.

#include "ls_api_gpu_device.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cudamat/cudamat_kernels.cuh"

#include "ls_api_gpu_kernels.cuh"

#include "CublasHandleSingleton.h"
#include "CudaPtr2D.h"
#include "ls_api.h"
#include "Log.h"
#include "CudaException.h"
#include "SessionManager.h"

int transposeMatrix_device(unsigned char *odata, const unsigned char *idata, int input_width, int input_height) {
    // setup execution parameters
    unsigned int grid_x = input_height / COPY_BLOCK_SIZE;
    if (input_height % COPY_BLOCK_SIZE)
        grid_x++;

    unsigned int grid_y = input_width / COPY_BLOCK_SIZE;
    if (input_width % COPY_BLOCK_SIZE)
        grid_y++;

    dim3 grid(grid_x, grid_y, 1);
    dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);

    kTransposeMatrix<<< grid, threads >>>(odata, const_cast<unsigned char*>(idata), input_width, input_height);
    return 0;
}

int calcAlgs_device(
    int sessionId,
    const float *d_W, 
    int nAlgs,
    float* d_scores,
    unsigned char* d_EV,
    int* d_EC, 
    unsigned int* d_hashes)
{
    if ((d_scores == NULL) && (d_EV == NULL) && (d_EC == NULL) && (d_hashes == NULL)) {
        LOG_F(logWARNING, "calcAlgs() was called with all arguments set to NULL.");
        return 0;
    }
    
    if (nAlgs >= MAX_GRID_SIZE) 
    {
        LOG_F(logERROR, "Can't execute calcAlgs method with nAlgs > %i", MAX_GRID_SIZE);
        return LINEAR_SAMPLING_ERROR;
    }

    CudaSession s = CudaSessionManager::getInstance().getSession(sessionId);
    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    cudaSetDevice(s.deviceId);
    int nItems = s.nItems;
    int nFeatures = s.nFeatures;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Handle the case when d_scores == NULL, but d_EV or d_EC or d_hashes are not.
    float* d_scores_notnull;
    CudaPtr2D<float> d_scores_ptr;
    if (d_scores != NULL) {
        d_scores_notnull = d_scores;
    } else {
        // this will temporary allocate memory on device, and free after calcAlgs_device is completed.
        d_scores_ptr.reset(nItems, nAlgs);
        d_scores_notnull = d_scores_ptr;
    }

    // Produce matrix of scores
    CUBLAS_CHECK(cublasSgemm(CublasHandleSingleton::getInstance(), CUBLAS_OP_N, CUBLAS_OP_T, nItems, nAlgs, nFeatures, &alpha, *s.d_X, nItems, d_W, nAlgs, &beta, d_scores_notnull, nItems));

    if ((d_EC != NULL) || (d_hashes != NULL)) {
        // Produce error counts.
        kCalcAlgs<<<nAlgs, BLOCK_SIZE>>>(d_scores_notnull, *s.d_target, nItems, nAlgs, d_EC, d_hashes);
    }

    if (d_EV != NULL) {
        kCalcEV<<<nAlgs, BLOCK_SIZE>>>(d_scores_notnull, *s.d_target, nItems, nAlgs, d_EV);        
    }

    if (d_scores_ptr.get() != NULL) {
        // Note that all cuda calls are async.
        // Before releasing d_scores the device must complete all its calculations.
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return 0;
}

int compareAlgsToSources_device(
    unsigned char* d_evAlgs,
    unsigned char* d_evSources,
    int nAlgs,
    int nSources,
    int nItems,
    int* d_ecAlgsNotSources,
    int* d_ecSourcesNotAlgs)
{
    dim3 blocks(nAlgs, nSources);
    kCompareAlgsToSources<<<blocks, BLOCK_SIZE>>>(d_evAlgs, d_evSources, nAlgs, nSources, nItems, d_ecAlgsNotSources, d_ecSourcesNotAlgs);
    return 0;
}

int calcAlgSourceQEps_device(
    const int* d_algsEC,        // nAlgs, errors count 
    const int* d_algsConnectivity,        // nAlgs, errors upper connectivity
    const int* d_ecAlgsNotSources,        // nAlgs * nSources, number of proper errors of algs comparing to sources
    const int* d_ecSourcesNotAlgs,        // nAlgs * nSources, number of proper errors of sources comparing to algs
    const float* d_logFactorial,        // nItems + 1, table of e-based logarithms of factorials. ln(n!) = logFactorisl[n].
    const float *d_epsValues,
    int nItems,    
    int nTrainItems,
    int nAlgs, 
    int nSources, 
    int nEpsValues,
    float* d_QEps) 
{
    int shared_size = (2 * nItems + 1) * sizeof(float);
    shared_size = (float)BLOCK_SIZE * ceil((float)shared_size / (float)BLOCK_SIZE);
    dim3 block(nAlgs, nSources);
    kAssignScalar<<<NUM_VECTOR_OP_BLOCKS, NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(d_QEps, FLT_MAX, nAlgs * nEpsValues);
    kCalcAlgSourceQEps<<<block, BLOCK_SIZE, shared_size>>>(
        d_algsEC, d_algsConnectivity, d_ecAlgsNotSources, d_ecSourcesNotAlgs, d_logFactorial, d_epsValues, nItems, nTrainItems, nAlgs, nSources, nEpsValues, d_QEps);
    return 0;
}
