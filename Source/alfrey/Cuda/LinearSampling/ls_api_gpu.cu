// This file contains implementation of CPU-functions, which takes CPU-data pointers.

#include <stdio.h>
#include <time.h>
#include <float.h>
#include <stdlib.h>
#include <vector>
#include <queue>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

#include "cudamat\cudamat_kernels.cuh"
#include "ls_api.h"
#include "ls_api_gpu_kernels.cuh"
#include "ls_api_gpu_device.cuh"

#include "CudaPtr2D.h"
#include "HostPtr2D.h"
#include "CublasHandleSingleton.h"
#include "SessionManager.h"
#include "Log.h"
#include "CudaException.h"
#include "recorder.h"
#include "SourcesContainer.h"

int calcAlgs(
    int sessionId,
    const float *h_W, 
    int nAlgs,
    float* h_scores,
    unsigned char* h_EV,
    int* h_EC, 
    unsigned int* h_hashes)
{
    LOG_F(logINFO, "calcAlgs(), sessionId = %i, nAlgs = %i", sessionId, nAlgs);
    CHECK(Recorder::getInstance().record_calcAlgs(sessionId, h_W, nAlgs, h_scores, h_EV, h_EC, h_hashes));

    if (nAlgs >= MAX_GRID_SIZE) 
    {
        LOG_F(logERROR, "Can't execute calcAlgs method with nAlgs > %i", MAX_GRID_SIZE);
        return LINEAR_SAMPLING_ERROR;
    }

    CudaSession s = CudaSessionManager::getInstance().getSession(sessionId);
    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    CUDA_CHECK(cudaSetDevice(s.deviceId));
    int nItems = s.nItems;
    int nFeatures = s.nFeatures;

    // Copy matrix of algorithms to device.
    CudaPtr2D<float> d_W(nAlgs, nFeatures, h_W);        

    CudaPtr2D<float> d_scores;
    CudaPtr2D<int> d_EC;
    CudaPtr2D<unsigned int> d_hashes;
    CudaPtr2D<unsigned char> d_EV;
    if (h_scores != NULL) d_scores.reset(nItems, nAlgs);
    if (h_EC != NULL)      d_EC.reset(nAlgs, 1);
    if (h_hashes != NULL) d_hashes.reset(nAlgs, 1);
    if (h_EV != NULL)      d_EV.reset(nItems, nAlgs); 
    
    CHECK(calcAlgs_device(sessionId, d_W, nAlgs, d_scores, d_EV, d_EC, d_hashes));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (h_scores != NULL) CUDA_CHECK(cudaMemcpy(h_scores, d_scores, nItems * nAlgs * sizeof(float), cudaMemcpyDeviceToHost));
    if (h_EC != NULL)      CUDA_CHECK(cudaMemcpy(h_EC, d_EC, nAlgs * sizeof(int), cudaMemcpyDeviceToHost));
    if (h_hashes != NULL) CUDA_CHECK(cudaMemcpy(h_hashes, d_hashes, nAlgs * sizeof(unsigned int), cudaMemcpyDeviceToHost));    
    if (h_EV != NULL)      CUDA_CHECK(cudaMemcpy(h_EV, d_EV, nItems * nAlgs * sizeof(unsigned char), cudaMemcpyDeviceToHost));    
    
    return 0;
}

// Calculates hashes and error counts based on error vectors.
int calcAlgsEV(
    const unsigned char* h_EV,  // nItems * nAlgs   - matrix of error vectors;
    const int* h_target,        // nItems * 1       - vector of target classes (labels must be 0 or 1);
    int nItems,                 //                  - number of items;
    int nAlgs,                  //                  - number of algorithms;
    int* h_EC,         // nAlgs * 1        - vector of error counts;
    unsigned int* h_hashes)     // nAlgs * 1        - vector of algorithm's hashes
{
    LOG_F(logINFO, "calcAlgsEV(), nItems = %i, nAlgs = %i", nItems, nAlgs);
    CHECK(Recorder::getInstance().record_calcAlgsEV(h_EV, h_target, nItems, nAlgs, h_EC, h_hashes));

    if (h_EV == NULL) {
        LOG_F(logERROR, "calcAlgsEV(), h_EV == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_target == NULL) {
        LOG_F(logERROR, "calcAlgsEV(), h_target == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    HostPtr2D<float> h_scores(nItems, nAlgs);
    for (int iItem = 0; iItem < nItems; ++iItem) {
        if ((h_target[iItem] != 0) && (h_target[iItem] != 1)) {
            LOG_F(logERROR, "calcAlgsEV(), h_target[%i] = %i. Expected target labels: 0 or 1.", iItem, h_target[iItem]);
            return LINEAR_SAMPLING_ERROR;
        }

        float target = static_cast<float>(2 * h_target[iItem] - 1);
        for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
            int error = h_EV[IDX2C(iItem, iAlg, nItems)];
            float score = error ? -target : target;
            h_scores[IDX2C(iItem, iAlg, nItems)] = score;
        }
    }

    CudaPtr2D<float> d_scores(nItems, nAlgs, h_scores);
    CudaPtr2D<int> d_target(nItems, 1, h_target);
    CudaPtr2D<int> d_EC;
    CudaPtr2D<unsigned int> d_hashes;
    if (h_EC != NULL)      d_EC.reset(nAlgs, 1);
    if (h_hashes != NULL) d_hashes.reset(nAlgs, 1);
    if ((d_EC != NULL) || (d_hashes != NULL)) {
        // Produce error counts.
        kCalcAlgs<<<nAlgs, BLOCK_SIZE>>>(d_scores, d_target, nItems, nAlgs, d_EC, d_hashes);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (h_EC != NULL)     CUDA_CHECK(cudaMemcpy(h_EC, d_EC, nAlgs * sizeof(int), cudaMemcpyDeviceToHost));
    if (h_hashes != NULL) CUDA_CHECK(cudaMemcpy(h_hashes, d_hashes, nAlgs * sizeof(unsigned int), cudaMemcpyDeviceToHost));    
    return 0;
}

// WARNING: Remember that all ***_device calls are async, and its OK to dispose memory only after cudaDeviceSynchronize().
int calcQEpsCombinatorial(
    int sessionId,
    const float *h_W,                    // nAlgs * nFeatures
    const unsigned char *h_isSource,     // nAlgs
    const float *epsValues,
    int nTrainItems,
    int nAlgs,
    int nEpsValues,
    int boundType,                       // type of bound; 0 = ESokolov bound, 1 = classic SC-bound, 2 = VC-type bound.
    float *h_QEps)                       // nAlgs * nEpsValues
{
    LOG_F(logINFO, "calcQEpsCombinatorial(): sessionId = %i, nTrainItems = %i, nAlgs = %i, nEpsValues = %i, boundType = %i", sessionId, nTrainItems, nAlgs, nEpsValues, boundType);
    CHECK(Recorder::getInstance().record_calcQEpsCombinatorial(sessionId, h_W, h_isSource, epsValues, nTrainItems, nAlgs, nEpsValues, boundType, h_QEps));

    if (h_W == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorial(), h_W == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_isSource == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorial(), h_isSource == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (boundType != 0 && boundType != 1 && boundType != 2) {
        LOG_F(logERROR, "calcQEpsCombinatorial(), boundType is set to %i, expected values are: 0, 1 or 2", boundType);
        return LINEAR_SAMPLING_ERROR;
    }

    if (boundType != 0) {
        LOG_F(logERROR, "calcQEpsCombinatorial(), boundType = %i is not yet implemented in GPU version of Linear Sampling library. Use boundType = 0.");
        return LINEAR_SAMPLING_ERROR;
    }

    CudaSession s = CudaSessionManager::getInstance().getSession(sessionId);
    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    if (nTrainItems <= 0 || nTrainItems >= s.nItems) {
        LOG_F(logERROR, "calcQEpsCombinatorial() was called with invalid nTrainItems = %i. Must be between 1 and %i.", nTrainItems, s.nItems - 1);
        return 0;
    }

    CUDA_CHECK(cudaSetDevice(s.deviceId));

    // Calc sources count
    int nSources = 0;
    for (int i = 0; i < nAlgs; ++i) {
        if (h_isSource[i]) ++nSources;
    }

    LOG_F(logINFO, "calcQEpsCombinatorial(): calculated nSources = %i", nSources);

    // Copy all sources into separate array.
    HostPtr2D<float> sourcesW(nSources, s.nFeatures);
    int iSource = 0;
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        if (!h_isSource[iAlg]) continue;

        for (int iFeature = 0; iFeature < s.nFeatures; ++iFeature) {
            sourcesW[IDX2C(iSource, iFeature, nSources)] = h_W[IDX2C(iAlg, iFeature, nAlgs)];
        }

        iSource++;
    }

    // Calculate algs connectivity
    CudaPtr2D<float> d_algsW(nAlgs, s.nFeatures, h_W);
    CudaPtr2D<unsigned char> d_algsEV(s.nItems, nAlgs);
    CudaPtr2D<int> d_algsEC(nAlgs, 1);
    CudaPtr2D<unsigned int> d_algsHashes(nAlgs, 1);
    CHECK(calcAlgs_device(s.id, d_algsW, nAlgs, NULL, d_algsEV, d_algsEC, d_algsHashes));
    CUDA_CHECK(cudaDeviceSynchronize());
    HostPtr2D<unsigned int> h_algsHashes(d_algsHashes);
    HostPtr2D<int> h_algsEC(d_algsEC);
    HostPtr2D<int> h_upperCon(nAlgs, 1);
    CHECK(calcAlgsConnectivity(h_algsHashes, h_algsEC, nAlgs, s.nItems, h_upperCon, NULL));
    CudaPtr2D<int> d_upperCon(h_upperCon);

    // Calculate sourcesEV on device
    CudaPtr2D<float> d_sourcesW(nSources, s.nFeatures, sourcesW);
    CudaPtr2D<unsigned char> d_sourcesEV(s.nItems, nSources);
    calcAlgs_device(s.id, d_sourcesW, nSources, NULL, d_sourcesEV, NULL, NULL);
    
    // Calculate d_QEps
    CudaPtr2D<int> d_ecAlgsNotSources(nAlgs, nSources);
    CudaPtr2D<int> d_ecSourcesNotAlgs(nAlgs, nSources);
    CudaPtr2D<float> d_QEps(nAlgs, nEpsValues);
    CHECK(compareAlgsToSources_device(d_algsEV, d_sourcesEV, nAlgs, nSources, s.nItems, d_ecAlgsNotSources, d_ecSourcesNotAlgs));

    CudaPtr2D<float> d_epsValues(nEpsValues, 1, epsValues);
    CHECK(calcAlgSourceQEps_device(d_algsEC, d_upperCon, d_ecAlgsNotSources, d_ecSourcesNotAlgs, *s.d_logFact, d_epsValues, s.nItems, nTrainItems, nAlgs, nSources, nEpsValues, d_QEps));

    CUDA_CHECK(cudaDeviceSynchronize());
        
    if (h_QEps != NULL) {
        if (h_QEps != NULL) CUDA_CHECK(cudaMemcpy(h_QEps, d_QEps, nAlgs * nEpsValues * sizeof(float), cudaMemcpyDeviceToHost));
    }

    return 0;
}

// Calculates overfitting based on combinatorial formulas. Uses Alexander Frey agorithm.
int calcQEpsCombinatorialAF(
    const unsigned char *h_algsEV,      // nItems * nAlgs       - error vectors
    const int *h_algsEC,       // nAlgs                - error counts
    const unsigned int *h_algsHashes,   // nAlgs                - hashes
    const unsigned char *h_isSource,    // nAlgs                - flags indicating whether algs are sources
    const float *epsValues,             // nEpsValues           - vector of thresholds "eps" to calculate the bound of QEps(eps)
    const int *h_clusterIds,            // nAlgs                - ids of clusters
    int nItems,                         //                      - number of items,
    int nTrainItems,                    //                      - ell, number of items for train sample
    int nAlgs,                          //                      - number of algorithms
    int nEpsValues,                     //                      - length of epsValues
    int nClusters,                      //                      - length of clusterIds
    float *h_QEps)                      // nClusters * nEpsValues - overfitting per algorithm and thresholds epsValues
{
    return LINEAR_SAMPLING_ERROR;
}

int calcQEpsCombinatorialEV(
    const unsigned char *h_algsEV,      // nItems * nAlgs
    const int *h_algsEC,       // nAlgs
    const unsigned int *h_algsHashes,   // nAlgs
    const unsigned char *h_isSource,    // nAlgs
    const float *epsValues,
    int nItems,
    int nTrainItems,
    int nAlgs,
    int nEpsValues,
    int boundType,
    float *h_QEps) 
{
    LOG_F(logINFO, "calcQEpsCombinatorialEV(): nItems = %i, nTrainItems = %i, nAlgs = %i, nEpsValues = %i, boundType = %i", nItems, nTrainItems, nAlgs, nEpsValues, boundType);
    CHECK(Recorder::getInstance().record_calcQEpsCombinatorialEV(h_algsEV, h_algsEC, h_algsHashes, h_isSource, epsValues, nItems, nTrainItems, nAlgs, nEpsValues, boundType, h_QEps));

    if (h_algsEV == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorialEV(), h_algsEV == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_algsEC == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorialEV(), h_algsEС == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_algsHashes == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorialEV(), h_algsHashes == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_isSource == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorialEV(), h_isSource == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (nTrainItems <= 0 || nTrainItems >= nItems) {
        LOG_F(logERROR, "calcQEpsCombinatorialEV() was called with invalid nTrainItems = %i. Must be between 1 and %i.", nTrainItems, nItems - 1);
        return 0;
    }

    if (boundType != 0 && boundType != 1 && boundType != 2) {
        LOG_F(logERROR, "calcQEpsCombinatorialEV(), boundType is set to %i, expected values are: 0, 1 or 2", boundType);
        return LINEAR_SAMPLING_ERROR;
    }

    if (boundType != 0) {
        LOG_F(logERROR, "calcQEpsCombinatorialEV(), boundType = %i is not yet implemented in GPU version of Linear Sampling library. Use boundType = 0.");
        return LINEAR_SAMPLING_ERROR;
    }

    // Calc algs upper connectivity
    HostPtr2D<int> h_upperCon(nAlgs, 1);
    CHECK(calcAlgsConnectivity(h_algsHashes, h_algsEC, nAlgs, nItems, h_upperCon, NULL));
    CudaPtr2D<int> d_upperCon(h_upperCon);

    // Calc sources count
    int nSources = 0;
    for (int i = 0; i < nAlgs; ++i) {
        if (h_isSource[i]) ++nSources;
    }
    
    // Create h_sourceEV, h_sourceEC, h_sourceHashes.
    HostPtr2D<unsigned char> h_sourcesEV(nItems, nSources);
    HostPtr2D<int> h_sourcesEC(nSources, 1);
    int iSource = 0;
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        if (!h_isSource[iAlg]) continue;

        for (int iItem = 0; iItem < nItems; ++iItem) {
            h_sourcesEV[IDX2C(iItem, iSource, nItems)] = h_algsEV[IDX2C(iItem, iAlg, nItems)];
        }

        iSource++;
    }

    // Create log factorial table.
    HostPtr2D<float> h_logFact(nItems + 1, 1);
    h_logFact[0] = 0;
    for (int i = 1; i <= nItems; ++i) h_logFact[i] = h_logFact[i - 1] + log((float)i);
    CudaPtr2D<float> d_logFact(nItems + 1, 1, h_logFact);

    CudaPtr2D<unsigned char> d_algsEV(nItems, nAlgs, h_algsEV);
    CudaPtr2D<int> d_algsEC(nAlgs, 1, h_algsEC);
    CudaPtr2D<unsigned char> d_sourcesEV(nItems, nSources, h_sourcesEV);

    CudaPtr2D<int> d_ecAlgsNotSources(nAlgs, nSources);
    CudaPtr2D<int> d_ecSourcesNotAlgs(nAlgs, nSources);
    CHECK(compareAlgsToSources_device(d_algsEV, d_sourcesEV, nAlgs, nSources, nItems, d_ecAlgsNotSources, d_ecSourcesNotAlgs));

    CudaPtr2D<float> d_QEps(nAlgs, nEpsValues);
    CudaPtr2D<float> d_epsValues(nEpsValues, 1, epsValues);
    CHECK(calcAlgSourceQEps_device(d_algsEC, d_upperCon, d_ecAlgsNotSources, d_ecSourcesNotAlgs, d_logFact, d_epsValues, nItems, nTrainItems, nAlgs, nSources, nEpsValues, d_QEps));

    CUDA_CHECK(cudaDeviceSynchronize());
        
    if (h_QEps != NULL) {
        if (h_QEps != NULL) CUDA_CHECK(cudaMemcpy(h_QEps, d_QEps, nAlgs * nEpsValues * sizeof(float), cudaMemcpyDeviceToHost));
    }

    return 0;
}

int closeSession(int sessionId)
{
    LOG_F(logINFO, "Close session: id=%i", sessionId);
    CHECK(Recorder::getInstance().record_closeSession(sessionId));
    CudaSessionManager::getInstance().closeSession(sessionId);
    return 0;
}

int closeAllSessions() 
{
    LOG_F(logINFO, "Close all sessions");
    CHECK(Recorder::getInstance().record_closeAllSessions());
    CudaSessionManager::getInstance().closeAllSessions();
    return 0;
}

// Creates session and outputs session id.
int createSession(const float *h_X, const int* h_target, const float *h_R, int nItems, int nFeatures, int nRays, int deviceId, int sessionId)
{
    LOG_F(logINFO, "Create session: nItems=%i, nFeatures=%i, nRays=%i, deviceId=%i", nItems, nFeatures, nRays, deviceId);

    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        LOG_F(logERROR, "No CUDA-compatible devices detected");
        return LINEAR_SAMPLING_ERROR;
    } else {
        LOG_F(logINFO, "cudaGetDeviceCount reports %i devices", deviceCount);
    }

    if (nRays > MAX_GRID_SIZE) 
    {
        LOG_F(logERROR, "Can't createSession nRays=%i. Current requirement: nRays <= %i", nRays, MAX_GRID_SIZE);
        return LINEAR_SAMPLING_ERROR;
    }

    if ((deviceId >= deviceCount) || (deviceId < 0))
    {
        LOG_F(logWARNING, "Unable to set deviceId to %i. Falling back to default device (0)", deviceId);
        deviceId = 0;
    }

    CUDA_CHECK(cudaSetDevice(deviceId));

    CudaPtr2D<float>::ptr d_X(new CudaPtr2D<float>(nItems, nFeatures, h_X));    // Feature matrix; rows = items, columns = features;
    CudaPtr2D<int>::ptr d_target(new CudaPtr2D<int>(nItems, 1, h_target));        // Vector of target classes (0 and 1).
    CudaPtr2D<float>::ptr d_R(new CudaPtr2D<float>(nRays, nFeatures, h_R));        // Matrix of search dirrections (rays); rows = rays, columns = features* (star means the "dual" space of hyperplanes);
    CudaPtr2D<float>::ptr d_XR(new CudaPtr2D<float>(nItems, nRays));            // Matrix product X * R'; rows = items, columns = rays;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // matrix multiplication of X and R
    CUBLAS_CHECK(cublasSgemm(CublasHandleSingleton::getInstance(), CUBLAS_OP_N, CUBLAS_OP_T, nItems, nRays, nFeatures, &alpha, *d_X, nItems, *d_R, nRays, &beta, *d_XR, nItems));

    // log factorial table.
    HostPtr2D<float> h_logFact(nItems + 1, 1);
    h_logFact[0] = 0;
    for (int i = 1; i <= nItems; ++i) {
        h_logFact[i] = h_logFact[i - 1] + log((float)i);
    }

    CudaPtr2D<float>::ptr d_logFact(new CudaPtr2D<float>(nItems + 1, 1, h_logFact));

    sessionId = CudaSessionManager::getInstance().createSession(d_X, d_target, d_R, d_XR, d_logFact, nItems, nFeatures, nRays, deviceId, sessionId);
    CHECK(Recorder::getInstance().record_createSession(sessionId, h_X, h_target, h_R, nItems, nFeatures, nRays, deviceId)); 
    return sessionId;
}

// Finds neighbours of w0 in cell arrangement.
int findNeighbors(int sessionId, const float *h_w0, float *h_W, float *h_t)
{
    LOG_F(logINFO, "findNeighbors(), sessionId = %i", sessionId);
    CHECK(Recorder::getInstance().record_findNeighbors(sessionId, h_w0, h_W, h_t));
    CudaSession s = CudaSessionManager::getInstance().getSession(sessionId);
    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    CUDA_CHECK(cudaSetDevice(s.deviceId));
    int nItems = s.nItems;
    int nFeatures = s.nFeatures;
    int nRays = s.nRays;

    CudaPtr2D<float> d_w0(nFeatures, 1, h_w0);            // Starting search point in feature* space.
    CudaPtr2D<float> d_Xw0(nItems, 1);                    // Matrix-vector product X * w';
    CudaPtr2D<float> d_t(nRays, 1);                        // Vector of nRays coefficients. Vector w[] = w0[] + t[i] * R[i][] is an adjecent to w0 in feature* space;
    CudaPtr2D<float> d_W(nRays, nFeatures);                // All found adjecent vectors; rows = rays, columns = features*;

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const int nW = 1;

    // matrix multiplication of X and w0
    CUBLAS_CHECK(cublasSgemv(CublasHandleSingleton::getInstance(), CUBLAS_OP_N, nItems, nFeatures, &alpha, *s.d_X, nItems, d_w0, 1, &beta, d_Xw0, 1));

    std::vector<int> h_rayId(nRays, 0), h_wId(nRays, 0);
    for (int i = 0; i < nRays; ++i) h_rayId[i] = i;
    
    CudaPtr2D<int> d_wId(nRays, 1, &h_wId[0]);
    CudaPtr2D<int> d_rayId(nRays, 1, &h_rayId[0]);

    // Produce d_t (vector of coeffieients t, indexed by rays)
    kFindNeighbors<<< nRays, BLOCK_SIZE >>>(*s.d_X, *s.d_R, *s.d_XR, d_w0, d_Xw0, d_rayId, d_wId, nItems, nFeatures, nRays, nW, d_t);
    
    // Produce d_W (matrix nRays * nFeatures with vectors of w0 neighbours)
    cublasSdgmm(CublasHandleSingleton::getInstance(), CUBLAS_SIDE_LEFT, nRays, nFeatures, *s.d_R, nRays, d_t, 1, d_W, nRays);
    kAddRowVector<<< GRID_SIZE, BLOCK_SIZE >>>(d_W, d_w0, d_W, nFeatures, nRays);

    CUDA_CHECK(cudaDeviceSynchronize());

    if (h_t != NULL) CUDA_CHECK(cudaMemcpy(h_t, d_t, nRays * sizeof(float), cudaMemcpyDeviceToHost));
    if (h_W != NULL) CUDA_CHECK(cudaMemcpy(h_W, d_W, nRays * nFeatures * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

// Finds random neighbors of hyperplanes w0.
// Output array:
//    - W is the matrix of adjacent vectors. Must be of size (nStartPoints*nIters) * nFeatures.
int findRandomNeighbors(
    int sessionId,
    const float *h_w0, 
    int n_w0,
    int randomSeed,
    float *h_W) 
{
    LOG_F(logINFO, "findRandomNeighbors(), sessionId = %i, randomSeed = %i", sessionId, randomSeed);
    CHECK(Recorder::getInstance().record_findRandomNeighbors(sessionId, h_w0, n_w0, randomSeed, h_W));
    CudaSession s = CudaSessionManager::getInstance().getSession(sessionId);
    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    CUDA_CHECK(cudaSetDevice(s.deviceId));
    int nItems = s.nItems;
    int nFeatures = s.nFeatures;
    int nRays = s.nRays;

    CudaPtr2D<float> d_w0(n_w0, nFeatures, h_w0);        // Starting search point in feature* space.
    CudaPtr2D<float> d_Xw0(nItems, n_w0);                // Matrix-vector product X * w';
    CudaPtr2D<float> d_t(n_w0, 1);                        // Vector of nRays coefficients. Vector w[] = w0[] + t[i] * R[i][] is an adjecent to w0 in feature* space;
    CudaPtr2D<float> d_W(n_w0, nFeatures);                // All found adjecent vectors; rows = rays, columns = features*;

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    
    // matrix multiplication of X and w0
    CUBLAS_CHECK(cublasSgemm(CublasHandleSingleton::getInstance(), CUBLAS_OP_N, CUBLAS_OP_T, nItems, n_w0, nFeatures, &alpha, *s.d_X, nItems, d_w0, n_w0, &beta, d_Xw0, nItems));

    // Initialize rand()
    if (randomSeed != 0) {
        srand(randomSeed);
    }

    std::vector<int> h_rayId(n_w0, 0), h_wId(n_w0, 0);
    for (int i = 0; i < n_w0; ++i) h_rayId[i] = rand() % nRays;
    for (int i = 0; i < n_w0; ++i) h_wId[i] = i;
    
    CudaPtr2D<int> d_wId(n_w0, 1, &h_wId[0]);
    CudaPtr2D<int> d_rayId(n_w0, 1, &h_rayId[0]);

    // Produce d_t (vector of coeffieients t, indexed by rays)
    kFindNeighbors<<< n_w0, BLOCK_SIZE >>>(*s.d_X, *s.d_R, *s.d_XR, d_w0, d_Xw0, d_rayId, d_wId, nItems, nFeatures, n_w0, n_w0, d_t);
    
    CUDA_CHECK(cudaDeviceSynchronize());

    HostPtr2D<float> h_R(nRays, nFeatures, *s.d_R);
    HostPtr2D<float> h_t(n_w0, 1, d_t);
    for (int iW = 0; iW < n_w0; ++iW)  {
        for (int iF = 0; iF < nFeatures; ++iF) {
            h_W[IDX2C(iW, iF, n_w0)] = h_w0[IDX2C(iW, iF, n_w0)] + h_R[IDX2C(h_rayId[iW], iF, nRays)] * h_t.get()[iW];
        }
    }

    // normalize
    for (int iW = 0; iW < n_w0; ++iW)  {
        double norm = 0.0;
        for (int iF = 0; iF < nFeatures; ++iF) {
            double cur = h_W[IDX2C(iW, iF, n_w0)];
            norm += (cur * cur);
        }

        float normf = static_cast<float>(sqrt(norm));
        for (int iF = 0; iF < nFeatures; ++iF) {
            h_W[IDX2C(iW, iF, n_w0)] /= normf;
        }
    }

    return 0;
}

// Returns session's statistics.
int getSessionStats(int sessionId, int *nItems, int *nFeatures, int *nRays, int *deviceId) {
    CudaSession s = CudaSessionManager::getInstance().getSession(sessionId);
    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    *nItems = s.nItems;
    *nFeatures = s.nFeatures;
    *nRays = s.nRays;
    *deviceId = s.deviceId;
    return 0;
}

int performCrossValidation(
    const unsigned char* h_EV, 
    const int* h_EC, 
    int nItems, 
    int nTrainItems,
    int nAlgs, 
    int nIters,
    int randomSeed,
    int* h_trainEC,
    int* h_testEC)
{
    LOG_F(logINFO, "performCrossValidation(), nItems = %i, nTrainItems = %i, nAlgs = %i, nIters = %i, randomSeed = %i", nItems, nTrainItems, nAlgs, nIters, randomSeed);
    CHECK(Recorder::getInstance().record_performCrossValidation(h_EV, h_EC, nItems, nTrainItems, nAlgs, nIters, randomSeed, h_trainEC, h_testEC));

    if (nItems > 8192) {
        LOG_F(logERROR, "PerformCrossValidation is called with nItems = %i. Current limitation: %i", nItems, 8192);
        return LINEAR_SAMPLING_ERROR;
    }

    if (nTrainItems <= 0 || nTrainItems >= nItems) {
        LOG_F(logERROR, "PerformCrossValidation is called with invalid nTrainItems = %i. Must be between 1 and %i.", nTrainItems, nItems - 1);
        return LINEAR_SAMPLING_ERROR;
    }

    if (nIters > MAX_GRID_SIZE) {
        LOG_F(logERROR, "PerformCrossValidation is called with nIteres = %i. Current limitation: %i", nIters, MAX_GRID_SIZE);
        return LINEAR_SAMPLING_ERROR;
    }

    int nItemsPadded = (int)(2 * BLOCK_SIZE * ceil((float)nItems / (float)(2 * BLOCK_SIZE)));

    CudaPtr2D<char> d_EV(nItems, nAlgs, (char *)h_EV);
    CudaPtr2D<int> d_EC(nAlgs, 1, h_EC);
    CudaPtr2D<int> d_trainEC(nIters, 1);
    CudaPtr2D<int> d_testEC(nIters, 1);

    kPerformCrossValidation<<<nIters, BLOCK_SIZE, sizeof(char) * nItemsPadded>>>(d_EV, d_EC, nItems, nTrainItems, nAlgs, nIters, nItemsPadded, randomSeed, d_trainEC, d_testEC);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (h_trainEC != NULL) CUDA_CHECK(cudaMemcpy(h_trainEC, d_trainEC, nIters * sizeof(int), cudaMemcpyDeviceToHost));
    if (h_testEC != NULL) CUDA_CHECK(cudaMemcpy(h_testEC, d_testEC, nIters * sizeof(int), cudaMemcpyDeviceToHost));

    return 0;
}

// =========== Debugging functions =========== 

int getXR(int sessionId, float *h_XR) {
    CudaSession s = CudaSessionManager::getInstance().getSession(sessionId);
    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    CUDA_CHECK(cudaSetDevice(s.deviceId));
    if (h_XR != NULL) CUDA_CHECK(cudaMemcpy(h_XR, *s.d_XR, s.nItems * s.nRays* sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

// ===========================================