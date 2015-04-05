// This file contains implementation of CPU-functions, which takes CPU-data pointers.

#include "ls_api.h"

#include <vector>

#include "common.h"
#include "ls_api_cpu_intl.h"
#include "Log.h"
#include "recorder.h"
#include "Ptr.h"

using namespace helpers;

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
    CpuSession s = CpuSessionManager::getInstance().getSession(sessionId);
    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    if ((h_scores == NULL) && (h_EV == NULL) && (h_EC == NULL) && (h_hashes == NULL)) {
        LOG_F(logWARNING, "calcAlgs() was called with all arguments set to NULL.");
        return 0;
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Handle the case when d_scores == NULL, but d_EV or d_EC or d_hashes are not.
    float* h_scores_notnull;
    Ptr<float> h_scores_ptr;
    if (h_scores != NULL) {
        h_scores_notnull = h_scores;
    } else {
        // this will temporary allocate memory, and free after calcAlgs is completed.
        h_scores_ptr.resize(s.nItems, nAlgs);
        h_scores_notnull = h_scores_ptr.get();
    }

    for (int iItem = 0; iItem < s.nItems; ++iItem) {
        for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
            float score = 0.0f;
            for (int iFeature = 0; iFeature < s.nFeatures; ++iFeature) {
                score += (s.get_X())[IDX2C(iItem, iFeature, s.nItems)] * (h_W[IDX2C(iAlg, iFeature, nAlgs)]);
            }

            h_scores_notnull[IDX2C(iItem, iAlg, s.nItems)] = score;
        }
    }

    if ((h_EC != NULL) || (h_hashes != NULL)) {
        // Produce error counts.
        kCalcAlgs(h_scores_notnull, s.get_target(), s.nItems, nAlgs, h_EC, h_hashes);
    }

    if (h_EV != NULL) {
        kCalcEV(h_scores_notnull, s.get_target(), s.nItems, nAlgs, h_EV);
    }

    return 0;
}

int calcAlgsEV(
    const unsigned char* h_EV,
    const int* h_target,
    int nItems,
    int nAlgs,
    int* h_EC,
    unsigned int* h_hashes)
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

    // Produce scores.
    Ptr<float> h_scores(nItems, nAlgs);
    for (int iItem = 0; iItem < nItems; ++iItem) {
        if ((h_target[iItem] != 0) && (h_target[iItem] != 1)) {
            LOG_F(logERROR, "calcAlgsEV(), h_target[%i] = %i. Expected target labels: 0 or 1.", iItem, h_target[iItem]);
            return LINEAR_SAMPLING_ERROR;
        }

        float target = static_cast<float>(2 * h_target[iItem] - 1);
        for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
            int error = h_EV[IDX2C(iItem, iAlg, nItems)];
            float score = error ? -target : target;
            h_scores.get()[IDX2C(iItem, iAlg, nItems)] = score;
        }
    }

    if ((h_EC != NULL) || (h_hashes != NULL)) {
        // Produce error counts.
        kCalcAlgs(h_scores.get(), h_target, nItems, nAlgs, h_EC, h_hashes);
    }

    return 0;
}

int calcQEpsCombinatorial(
    int sessionId,
    const float *h_W,
    const unsigned char *h_isSource,
    const float *epsValues,
    int nTrainItems,
    int nAlgs,
    int nEpsValues,
    int boundType,
    float *h_QEps)
{
    LOG_F(logINFO, "calcQEpsCombinatorial(): sessionId = %i, nTrainItems = %i, nAlgs = %i, nEpsValues = %i, boundType = %i", sessionId, nTrainItems, nAlgs, nEpsValues, boundType);
    CHECK(Recorder::getInstance().record_calcQEpsCombinatorial(sessionId, h_W, h_isSource, epsValues, nTrainItems, nAlgs, nEpsValues, boundType, h_QEps));
    CpuSession s = CpuSessionManager::getInstance().getSession(sessionId);

    const float* logFact = (*s.d_logFact).get();

    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    if (h_W == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorial(), h_W == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_isSource == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorial(), h_isSource == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_QEps == NULL) {
        LOG_F(logWARNING, "calcQEpsCombinatorial(), h_QEps == NULL");
        return 0;
    }

    if (nTrainItems <= 0 || nTrainItems >= s.nItems) {
        LOG_F(logERROR, "calcQEpsCombinatorial() was called with invalid nTrainItems = %i. Must be between 1 and %i.", nTrainItems, s.nItems - 1);
        return 0;
    }

    if (boundType != 0 && boundType != 1 && boundType != 2) {
        LOG_F(logERROR, "calcQEpsCombinatorial(), boundType is set to %i, expected values are: 0, 1 or 2", boundType);
        return LINEAR_SAMPLING_ERROR;
    }

    if (boundType == 2) {
        // Calc simple VC-type bound, equivalent to SC-bound with upperCon=0 and inferiority=0 (for all algs).
        Ptr<int> h_upperCon(nAlgs, 1);
        Ptr<int> h_algsInferiority(nAlgs, 1);
        Ptr<int> h_EC(nAlgs);
        memset(h_upperCon.get(), 0, sizeof(int) * nAlgs);
        memset(h_algsInferiority.get(), 0, sizeof(int) * nAlgs);
        CHECK(calcAlgs(sessionId, h_W, nAlgs, NULL, NULL, h_EC.get(), NULL));
        CHECK(kCalcAlgSourceQEpsClassicSC(h_EC.get(), h_upperCon.get(), h_algsInferiority.get(), s.get_logFact(), epsValues, s.nItems, nTrainItems, nAlgs, nEpsValues, h_QEps));
        return 0;
    }

    // Calc sources count
    int nSources = 0;
    for (int i = 0; i < nAlgs; ++i) {
        if (h_isSource[i]) ++nSources;
    }

    LOG_F(logINFO, "calcQEpsCombinatorial(): calculated nSources = %i", nSources);

    Ptr<int> algIds(nAlgs);
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        algIds.get()[iAlg] = iAlg;
    }

    Ptr<int> sourceIds(nSources);
    for (int iAlg = 0, iSource = 0; iAlg < nAlgs; ++iAlg) {
        if (h_isSource[iAlg]) sourceIds.get()[iSource++] = iAlg;
    }

    Ptr<unsigned char> h_EV(s.nItems, nAlgs);
    Ptr<int> h_EC(nAlgs);
    Ptr<unsigned int> h_hashes(nAlgs);
    CHECK(calcAlgs(sessionId, h_W, nAlgs, NULL, h_EV.get(), h_EC.get(), h_hashes.get()));

    Ptr<int> h_upperCon(nAlgs);
    CHECK(calcAlgsConnectivity(h_hashes.get(), h_EC.get(), nAlgs, s.nItems, h_upperCon.get(), NULL));

    // Calculate QEps
    if (boundType == 0) {
        Ptr<int> h_ecAlgsNotSources(nAlgs, nSources);
        Ptr<int> h_ecSourcesNotAlgs(nAlgs, nSources);
        kCompareAlgsToSources(h_EV.get(), algIds.get(), sourceIds.get(), nAlgs, nSources, s.nItems, nAlgs, h_ecAlgsNotSources.get(), h_ecSourcesNotAlgs.get(), NULL);
        kCalcAlgSourceQEpsESokolov(h_EC.get(), h_upperCon.get(), h_ecAlgsNotSources.get(), h_ecSourcesNotAlgs.get(), s.get_logFact(), epsValues, s.nItems, nTrainItems, nAlgs, nSources, nEpsValues, h_QEps);
    } else {
        Ptr<int> h_algsInferiority(nAlgs, 1);
        kCompareAlgsToSources(h_EV.get(), algIds.get(), sourceIds.get(), nAlgs, nSources, s.nItems, nAlgs, NULL, NULL, h_algsInferiority.get());
        CHECK(kCalcAlgSourceQEpsClassicSC(h_EC.get(), h_upperCon.get(), h_algsInferiority.get(), s.get_logFact(), epsValues, s.nItems, nTrainItems, nAlgs, nEpsValues, h_QEps));
    }

    return 0;
}

int calcQEpsCombinatorialAF(
    const unsigned char *h_algsEV,
    const int *h_algsEC,
    const unsigned int *h_algsHashes,
    const unsigned char *h_isSource,
    const float *epsValues,
    const int *h_clusterIds,
    int nItems,
    int nTrainItems,
    int nAlgs,
    int nEpsValues,
    int nClusters,
    float *h_QEps)
{
    LOG_F(logINFO, "calcQEpsCombinatorialAF(): nItems=%i, nTrainItems=%i, nAlgs=%i, nEpsValues = %i, nClusters = %i", nItems, nTrainItems, nAlgs, nEpsValues, nClusters);
    CHECK(Recorder::getInstance().record_calcQEpsCombinatorialAF(h_algsEV, h_algsEC, h_algsHashes, h_isSource, epsValues, h_clusterIds, nItems, nTrainItems, nAlgs, nEpsValues, nClusters));

    if (h_algsEV == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorialAF(), h_algsEV == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_algsEC == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorialAF(), h_algsEC == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_algsHashes == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorialAF(), h_algsHashes == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_isSource == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorialAF(), h_isSource == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_clusterIds == NULL) {
        LOG_F(logERROR, "calcQEpsCombinatorialAF(), h_clusterIds == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_QEps == NULL) {
        LOG_F(logWARNING, "calcQEpsCombinatorialAF(), h_QEps == NULL");
        return 0;
    }

    if (nTrainItems <= 0 || nTrainItems >= nItems) {
        LOG_F(logERROR, "calcQEpsCombinatorialAF() was called with invalid nTrainItems = %i. Must be between 1 and %i.", nTrainItems, nItems - 1);
        return 0;
    }

    // ToDo: check that clusterIds is a valid clustering
    //  1. All values are in 0 ... (nClusters - 1)
    //  2. Each cluster has at least one algorithm.
    //  3. Withing each cluster all algs has consistent number of errors.
    Ptr<int> clustersEC(nClusters);
    Ptr<int> clusterSize(nClusters);
    memset(clusterSize.get(), 0, sizeof(int) * nClusters);
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        if (h_clusterIds[iAlg] >= nClusters) {
            LOG_F(logERROR, "calcQEpsCombinatorialAF() has received malformed clusterIds. clusterIds[%i] = %i >= %i = nClusters", iAlg, h_clusterIds[iAlg], nClusters);
            return LINEAR_SAMPLING_ERROR;
        }

        int clusterId = h_clusterIds[iAlg];
        if (clusterSize.get()[clusterId] == 0) {
            clustersEC.get()[clusterId] = h_algsEC[iAlg];
        } else {
            if (clustersEC.get()[clusterId] != h_algsEC[iAlg]) {
                LOG_F(logERROR, "calcQEpsCombinatorialAF() has received malformed clusterIds. Cluster %i contains algs with different count of errors.", clusterId);
                return LINEAR_SAMPLING_ERROR;
            }
        }
        clusterSize.get()[clusterId]++;

    }

    for (int i = 0; i < nClusters; ++i) {
        if (clusterSize.get()[i] == 0) {
            LOG_F(logERROR, "calcQEpsCombinatorialAF() has received malformed clusterIds. Cluster %i is empty.", i);
            return LINEAR_SAMPLING_ERROR;
        }
    }



    // Calc sources count
    int nSources = 0;
    for (int i = 0; i < nAlgs; ++i) {
        if (h_isSource[i]) ++nSources;
    }

    LOG_F(logINFO, "calcQEpsCombinatorialAF(): calculated nSources = %i", nSources);

    Ptr<int> sourceIds(nSources);
    for (int iAlg = 0, iSource = 0; iAlg < nAlgs; ++iAlg) {
        if (h_isSource[iAlg]) sourceIds.get()[iSource++] = iAlg;
    }

    // error count, upper connectivity and inferriority per cluster.
    Ptr<int> clustersUpperCon(nClusters), clustersInferiority(nClusters), clustersCommonErrors(nClusters), clustersUnionErrors(nClusters);
    kAnalyzeClusters(h_algsEV, h_clusterIds, sourceIds.get(), nAlgs, nSources, nItems, nClusters, clustersInferiority.get(), clustersCommonErrors.get(), clustersUnionErrors.get());

    // Create log factorial table.
    Ptr<float> h_logFact(nItems + 1, 1);
    h_logFact.get()[0] = 0;
    for (int i = 1; i <= nItems; ++i) h_logFact.get()[i] = h_logFact.get()[i - 1] + log((float)i);

    // Calc algs upper connectivity and manualy set it to zero for clasters with >1 alg.
    Ptr<int> h_upperCon(nAlgs, 1);
    CHECK(calcAlgsConnectivity(h_algsHashes, h_algsEC, nAlgs, nItems, h_upperCon.get(), NULL));
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        int clusterId = h_clusterIds[iAlg];

        // "The right" way according to theoretical results.
        // clustersUpperCon[clusterId] = (clusterSize[clusterId] > 1) ? 0 : h_upperCon[iAlg];

        // Heuristic that works better:
        clustersUpperCon.get()[clusterId] = std::max(clustersUpperCon.get()[clusterId], h_upperCon.get()[iAlg]);
    }

    CHECK(kCalcAlgSourceQEpsAFrey(
        clustersEC.get(),
        clustersUpperCon.get(),
        clustersInferiority.get(),
        clustersCommonErrors.get(),
        clustersUnionErrors.get(),
        clusterSize.get(),
        h_logFact.get(),
        epsValues,
        nItems,
        nTrainItems,
        nClusters,
        nEpsValues,
        h_QEps));

    return SUCCESS;
}

int calcQEpsCombinatorialEV(
    const unsigned char *h_algsEV,
    const int *h_algsEC,
    const unsigned int *h_algsHashes,
    const unsigned char *h_isSource,
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
        LOG_F(logERROR, "calcQEpsCombinatorialEV(), h_algsEC == NULL");
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

    if (h_QEps == NULL) {
        LOG_F(logWARNING, "calcQEpsCombinatorialEV(), h_QEps == NULL");
        return 0;
    }

   if (nTrainItems <= 0 || nTrainItems >= nItems) {
        LOG_F(logERROR, "calcQEpsCombinatorialEV() was called with invalid nTrainItems = %i. Must be between 1 and %i.", nTrainItems, nItems - 1);
        return 0;
    }

    if (boundType != 0 && boundType != 1 && boundType != 2) {
        LOG_F(logERROR, "calcQEpsCombinatorialEV(), boundType is set to %i, expected values are: 0, 1 or 2", boundType);
        return LINEAR_SAMPLING_ERROR;
    }

    // Create log factorial table.
    Ptr<float> h_logFact(nItems + 1, 1);
    h_logFact.get()[0] = 0;
    for (int i = 1; i <= nItems; ++i) h_logFact.get()[i] = h_logFact.get()[i - 1] + log((float)i);

    if (boundType == 2) {
        // Calc simple VC-type bound, equivalent to SC-bound with upperCon=0 and inferiority=0 (for all algs).
        Ptr<int> h_upperCon(nAlgs, 1);
        Ptr<int> h_algsInferiority(nAlgs, 1);
        memset(h_upperCon.get(), 0, sizeof(int) * nAlgs);
        memset(h_algsInferiority.get(), 0, sizeof(int) * nAlgs);
        CHECK(kCalcAlgSourceQEpsClassicSC(h_algsEC, h_upperCon.get(), h_algsInferiority.get(), h_logFact.get(), epsValues, nItems, nTrainItems, nAlgs, nEpsValues, h_QEps));
        return 0;
    }

    // Calc algs upper connectivity
    Ptr<int> h_upperCon(nAlgs, 1);
    CHECK(calcAlgsConnectivity(h_algsHashes, h_algsEC, nAlgs, nItems, h_upperCon.get(), NULL));

    // Calc sources count
    int nSources = 0;
    for (int i = 0; i < nAlgs; ++i) {
        if (h_isSource[i]) ++nSources;
    }

    LOG_F(logINFO, "calcQEpsCombinatorialEV(): calculated nSources = %i", nSources);

    Ptr<int> algIds(nAlgs);
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        algIds.get()[iAlg] = iAlg;
    }

    Ptr<int> sourceIds(nSources);
    for (int iAlg = 0, iSource = 0; iAlg < nAlgs; ++iAlg) {
        if (h_isSource[iAlg]) sourceIds.get()[iSource++] = iAlg;
    }

    // Calculate QEps
    if (boundType == 0) {
        Ptr<int> h_ecAlgsNotSources(nAlgs, nSources);
        Ptr<int> h_ecSourcesNotAlgs(nAlgs, nSources);
        kCompareAlgsToSources(h_algsEV, algIds.get(), sourceIds.get(), nAlgs, nSources, nItems, nAlgs, h_ecAlgsNotSources.get(), h_ecSourcesNotAlgs.get(), NULL);
        kCalcAlgSourceQEpsESokolov(h_algsEC, h_upperCon.get(), h_ecAlgsNotSources.get(), h_ecSourcesNotAlgs.get(), h_logFact.get(), epsValues, nItems, nTrainItems, nAlgs, nSources, nEpsValues, h_QEps);
    } else {
        Ptr<int> h_algsInferiority(nAlgs, 1);
        kCompareAlgsToSources(h_algsEV, algIds.get(), sourceIds.get(), nAlgs, nSources, nItems, nAlgs, NULL, NULL, h_algsInferiority.get());
        CHECK(kCalcAlgSourceQEpsClassicSC(h_algsEC, h_upperCon.get(), h_algsInferiority.get(), h_logFact.get(), epsValues, nItems, nTrainItems, nAlgs, nEpsValues, h_QEps));
    }

    return 0;
}

int closeSession(int sessionId)
{
    LOG_F(logINFO, "Close session: id=%i", sessionId);
    CHECK(Recorder::getInstance().record_closeSession(sessionId));
    CpuSessionManager::getInstance().closeSession(sessionId);
    return 0;
}

int closeAllSessions()
{
    LOG_F(logINFO, "Close all sessions");
    CHECK(Recorder::getInstance().record_closeAllSessions());
    CpuSessionManager::getInstance().closeAllSessions();
    return 0;
}

int createSession(const float *h_X, const int* h_target, const float *h_R, int nItems, int nFeatures, int nRays, int deviceId, int sessionId)
{
    LOG_F(logINFO, "Create session: nItems=%i, nFeatures=%i, nRays=%i, deviceId=%i", nItems, nFeatures, nRays, deviceId);

    Ptr<float>::ptr d_X(new Ptr<float>(nItems, nFeatures));    // Feature matrix; rows = items, columns = features;
    Ptr<int>::ptr d_target(new Ptr<int>(nItems, 1));           // Vector of target classes (0 and 1).
    Ptr<float>::ptr d_R(new Ptr<float>(nRays, nFeatures));     // Matrix of search dirrections (rays); rows = rays, columns = features* (star means the "dual" space of hyperplanes);
    Ptr<float>::ptr d_XR(new Ptr<float>(nItems, nRays));       // Matrix product X * R'; rows = items, columns = rays;

    memcpy((*d_X).get(), h_X, sizeof(float) * nItems * nFeatures);
    memcpy((*d_target).get(), h_target, sizeof(int) * nItems);
    memcpy((*d_R).get(), h_R, sizeof(float) * nRays * nFeatures);
    memset((*d_XR).get(), 0, sizeof(float) * nItems * nRays);

    // Calc XR as matrix product
    for (int i = 0; i < nItems; i++) {
        for (int j = 0; j < nRays; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < nFeatures; ++k) {
                sum += h_X[IDX2C(i, k, nItems)] * h_R[IDX2C(j, k, nRays)];
            }

            (*d_XR).get()[IDX2C(i, j, nItems)] = sum;
        }
    }

    // log factorial table.
    Ptr<float>::ptr d_logFact(new Ptr<float>(nItems + 1, 1));
    (*d_logFact).get()[0] = 0;
    for (int i = 1; i <= nItems; ++i) {
        (*d_logFact).get()[i] = (*d_logFact).get()[i - 1] + log((float)i);
    }

    sessionId = CpuSessionManager::getInstance().createSession(d_X, d_target, d_R, d_XR, d_logFact, nItems, nFeatures, nRays, deviceId, sessionId);
    CHECK(Recorder::getInstance().record_createSession(sessionId, h_X, h_target, h_R, nItems, nFeatures, nRays, deviceId));
    return sessionId;
}

int findNeighbors(int sessionId, const float *h_w0, float *h_W, float *h_t)
{
    LOG_F(logINFO, "findNeighbors(), sessionId = %i", sessionId);
    CHECK(Recorder::getInstance().record_findNeighbors(sessionId, h_w0, h_W, h_t));
    CpuSession s = CpuSessionManager::getInstance().getSession(sessionId);
    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    if ((h_t == NULL) && (h_W == NULL)) {
        LOG_F(logWARNING, "findNeighbors() was called with all arguments set to NULL.");
        return 0;
    }

    Ptr<float> d_Xw0(s.nItems, 1); // Matrix-vector product X * w';
    for (int iItem = 0; iItem < s.nItems; ++iItem) {
        float value = 0.0f;
        for (int iFeature = 0; iFeature < s.nFeatures; ++iFeature) {
            value += s.get_X()[IDX2C(iItem, iFeature, s.nItems)] * h_w0[iFeature];
        }

        d_Xw0.get()[iItem] = value;
    }

    std::vector<int> h_rayId(s.nRays, 0), h_wId(s.nRays, 0);
    for (int i = 0; i < s.nRays; ++i) h_rayId[i] = i;

    // Produce d_t (vector of coeffieients t, indexed by rays)
    Ptr<float> d_t(s.nRays, 1);   // Vector of nRays coefficients. Vector w[] = w0[] + t[i] * R[i][] is an adjecent to w0 in feature* space;
    kFindNeighbors(s.get_X(), s.get_R(), s.get_XR(), h_w0, d_Xw0.get(), &h_rayId[0], &h_wId[0], s.nItems, s.nFeatures, s.nRays, 1, d_t.get());
    if (h_t != NULL) memcpy(h_t, d_t.get(), sizeof(float) * s.nRays);

    if (h_W != NULL) {
        // Produce d_W (matrix nRays * nFeatures with vectors of w0 neighbours)
        for (int iW = 0; iW < s.nRays; ++iW)  {
            for (int iF = 0; iF < s.nFeatures; ++iF) {
                h_W[IDX2C(iW, iF, s.nRays)] = h_w0[iF] + s.get_R()[IDX2C(h_rayId[iW], iF, s.nRays)] * d_t.get()[iW];
            }
        }
    }

    return 0;
}

int findRandomNeighbors(
    int sessionId,
    const float *h_w0,
    int n_w0,
    int randomSeed,
    float *h_W)
{
    LOG_F(logINFO, "findRandomNeighbors(), sessionId = %i, randomSeed = %i", sessionId, randomSeed);
    CHECK(Recorder::getInstance().record_findRandomNeighbors(sessionId, h_w0, n_w0, randomSeed, h_W));
    CpuSession s = CpuSessionManager::getInstance().getSession(sessionId);
    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    Ptr<float> d_Xw0(s.nItems, n_w0); // Matrix-matrix product X * w0';
    for (int iItem = 0; iItem < s.nItems; ++iItem) {
        for (int iAlg = 0; iAlg < n_w0; ++iAlg) {
            float value = 0.0f;
            for (int iFeature = 0; iFeature < s.nFeatures; ++iFeature) {
                value += s.get_X()[IDX2C(iItem, iFeature, s.nItems)] * h_w0[IDX2C(iAlg, iFeature, n_w0)];
            }

            d_Xw0.get()[IDX2C(iItem, iAlg, s.nItems)] = value;
        }
    }

    // Initialize rand()
    if (randomSeed != 0) {
        srand(randomSeed);
    }

    std::vector<int> h_rayId(n_w0, 0), h_wId(n_w0, 0);
    for (int i = 0; i < n_w0; ++i) h_rayId[i] = rand() % s.nRays;
    for (int i = 0; i < n_w0; ++i) h_wId[i] = i;
    Ptr<float> d_t(n_w0, 1);        // Vector of nRays coefficients. Vector w[] = w0[] + t[i] * R[i][] is an adjecent to w0 in feature* space;
    kFindNeighbors(s.get_X(), s.get_R(), s.get_XR(), h_w0, d_Xw0.get(), &h_rayId[0], &h_wId[0], s.nItems, s.nFeatures, n_w0, n_w0, d_t.get());

    if (h_W != NULL) {
        memset(h_W, 0, n_w0 * s.nFeatures);
        for (int iW = 0; iW < n_w0; ++iW)  {
            for (int iF = 0; iF < s.nFeatures; ++iF) {
                h_W[IDX2C(iW, iF, n_w0)] = h_w0[IDX2C(iW, iF, n_w0)] + s.get_R()[IDX2C(h_rayId[iW], iF, s.nRays)] * d_t.get()[iW];
            }
        }

        // normalize
        for (int iW = 0; iW < n_w0; ++iW)  {
            double norm = 0.0;
            for (int iF = 0; iF < s.nFeatures; ++iF) {
                double cur = h_W[IDX2C(iW, iF, n_w0)];
                norm += (cur * cur);
            }

            float normf = static_cast<float>(sqrt(norm));
            for (int iF = 0; iF < s.nFeatures; ++iF) {
                h_W[IDX2C(iW, iF, n_w0)] /= normf;
            }
        }
    }

    return 0;
}

int getSessionStats(int sessionId, int *nItems, int *nFeatures, int *nRays, int *deviceId) {
    CpuSession s = CpuSessionManager::getInstance().getSession(sessionId);
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

    if (h_trainEC == NULL && h_testEC == NULL) {
        LOG_F(logWARNING, "performCrossValidation() was called with h_trainEC and h_testEC set to NULL.");
        return 0;
    }

    if (nTrainItems <= 0 || nTrainItems >= nItems) {
        LOG_F(logERROR, "performCrossValidation() was called with invalid nTrainItems = %i. Must be between 1 and %i.", nTrainItems, nItems - 1);
        return 0;
    }

    kPerformCrossValidation(h_EV, h_EC, nItems, nTrainItems, nAlgs, nIters, randomSeed, h_trainEC, h_testEC);
    return 0;
}

// =========== Debugging functions ===========

int getXR(int sessionId, float *h_XR) {
    CpuSession s = CpuSessionManager::getInstance().getSession(sessionId);
    if (!s.isInitialized()) return LINEAR_SAMPLING_ERROR;

    if (h_XR != NULL) memcpy(h_XR, s.get_XR(), s.nItems * s.nRays* sizeof(float));
    return 0;
}

// ===========================================