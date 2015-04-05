// Contains shared implementation for GPU and CPU version.
#include "ls_api.h"

#include <hash_map>
#include <hash_set>
#include <map>
#include <memory>
#include <queue>
#include <vector>

#include "common.h"
#include "Log.h"
#include "Ptr.h"
#include "recorder.h"
#include "replayer.h"
#include "SourcesContainer.h"

// Calculates lower and upper connectivity for classifiers, base on their hashes.
int calcAlgsConnectivity(
    const unsigned int* h_algsHashes,
    const int* h_algsEC,
    int nAlgs,
    int nItems,
    int* upperCon,
    int* lowerCon)
{
    LOG_F(logINFO, "calcAlgsConnectivity(), nAlgs = %i, nItems = %i...", nAlgs, nItems );
    CHECK(Recorder::getInstance().record_calcAlgsConnectivity(h_algsHashes, h_algsEC, nAlgs, nItems, upperCon, lowerCon));

    if (h_algsHashes == NULL) {
        LOG_F(logERROR, "calcAlgsConnectivity(), h_algsHashes == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    if (h_algsEC == NULL) {
        LOG_F(logERROR, "calcAlgsConnectivity(), h_algsEC == NULL");
    }

    std::multimap<unsigned int, int> known_algs;
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        known_algs.insert(std::pair<unsigned int, int>(h_algsHashes[iAlg], iAlg));
    }

    const int sign[2] = {-1, 1};
    for (auto iter = known_algs.begin(); iter != known_algs.end(); ++iter) {
        unsigned int hash = 1;
        int tmpUpperCon = 0;
        int tmpLowerCon = 0;
        int iterAlgEC = h_algsEC[iter->second];
        for (int iItem = 0; iItem < nItems; ++iItem) {
            for (int iSign = 0; iSign < 2; ++iSign) {
                auto neighbourAlgIter = known_algs.find(iter->first + sign[iSign] * hash);
                if (neighbourAlgIter != known_algs.end()) {
                    int neighbourEC = h_algsEC[neighbourAlgIter -> second];
                    if (neighbourEC == (iterAlgEC + 1)) {
                        tmpUpperCon++;
                    } else if (neighbourEC == (iterAlgEC - 1)) {
                        tmpLowerCon++;
                    } else {
                        // collision happened; ignore.
                    }
                }
            }

            hash *= 23;
        }

        if (upperCon != NULL) upperCon[iter->second] = tmpUpperCon;
        if (lowerCon != NULL) lowerCon[iter->second] = tmpLowerCon;
    }

    LOG_F(logINFO, "calcAlgsConnectivity(), calc algs connectivity done.");
    return 0;
}

// Detect clusters. 
// Current implementation: group pairs of classifiers (a, b) when \rho(a, b) = 2 and m(a, \XX) = m(b, \XX).
int detectClusters(
    const unsigned char *h_algsEV,
    const int *h_algsEC,
    int nItems,
    int nAlgs,
    int* clusterIds) 
{
    LOG_F(logINFO, "Detect clusters: nItems=%i, nAlgs=%i", nItems, nAlgs);

    int nextClusterId = 0;
    for (int i = 0; i < nAlgs; ++i) clusterIds[i] = -1;

    // Group all algs with the same number of errors.
    std::hash_map<int, std::shared_ptr<std::vector<int>>> ec_to_algIds_map;
    typedef std::pair<int, std::shared_ptr<std::vector<int>>> KeyValuePair;
    for (int i = 0; i < nAlgs; ++i) {
        auto iter = ec_to_algIds_map.find(h_algsEC[i]);
        if (iter == ec_to_algIds_map.end()) {
            iter = ec_to_algIds_map.insert(KeyValuePair(h_algsEC[i], std::make_shared<std::vector<int>>())).first;
        }

        iter->second->push_back(i);
    }

    for (auto iter = ec_to_algIds_map.begin(); iter != ec_to_algIds_map.end(); ++iter) {
        int errorCount = iter->first; // number of errors
        std::vector<int>& vec = *(iter->second); // vector of alg ids with errorCount errors;
        std::hash_set<int> processedAlgs;
        for (int i = 0; i < vec.size() - 1; ++i) {
            if (processedAlgs.find(vec[i]) != processedAlgs.end()) continue;
            for (int j = i + 1; j < vec.size(); ++j) {
                if (processedAlgs.find(vec[j]) != processedAlgs.end()) continue;

                int rho = 0;
                for (int iItem = 0; iItem < nItems; ++iItem) {
                    if (h_algsEV[IDX2C(iItem, vec[i], nItems)] != h_algsEV[IDX2C(iItem, vec[j], nItems)]) rho++;
                    if (rho > 2) break;
                }

                if (rho == 2) {
                    // join algs into cluster
                    processedAlgs.insert(vec[i]);
                    processedAlgs.insert(vec[j]);
                    clusterIds[vec[i]] = nextClusterId;
                    clusterIds[vec[j]] = nextClusterId;
                    nextClusterId++;
                }
            }
        }
    }

    for (int i = 0; i < nAlgs; ++i) {
        if (clusterIds[i] == -1) {
            clusterIds[i] = nextClusterId++;            
        }
    }   

    return SUCCESS;
}

int findAllNeighbors(
    int sessionId,
    const float *h_w0,
    int maxAlgs,
    int maxIterations,
    int nErrorsLimit,
    float *h_W)
{
    LOG_F(logINFO, "findAllNeighbors(): maxAlgs=%i, maxIterations=%i, nErrorsLimit=%i", maxAlgs, maxIterations, nErrorsLimit);
    CHECK(Recorder::getInstance().record_findAllNeighbors(sessionId, h_w0, maxAlgs, maxIterations, nErrorsLimit, h_W));

    int nItems, nFeatures, nRays, deviceId;
    CHECK(getSessionStats(sessionId, &nItems, &nFeatures, &nRays, &deviceId));

    unsigned int w0_hash;
    CHECK(calcAlgs(sessionId, h_w0, 1, NULL, NULL, NULL, &w0_hash));

    std::queue<unsigned int> taskQueue;
    std::map<unsigned int, int> knownAlgs;

    int algsFound = 1; // starting with w0.
    for (int i = 0; i < nFeatures; ++i) {
        h_W[IDX2C(0, i, maxAlgs)] = h_w0[i];
    }

    knownAlgs.insert(std::pair<unsigned int, int>(w0_hash, 0));
    taskQueue.push(w0_hash);

    helpers::Ptr<float> h_w(nFeatures, 1);
    helpers::Ptr<float> h_Wtmp(nRays, nFeatures);
    helpers::Ptr<int> h_errorCounts(nRays, 1);
    helpers::Ptr<unsigned int> h_hashes(nRays, 1);

    for (int iIter = 0; iIter < maxIterations; ++iIter)
    {
        // printf("%i\t%i\t%i\n", iIter, knownAlgs.size(), taskQueue.size());
        if (taskQueue.empty()) return algsFound;

        unsigned int hash = taskQueue.front();
        taskQueue.pop();
        int id = knownAlgs.find(hash)->second;

        for (int iFeatures = 0; iFeatures < nFeatures; ++iFeatures) {
            h_w.get()[iFeatures] = h_W[IDX2C(id, iFeatures, maxAlgs)];
        }

        findNeighbors(sessionId, h_w.get(), h_Wtmp.get(), NULL);
        CHECK(calcAlgs(sessionId, h_Wtmp.get(), nRays, NULL, NULL, h_errorCounts.get(), h_hashes.get()));

        for (int iRay = 0; iRay < nRays; ++iRay) {
            if (helpers::isnan(h_Wtmp.get()[IDX2C(iRay, 0, nRays)])) continue; // not found.
            if ((nErrorsLimit >= 0) && (h_errorCounts.get()[iRay] > nErrorsLimit)) continue; // algorithm with to many errors;
            if (knownAlgs.find(h_hashes.get()[iRay]) != knownAlgs.end()) continue; // already known algorithm.

            taskQueue.push(h_hashes.get()[iRay]);
            knownAlgs.insert(std::pair<unsigned int, int>(h_hashes.get()[iRay], algsFound));
            for (int iFeatures = 0; iFeatures < nFeatures; ++iFeatures) {
                h_W[IDX2C(algsFound, iFeatures, maxAlgs)] = h_Wtmp.get()[IDX2C(iRay, iFeatures, nRays)];
            }
            algsFound++;

            if (algsFound >= maxAlgs) {
                return algsFound; // immediatelly exit.
            }
        }
    }

    LOG_F(logINFO, "findAllNeighbors(): %i algs found.", algsFound);
    return algsFound;
}

// Determines list of source algoriths.
int findSources(
    unsigned char *h_ev,            // nItems * nAlgs    - matrix of error vectors
    int nItems,                     //                   - number of items
    int nAlgs,                      //                   - number of algorithms
    unsigned char *h_isSource)        // nAlgs * 1        - flags indicating whether algs are sources
{
    LOG_F(logINFO, "findSources(), nItems = %i, nAlgs = %i", nItems, nAlgs);
    CHECK(Recorder::getInstance().record_findSources(h_ev, nItems, nAlgs, h_isSource));

    if (h_ev == NULL) {
        LOG_F(logERROR, "findSources(), h_ev == NULL");
        return LINEAR_SAMPLING_ERROR;
    }

    SourcesContainer sources;
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        auto ptr = SourcesContainer::ConstructSourcePtr(&h_ev[nItems * iAlg], &h_ev[nItems * (iAlg + 1)]);
        if (!sources.is_new_source(ptr)) continue;
        sources.remove_old_sources(ptr);
        sources.push_back(iAlg, ptr);
    }

    if (h_isSource != NULL) {
        memset(h_isSource, 0, sizeof(unsigned char) * nAlgs);
        for (auto iter = sources.begin(); iter != sources.end(); ++iter) {
            h_isSource[iter->first] = 1;
        }
    }

    LOG_F(logINFO, "findSources(), %i sources found.", sources.size());
    return 0;
}

// Random walking, starting from the set of hyperplanes w0. Searches for "sources" (algorithms with no incoming edges).
// Stop conditions: either reached given number of iterations, or found required number of algorithms.
// Remark: only returns algorithms with no more than nErrorsLimit errors.
// Output array:
//    - W is the matrix of adjacent vectors. Must be of size maxAlgs * nFeatures.
//  - isSource --- vector of length nAlgs, where 1 indicate source, otherwise 0.
// Returns: actual count of sources found. Negative value indicates failure.
int runRandomWalking(
    int sessionId,
    const float *h_w0,
    int n_w0,
    int maxAlgs,
    int nIters,
    int nErrorsLimit,
    int allowSimilar,
    const float* pTransition,
    int randomSeed,
    int onlineCV_nIters,
    int onlineCV_nCheckpoints,
    const int *onlineCV_checkpoints,
    int onlineCV_nTrainItems,
    float *h_W,                   // maxAlgs * nFeatures
    unsigned char *h_isSource,    // maxAlgs
    int *onlineCV_trainEC,        // onlineCV_nIters * onlineCV_nCheckpoints - for each monte-carlo iteration reports train errors count of the best ERM alg.
    int *onlineCV_testEC          // onlineCV_nIters * onlineCV_nCheckpoints
    )    
{
    LOG_F(logINFO, "runRandomWalking(): sessionId = %i, n_w0 = %i, maxAlgs = %i, nIters = %i, nErrorsLimit = %i, allowSimilar = %i, pTransition = %.3f, randomSeed = %i, onlineCV_nIters = %i, onlineCV_nCheckpoints = %i, onlineCV_nTrainItems = %i, h_W%s, isSource%s, onlineCV_trainEC%s, onlineCV_testEC%s",
        sessionId, n_w0, maxAlgs, nIters, nErrorsLimit, allowSimilar, pTransition[0], randomSeed, onlineCV_nIters, onlineCV_nCheckpoints, onlineCV_nTrainItems, IS_NULL_STR(h_W), IS_NULL_STR(h_isSource), IS_NULL_STR(onlineCV_trainEC), IS_NULL_STR(onlineCV_testEC));
    CHECK(Recorder::getInstance().record_runRandomWalking(sessionId, h_w0, n_w0, maxAlgs, nIters, nErrorsLimit, allowSimilar, pTransition[0], randomSeed, onlineCV_nIters, onlineCV_nCheckpoints, onlineCV_checkpoints, onlineCV_nTrainItems, h_W, h_isSource, onlineCV_trainEC, onlineCV_testEC));
    if ((pTransition[0] < 0) || (pTransition[0] > 1.0)) {
        LOG_F(logERROR, "runRandomWalking(): Can't use pTransition = %.2f, expected number between 0 and 1.", pTransition[0]);
        return LINEAR_SAMPLING_ERROR;
    }

    int nItems, nFeatures, nRays, deviceId;
    CHECK(getSessionStats(sessionId, &nItems, &nFeatures, &nRays, &deviceId));

    if (randomSeed > 0) {
        srand(randomSeed);
    }

    bool performOnlineCV = (onlineCV_nIters > 0) && (onlineCV_nCheckpoints > 0);
    
    helpers::Ptr<float> h_w1(n_w0, nFeatures);
    helpers::Ptr<float> h_w2(n_w0, nFeatures);
    memcpy(h_w1.get(), h_w0, sizeof(float) * n_w0 * nFeatures);

    helpers::Ptr<int> ec1(n_w0, 1);
    helpers::Ptr<unsigned char> ev2(nItems, n_w0);
    helpers::Ptr<int> ec2(n_w0, 1);
    helpers::Ptr<unsigned int> hashes2(n_w0, 1);
    std::map<unsigned int, int> known_algs;
    SourcesContainer sources;

    // Calc error coutns for all starting points.
    CHECK(calcAlgs(sessionId, h_w1.get(), n_w0, NULL, NULL, ec1.get(), NULL));

    int algId = 0;
    for (int iIter = 0; ((iIter < nIters) || (nIters < 0)) && (algId < maxAlgs); ++iIter)
    {
        CHECK(findRandomNeighbors(sessionId, h_w1.get(), n_w0, 0, h_w2.get()));
        CHECK(calcAlgs(sessionId, h_w2.get(), n_w0, NULL, ev2.get(), ec2.get(), hashes2.get()));

        for (int iW = 0; iW < n_w0; ++iW) {
            // Step 1. Skip NaNs (find random neighbors might be unable to move along some
            // random rays; retry should help).
            if (helpers::isnan(h_w2.get()[IDX2C(iW, 0, n_w0)])) {
                continue;
            }

            // Step 2. Reuse previous point because new algorithm has to many errors
            if ((nErrorsLimit >= 0) && (ec2.get()[iW] >= nErrorsLimit)) {
                continue;
            }

            // Step 3. If new error count is greater than previous one we should stay at current point with appropriate probability.
            if ((ec2.get()[iW] > ec1.get()[iW]) && (rand() > (RAND_MAX * pTransition[0]))) {
                continue;
            }

            // Step 4. Copy from h_w2 to h_w1
            ec1.get()[iW] = ec2.get()[iW];
            for (int iF = 0; iF < nFeatures; ++iF) {
                h_w1.get()[IDX2C(iW, iF, n_w0)] = h_w2.get()[IDX2C(iW, iF, n_w0)];
            }

            // Step 5. Test whether new alg is already known (simple hash check)
            // WARNING: this step must be after step 4, otherwise the walk might hang.
            if (known_algs.find(hashes2.get()[iW]) != known_algs.end()) {
                if (!allowSimilar) continue;
            }

            known_algs.insert(std::pair<unsigned int, int>(hashes2.get()[iW], algId));

            if (h_W != NULL) {
                // save algorithm
                for (int iFeature = 0; iFeature < nFeatures; ++iFeature) {
                    h_W[IDX2C(algId, iFeature, maxAlgs)] = h_w1.get()[IDX2C(iW, iFeature, n_w0)];
                }
            }

            if (h_isSource != NULL) {
                // update known sources
                auto ptr = SourcesContainer::ConstructSourcePtr(&ev2.get()[nItems * iW], &ev2.get()[nItems * (iW + 1)]);
                if (sources.is_new_source(ptr)) {
                    sources.remove_old_sources(ptr);
                    sources.push_back(algId, ptr);
                }
            }

            algId++;
            if (algId >= maxAlgs) {
                break;
            }
        }
    }

    if (h_isSource != NULL) {
        memset(h_isSource, 0, sizeof(unsigned char) * maxAlgs);
        for (auto iter = sources.begin(); iter != sources.end(); ++iter) {
            h_isSource[iter->first] = 1;
        }
    }

    LOG_F(logINFO, "runRandomWalking(), %i algorithms, %i sources found.", algId, sources.size());
    return algId;
}

// =========== Debugging functions ===========

int replayRecording(const char* filename) {
    Replayer replayer;
    return replayer.replay(filename);
}

// Sets log level
int setLogLevel(int logLevel) {
    TLogLevel newLevel = FILELog::FromInt(logLevel);
    LOG_F(logINFO, "setLogLevel() sets level to %i", logLevel);
    FILELog::ReportingLevel() = newLevel;
    return 0;
}

// Starts recording of all input data
int startRecording(const char* filename) {
    return Recorder::getInstance().startRecording(filename);
}

// Stops recording of all input data
int stopRecording() {
    return Recorder::getInstance().stopRecording();
}

// ===========================================