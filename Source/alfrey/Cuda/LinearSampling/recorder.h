#ifndef __RECORDER_H
#define __RECORDER_H

#include <fstream>
#include <sstream>

#include "ls_api.h"

#include "common.h"
#include "helpers.h"
#include "Log.h"

class Recorder {
public:
    int record_calcAlgs(
        int sessionId,
        const float *h_W,
        int nAlgs,
        float *h_scores,
        unsigned char* h_EV,
        int* h_EC,
        unsigned int* h_hashes)
    {
        if (!recording) return 0;
        int nItems, nFeatures, nRays, deviceId;
        CHECK(getSessionStats(sessionId, &nItems, &nFeatures, &nRays, &deviceId));

        write((int)CalcAlgs);
        write(sessionId);
        write(nItems);
        write(nFeatures);
        write(nAlgs);
        write(h_W, nAlgs * nFeatures);
        return 0;
    }

    int record_calcAlgsEV(
        const unsigned char* h_EV,
        const int* h_target,
        int nItems,
        int nAlgs,
        int* h_EC,
        unsigned int* h_hashes)
    {
        if (!recording) return 0;
        write((int)CalcAlgsEV);
        write(nItems);
        write(nAlgs);
        write(h_EV, nItems * nAlgs);
        write(h_target, nItems);
        return 0;
    }

    int record_calcAlgsConnectivity(
        const unsigned int* h_algsHashes,
        const int* h_algsEC,
        int nAlgs,
        int nItems,
        int* upperCon,
        int* lowerCon)
    {
        if (!recording) return 0;
        write((int)CalcAlgsConnectivity);
        write(nAlgs);
        write(nItems);
        write(h_algsHashes, nAlgs);
        write(h_algsEC, nAlgs);
        return 0;
    }

    int record_calcQEpsCombinatorial(
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
        if (!recording) return 0;
        int nItems, nFeatures, nRays, deviceId;
        CHECK(getSessionStats(sessionId, &nItems, &nFeatures, &nRays, &deviceId));
        write((int)CalcQEpsCombinatorial);
        write(sessionId);
        write(nAlgs);
        write(nItems);
        write(nTrainItems);
        write(nFeatures);
        write(nEpsValues);
        write(boundType);
        write(h_W, nAlgs * nFeatures);
        write(h_isSource, nAlgs);
        write(epsValues, nEpsValues);
        return 0;
    }

    int record_calcQEpsCombinatorialAF(
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
        int nClusters)
    {
        if (!recording) return 0;
        write((int)CalcQEpsCombinatorialAF);
        write(nItems);
        write(nTrainItems);
        write(nAlgs);
        write(nEpsValues);
        write(nClusters);
        write(h_algsEV, nItems * nAlgs);
        write(h_algsEC, nAlgs);
        write(h_algsHashes, nAlgs);
        write(h_isSource, nAlgs);
        write(epsValues, nEpsValues);
        write(h_clusterIds, nAlgs);
        return SUCCESS;
    }

    int record_calcQEpsCombinatorialEV(
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
        if (!recording) return 0;
        write((int)CalcQEpsCombinatorialEV);
        write(nItems);
        write(nTrainItems);
        write(nAlgs);
        write(nEpsValues);
        write(boundType);
        write(h_algsEV, nItems * nAlgs);
        write(h_algsEC, nAlgs);
        write(h_algsHashes, nAlgs);
        write(h_isSource, nAlgs);
        write(epsValues, nEpsValues);
        return 0;
    }

    int record_closeSession(int sessionId) {
        if (!recording) return 0;
        write((int)CloseSession);
        write(sessionId);
        return 0;
    }

    int record_closeAllSessions() {
        if (!recording) return 0;
        write((int)CloseAllSessions);
        return 0;
    }

    int record_createSession(
        int sessionId,
        const float *h_X,
        const int* h_target,
        const float *h_R,
        int nItems,
        int nFeatures,
        int nRays,
        int deviceId)
    {
        if (!recording) return 0;
        write((int)CreateSession);
        write(sessionId);
        write(nItems);
        write(nFeatures);
        write(nRays);
        write(deviceId);
        write(h_X, nItems * nFeatures);
        write(h_target, nItems);
        write(h_R, nRays * nFeatures);
        return 0;
    }

    int record_findAllNeighbors(
        int sessionId,
        const float *h_w0,
        int maxAlgs,
        int maxIterations,
        int nErrorsLimit,
        float *h_W)
    {
        if (!recording) return 0;
        int nItems, nFeatures, nRays, deviceId;
        CHECK(getSessionStats(sessionId, &nItems, &nFeatures, &nRays, &deviceId));

        write((int)FindAllNeighbors);
        write(sessionId);
        write(maxAlgs);
        write(maxIterations);
        write(nErrorsLimit);
        write(nFeatures);
        write(h_w0, nFeatures);
        return 0;
    }

    int record_findNeighbors(
        int sessionId,
        const float *h_w0,
        float *h_W,
        float *h_t)
    {
        if (!recording) return 0;
        int nItems, nFeatures, nRays, deviceId;
        CHECK(getSessionStats(sessionId, &nItems, &nFeatures, &nRays, &deviceId));

        write((int)FindNeighbors);
        write(sessionId);
        write(nFeatures);
        write(nRays);
        write(h_w0, nFeatures);
        return 0;
    }

    int record_findRandomNeighbors(
        int sessionId,
        const float *h_w0,
        int n_w0,
        int randomSeed,
        float *h_W)
    {
        if (!recording) return 0;
        int nItems, nFeatures, nRays, deviceId;
        CHECK(getSessionStats(sessionId, &nItems, &nFeatures, &nRays, &deviceId));

        write((int)FindRandomNeighbors);
        write(sessionId);
        write(n_w0);
        write(randomSeed);
        write(nFeatures);
        write(h_w0, n_w0 * nFeatures);
        return 0;
    }

    int record_findSources(
        unsigned char *h_ev,
        int nItems,
        int nAlgs,
        unsigned char *h_isSource)
    {
        if (!recording) return 0;

        write((int)FindSources);
        write(nItems);
        write(nAlgs);
        write(h_ev, nItems * nAlgs);
        return 0;
    }

    // int record_getSessionStats(int sessionId, int *nItems, int *nFeatures, int *nRays, int *deviceId);

    int record_performCrossValidation(
        const unsigned char* h_EV,
        const int*  h_EC,
        int nItems,
        int nTrainItems,
        int nAlgs,
        int nIters,
        int randomSeed,
        int* h_trainEC,
        int* h_testEC)
    {
        if (!recording) return 0;
        write((int)PerformCrossValidation);
        write(nItems);
        write(nTrainItems);
        write(nAlgs);
        write(nIters);
        write(randomSeed);
        write(h_EV, nItems * nAlgs);
        write(h_EC, nAlgs);
        return 0;
    }

    int record_runRandomWalking(
        int sessionId,
        const float *h_w0,
        int n_w0,
        int maxAlgs,
        int nIters,
        int nErrorsLimit,
        int allowSimilar,
        float pTransition,
        int randomSeed,
        int onlineCV_nIters,
        int onlineCV_nCheckpoints,
        const int *onlineCV_checkpoints,
        int onlineCV_nTrainItems,
        float *h_W,
        unsigned char *h_isSource,
        int *onlineCV_trainEC,
        int *onlineCV_testEC)
    {
        if (!recording) return 0;
        int nItems, nFeatures, nRays, deviceId;
        CHECK(getSessionStats(sessionId, &nItems, &nFeatures, &nRays, &deviceId));

        write((int)RunRandomWalking);
        write(sessionId);
        write(n_w0);
        write(maxAlgs);
        write(nIters);
        write(nErrorsLimit);
        write(allowSimilar);
        write(pTransition);
        write(randomSeed);
        write(h_W != NULL);
        write(h_isSource != NULL);
        write(onlineCV_nIters);
        write(onlineCV_nCheckpoints);
        if (onlineCV_nCheckpoints != 0) write(onlineCV_checkpoints, onlineCV_nCheckpoints);
        write(onlineCV_nTrainItems);        
        write(nFeatures);
        write(h_w0, n_w0 * nFeatures);
        return 0;
    }

    int startRecording(const char* filename) {
        if (recording) {
            LOG_F(logERROR, "startRecording(): recording is already running.");
            return LINEAR_SAMPLING_ERROR;
        }

        LOG_F(logINFO, "startRecording(), target binary file: %s", filename);
        stream.open(filename, std::ios::out | std::ios::binary);
        if (stream.fail()) return LINEAR_SAMPLING_ERROR;
        recording = true;
        return 0;
    }

    int stopRecording() {
        if (!recording) {
            LOG_F(logWARNING, "stopRecording(): recording is not running.");
            return 0;
        }

        LOG_F(logINFO, "stopRecording()");
        stream.close();
        recording = false;
        return 0;
    }

    void flush() {
        stream.flush();
    }

    static Recorder& getInstance() {
        static Recorder recorder;
        return recorder;
    }

    ~Recorder() {
        if (recording) {
            stopRecording();
        }
    }
private:
    std::ofstream stream;
    bool recording;

    template<class T>
    void write(const T* values, int count) {
        stream.write(reinterpret_cast<const char *>(values), sizeof(T) * count);
    }

    template<class T>
    void write(const T& value) {
        stream.write(reinterpret_cast<const char *>(&value), sizeof(T));
    }

    Recorder() : recording(false) { }
};

#endif //__RECORDER_H