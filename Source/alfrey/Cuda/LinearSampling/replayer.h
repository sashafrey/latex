#ifndef __REPLAYER_H
#define __REPLAYER_H

#include <fstream>
#include <sstream>

#include "ls_api.h"

#include "common.h"
#include "helpers.h"
#include "Ptr.h"

class Replayer {
public:
    Replayer() { }

    int replay_calcAlgs()
    {
        int sessionId, nItems, nFeatures, nAlgs;
        read(sessionId);
        read(nItems);
        read(nFeatures);
        read(nAlgs);
        helpers::Ptr<float> h_W(nAlgs, nFeatures);
        read(h_W.get(), nAlgs * nFeatures);

        helpers::Ptr<float> h_scores(nItems, nAlgs);
        helpers::Ptr<unsigned char> h_EV(nItems, nAlgs);
        helpers::Ptr<int> h_EC(nAlgs, 1);
        helpers::Ptr<unsigned int> h_hashes(nAlgs, 1);
        return calcAlgs(sessionId, h_W.get(), nAlgs, h_scores.get(), h_EV.get(), h_EC.get(), h_hashes.get());
    }

    int replay_calcAlgsEV()
    {
        int nItems, nAlgs;
        read(nItems);
        read(nAlgs);
        helpers::Ptr<unsigned char> h_EV(nItems, nAlgs);
        helpers::Ptr<int> h_target(nItems);
        read(h_EV.get(), nItems * nAlgs);
        read(h_target.get(), nItems);

        helpers::Ptr<int> h_EC(nAlgs, 1);
        helpers::Ptr<unsigned int> h_hashes(nAlgs, 1);
        return calcAlgsEV(h_EV.get(), h_target.get(), nItems, nAlgs, h_EC.get(), h_hashes.get());
    }

    int replay_calcAlgsConnectivity()
    {
        int nAlgs, nItems;
        read(nAlgs);
        read(nItems);
        helpers::Ptr<unsigned int> h_algsHashes(nAlgs, 1);
        helpers::Ptr<int> h_algsEC(nAlgs, 1);
        helpers::Ptr<int> upperCon(nAlgs, 1);
        helpers::Ptr<int> lowerCon(nAlgs, 1);
        read(h_algsHashes.get(), nAlgs);
        read(h_algsEC.get(), nAlgs);
        return calcAlgsConnectivity(h_algsHashes.get(), h_algsEC.get(), nAlgs, nItems, upperCon.get(), lowerCon.get());
    }

    int replay_calcQEpsCombinatorial()
    {
        int sessionId, nAlgs, nItems, nTrainItems, nFeatures, nEpsValues, boundType;
        read(sessionId);
        read(nAlgs);
        read(nItems);
        read(nTrainItems);
        read(nFeatures);
        read(nEpsValues);
        read(boundType);

        helpers::Ptr<float> h_W(nAlgs, nFeatures);
        helpers::Ptr<unsigned char> h_isSource(nAlgs, 1);
        helpers::Ptr<float> epsValues(nEpsValues, 1);
        helpers::Ptr<float> h_QEps(nAlgs * nEpsValues);
        read(h_W.get(), nAlgs * nFeatures);
        read(h_isSource.get(), nAlgs);
        read(epsValues.get(), nEpsValues);
        return calcQEpsCombinatorial(sessionId, h_W.get(), h_isSource.get(), epsValues.get(), nTrainItems, nAlgs, nEpsValues, boundType, h_QEps.get());
    }

    int replay_calcQEpsCombinatorialAF()
    {
        int nItems, nTrainItems, nAlgs, nEpsValues, nClusters;
        read(nItems);
        read(nTrainItems);
        read(nAlgs);
        read(nEpsValues);
        read(nClusters);

        helpers::Ptr<unsigned char> h_algsEV(nItems, nAlgs);
        helpers::Ptr<int> h_algsEC(nAlgs);
        helpers::Ptr<unsigned int> h_algsHashes(nAlgs);
        helpers::Ptr<unsigned char> h_isSource(nAlgs);
        helpers::Ptr<float> epsValues(nEpsValues);
        helpers::Ptr<int> h_clusterIds(nAlgs);
        helpers::Ptr<float> h_QEps(nClusters * nEpsValues);

        read(h_algsEV.get(), nItems * nAlgs);
        read(h_algsEC.get(), nAlgs);
        read(h_algsHashes.get(), nAlgs);
        read(h_isSource.get(), nAlgs);
        read(epsValues.get(), nEpsValues);
        read(h_clusterIds.get(), nAlgs);
        return calcQEpsCombinatorialAF(h_algsEV.get(), h_algsEC.get(), h_algsHashes.get(), h_isSource.get(), epsValues.get(), h_clusterIds.get(), nItems, nTrainItems, nAlgs, nEpsValues, nClusters, h_QEps.get());
    }

    int replay_calcQEpsCombinatorialEV()
    {
        int nAlgs, nItems, nTrainItems, nEpsValues, boundType;
        read(nItems);
        read(nTrainItems);
        read(nAlgs);
        read(nEpsValues);
        read(boundType);

        helpers::Ptr<unsigned char> h_algsEV(nItems, nAlgs);
        helpers::Ptr<int> h_algsEC(nAlgs);
        helpers::Ptr<unsigned int> h_algsHashes(nAlgs);
        helpers::Ptr<unsigned char> h_isSource(nAlgs);
        helpers::Ptr<float> epsValues(nEpsValues);
        helpers::Ptr<float> h_QEps(nAlgs * nEpsValues);
        read(h_algsEV.get(), nItems * nAlgs);
        read(h_algsEC.get(), nAlgs);
        read(h_algsHashes.get(), nAlgs);
        read(h_isSource.get(), nAlgs);
        read(epsValues.get(), nEpsValues);
        return calcQEpsCombinatorialEV(h_algsEV.get(), h_algsEC.get(), h_algsHashes.get(), h_isSource.get(), epsValues.get(), nItems, nTrainItems, nAlgs, nEpsValues, boundType, h_QEps.get());
    }

    int replay_closeSession() {
        int sessionId;
        read(sessionId);
        return closeSession(sessionId);
    }

    int replay_closeAllSessions() {
        return closeAllSessions();
    }

    int replay_createSession()
    {
        int sessionId, nItems, nFeatures, nRays, deviceId;
        read(sessionId);
        read(nItems);
        read(nFeatures);
        read(nRays);
        read(deviceId);

        helpers::Ptr<float> h_X(nItems, nFeatures);
        helpers::Ptr<int> h_target(nItems);
        helpers::Ptr<float> h_R(nRays, nFeatures);
        read(h_X.get(), nItems * nFeatures);
        read(h_target.get(), nItems);
        read(h_R.get(), nRays * nFeatures);
        return createSession(h_X.get(), h_target.get(), h_R.get(), nItems, nFeatures, nRays, deviceId, sessionId);
    }

    int replay_findAllNeighbors()
    {
        int sessionId, maxAlgs, maxIterations, nErrorsLimit, nFeatures;
        read(sessionId);
        read(maxAlgs);
        read(maxIterations);
        read(nErrorsLimit);
        read(nFeatures);

        helpers::Ptr<float> h_w0(nFeatures);
        helpers::Ptr<float> h_W(maxAlgs, nFeatures);
        read(h_w0.get(), nFeatures);
        return findAllNeighbors(sessionId, h_w0.get(), maxAlgs, maxIterations, nErrorsLimit, h_W.get());
    }

    int replay_findNeighbors()
    {
        int sessionId, nFeatures, nRays;
        read(sessionId);
        read(nFeatures);
        read(nRays);

        helpers::Ptr<float> h_w0(nFeatures);
        helpers::Ptr<float> h_W(nRays, nFeatures);
        helpers::Ptr<float> h_t(nRays);
        read(h_w0.get(), nFeatures);
        return findNeighbors(sessionId, h_w0.get(), h_W.get(), h_t.get());
    }

    int replay_findRandomNeighbors()
    {
        int sessionId, n_w0, randomSeed, nFeatures;
        read(sessionId);
        read(n_w0);
        read(randomSeed);
        read(nFeatures);

        helpers::Ptr<float> h_w0(n_w0, nFeatures);
        helpers::Ptr<float> h_W(n_w0, nFeatures);

        read(h_w0.get(), n_w0 * nFeatures);
        return findRandomNeighbors(sessionId, h_w0.get(), n_w0, randomSeed, h_W.get());
    }

    int replay_findSources()
    {
        int nItems, nAlgs;
        read(nItems);
        read(nAlgs);

        helpers::Ptr<unsigned char> h_ev(nItems * nAlgs);
        helpers::Ptr<unsigned char> h_isSource(nAlgs);
        read(h_ev.get(), nItems * nAlgs);
        return findSources(h_ev.get(), nItems, nAlgs, h_isSource.get());
    }

    int replay_performCrossValidation()
    {
        int nItems, nTrainItems, nAlgs, nIters, randomSeed;
        read(nItems);
        read(nTrainItems);
        read(nAlgs);
        read(nIters);
        read(randomSeed);

        helpers::Ptr<unsigned char> h_EV(nItems * nAlgs);
        read(h_EV.get(), nItems * nAlgs);

        helpers::Ptr<int> h_EC(nAlgs, 1);
        read(h_EC.get(), nAlgs);

        helpers::Ptr<int> h_trainEC(nIters, 1);
        helpers::Ptr<int> h_testEC(nIters, 1);
        return performCrossValidation(h_EV.get(), h_EC.get(), nItems, nTrainItems, nAlgs, nIters, randomSeed, h_trainEC.get(), h_testEC.get());
    }

    int replay_runRandomWalking()
    {
        int sessionId, n_w0, maxAlgs, nIters, nErrorsLimit, allowSimilar, randomSeed, nFeatures;
        int onlineCV_nIters, onlineCV_nCheckpoints, onlineCV_nTrainItems;
        bool W_notNull, isSource_notNull;
        float pTransition[1];
        read(sessionId);
        read(n_w0);
        read(maxAlgs);
        read(nIters);
        read(nErrorsLimit);
        read(allowSimilar);
        read(pTransition[0]);
        read(randomSeed);
        read(W_notNull);
        read(isSource_notNull);

        read(onlineCV_nIters);
        read(onlineCV_nCheckpoints);
        helpers::Ptr<int> onlineCV_checkpoints;
        if (onlineCV_nCheckpoints != 0) {
            onlineCV_checkpoints.resize(onlineCV_nCheckpoints);
            read(onlineCV_checkpoints.get(), onlineCV_nCheckpoints);
        }

        read(onlineCV_nTrainItems);
        read(nFeatures);

        helpers::Ptr<float> h_w0(n_w0, nFeatures);
        helpers::Ptr<float> h_W(maxAlgs, nFeatures);
        helpers::Ptr<unsigned char> h_isSource(maxAlgs);
        helpers::Ptr<int> onlineCV_trainEC, onlineCV_testEC;
        if ((onlineCV_nIters != 0) && (onlineCV_nCheckpoints) != 0) {
            onlineCV_trainEC.resize(onlineCV_nIters * onlineCV_nCheckpoints);
            onlineCV_testEC.resize(onlineCV_nIters * onlineCV_nCheckpoints);
        }

        read(h_w0.get(), n_w0 * nFeatures);
        return runRandomWalking(sessionId, h_w0.get(), n_w0, maxAlgs, nIters, nErrorsLimit, allowSimilar, pTransition, randomSeed, 
            onlineCV_nIters, onlineCV_nCheckpoints, onlineCV_checkpoints.get(), onlineCV_nTrainItems, h_W.get(), h_isSource.get(), 
            onlineCV_trainEC.get(), onlineCV_testEC.get());
    }

    int replay(std::string filename) {
        stream.open(filename, std::ios::in | std::ios::binary);
        if (stream.fail()) return LINEAR_SAMPLING_ERROR;
        helpers::call_on_destruction c([&]() {
            stream.close();
        } );

        while(!stream.eof()) {
            OperationType op = (OperationType)read<int>();
            if (stream.eof()) break;

            switch(op) {
                case CalcAlgs:
                    if (replay_calcAlgs() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case CalcAlgsEV:
                    if (replay_calcAlgsEV() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case CalcAlgsConnectivity:
                    if (replay_calcAlgsConnectivity() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case CalcQEpsCombinatorial:
                    if (replay_calcQEpsCombinatorial() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case CalcQEpsCombinatorialAF:
                    if (replay_calcQEpsCombinatorialAF() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case CalcQEpsCombinatorialEV:
                    if (replay_calcQEpsCombinatorialEV() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case CloseSession:
                    if (replay_closeSession() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case CloseAllSessions:
                    if (replay_closeAllSessions() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case CreateSession:
                    if (replay_createSession() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case FindAllNeighbors:
                    if (replay_findAllNeighbors() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case FindNeighbors:
                    if (replay_findNeighbors() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case FindRandomNeighbors:
                    if (replay_findRandomNeighbors() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case FindSources:
                    if (replay_findSources() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case PerformCrossValidation:
                    if (replay_performCrossValidation() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                case RunRandomWalking:
                    if (replay_runRandomWalking() < 0) return LINEAR_SAMPLING_ERROR;
                    break;

                default:
                    return LINEAR_SAMPLING_ERROR;
            }
        }
        return 0;
    }
private:
    std::ifstream stream;
    Replayer(const Replayer& r)  { ; }

    template<class T>
    T& read(T& value) {
        stream.read(reinterpret_cast<char *>(&value), sizeof(T));
        return value;
    }

    template<class T>
    T read() {
        T value;
        stream.read(reinterpret_cast<char *>(&value), sizeof(T));
        return value;
    }

    template<class T>
    T* read(T* value, int count) {
        stream.read(reinterpret_cast<char *>(value), sizeof(T) * count);
        return value;
    }
};

#endif // _REPLAYER_H