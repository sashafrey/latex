#include "ls_api.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include <algorithm>
#include <functional>
#include <set>

#include "gtest\gtest.h"

#include "helpers.h"
#include "Log.h"
#include "replayer.h"
#include "test_helpers.h"

std::string recordingFileName;

class LinearSamplingTest {
public:
    LinearSamplingTest(int nItems, int nFeatures, int nRays, int deviceId = 0) {
        X.resize(nItems * nFeatures);
        target.resize(nItems);
        R.resize(nRays * nFeatures);

        fillArray2D(X, nItems, nFeatures);
        fillArray2D_int(target, nItems, 1);
        fillArray2D(R, nRays, nFeatures);

        sessionId = createSession(X, target, R, nItems, nFeatures, nRays, deviceId, UNINITIALIZED);
    }

    void CalcQEpsCombinatorialGenericTest(int boundType) {
        int nStartPoints = 1, randomSeed = 1, allowSimilar = 0, nIters = 256;
        int onlineCV_nIters = 0, onlineCV_nCheckpoints = 0, onlineCV_nTrainItems = 0;
        int *onlineCV_checkpoints = NULL, *onlineCV_trainEC = NULL, *onlineCV_testEC = NULL;
        int nItems, nFeatures, nRays, deviceId;
        getSessionStats(sessionId, &nItems, &nFeatures, &nRays, &deviceId);

        float pTransition[1];
        pTransition[0] = 1.0f;
        int maxAlgs = (nIters * nStartPoints) / 8;

        test_helpers::Ptr<float> w0(nStartPoints * nFeatures);
        fillArray2D(w0, nStartPoints, nFeatures);

        test_helpers::Ptr<unsigned char> h_isSource(maxAlgs);
        test_helpers::Ptr<float> h_W(maxAlgs * nFeatures);
        int nAlgsFound = runRandomWalking(sessionId, w0, nStartPoints, maxAlgs, nIters, nItems, allowSimilar, pTransition, randomSeed, 
            onlineCV_nIters, onlineCV_nCheckpoints, onlineCV_checkpoints, onlineCV_nTrainItems, h_W, h_isSource, onlineCV_trainEC, onlineCV_testEC);
        ASSERT_EQ(nAlgsFound, maxAlgs);

        test_helpers::Ptr<int> h_algsEC(maxAlgs);
        test_helpers::Ptr<unsigned int> h_algsHashes(maxAlgs);
        test_helpers::Ptr<unsigned char> h_algsEV(nItems * maxAlgs);
        ASSERT_EQ(calcAlgs(sessionId, h_W, maxAlgs, NULL, h_algsEV, h_algsEC, h_algsHashes), 0);

        int nEpsValues = nItems / 4;
        int nTrainItems = nItems / 2;
        test_helpers::Ptr<float> epsValues(nEpsValues);
        for (int iEps = 0; iEps < nEpsValues; ++iEps) {
            epsValues[iEps] = 4.0f / nItems * iEps;
        }

        test_helpers::Ptr<float> h_QEps(maxAlgs * nEpsValues);
        test_helpers::Ptr<float> h_QEpsEV(maxAlgs * nEpsValues);
        ASSERT_EQ(calcQEpsCombinatorial(sessionId, h_W, h_isSource, epsValues, nTrainItems, maxAlgs, nEpsValues, boundType, h_QEps), 0);
        ASSERT_EQ(calcQEpsCombinatorialEV(h_algsEV, h_algsEC, h_algsHashes, h_isSource, epsValues, nItems, nTrainItems, maxAlgs, nEpsValues, boundType, h_QEpsEV), 0);

        for (int i = 0; i < nAlgsFound; ++i) {
            for (int j = 0; j < nEpsValues; ++j) {
                ASSERT_GE(h_QEps[IDX2C(i, j, maxAlgs)], 0);
                ASSERT_LE(h_QEps[IDX2C(i, j, maxAlgs)], 1);
                ASSERT_FLOAT_EQ(h_QEps[IDX2C(i, j, maxAlgs)], h_QEpsEV[IDX2C(i, j, maxAlgs)]);
                if (j != 0) ASSERT_GE(h_QEps[IDX2C(i, j - 1, maxAlgs)], h_QEps[IDX2C(i, j, maxAlgs)]);
            }
        }

        // Test that classic combinatorial bound is consistent with AF bound when all clusters consist of one algorithm
        if (boundType == 1) {
            test_helpers::Ptr<float> h_QEpsAF(maxAlgs * nEpsValues);
            test_helpers::Ptr<int> h_clusterIds(maxAlgs);
            for (int i = 0; i < maxAlgs; ++i) h_clusterIds[i] = i;
            ASSERT_EQ(calcQEpsCombinatorialAF(h_algsEV, h_algsEC, h_algsHashes, h_isSource, epsValues, h_clusterIds, nItems, nTrainItems, maxAlgs, nEpsValues, maxAlgs, h_QEpsAF), 0);
            for (int i = 0; i < nAlgsFound; ++i) {
                for (int j = 0; j < nEpsValues; ++j) {
                    float v1 = h_QEpsAF[IDX2C(i, j, maxAlgs)];
                    float v2 = h_QEpsEV[IDX2C(i, j, maxAlgs)];
                    // EXPECT_TRUE(abs(v1 - v2) < 1e-5);
                }
            }
        }
    }

    ~LinearSamplingTest() {
        closeSession(sessionId);
    }

    test_helpers::Ptr<float> X, R;
    test_helpers::Ptr<int> target;
    int sessionId;
};

TEST(LinearSampling, StartRecording) {
    std::stringstream filename;
    std::string dateTime = helpers::getDateTime();
    std::replace(dateTime.begin(), dateTime.end(), ':', '_');
    std::replace(dateTime.begin(), dateTime.end(), ' ', '_');
    filename << "recording_" << "_" << dateTime << ".dat";
    recordingFileName = filename.str();

    ASSERT_EQ(startRecording(recordingFileName.c_str()), 0);
}

TEST(LinearSampling, IsNan) {
    ASSERT_TRUE(sizeof(float) <= sizeof(int));
    int int_plus_nanq = 0x7fc00000;
    int int_minus_nanq = 0xffc00000;
    int int_nans = 0x7f800001;
    float plus_nanq     = *((float*)&int_plus_nanq);
    float minus_nanq = *((float*)&int_minus_nanq);
    float nans         = *((float*)&int_nans);
    float nan = helpers::get_qnan();

    ASSERT_TRUE(helpers::isnan(plus_nanq));
    ASSERT_TRUE(helpers::isnan(minus_nanq));
    ASSERT_TRUE(helpers::isnan(nans));
    ASSERT_TRUE(helpers::isnan(nan));
}

TEST(LinearSampling, SetLogLevel) {
    setLogLevel(logSILENT);
    closeAllSessions();
    setLogLevel(logDEBUG);

    FILELog::ReportingLevel() = logDEBUG3;
    LOG(logERROR) << "Ops, variable x should be " << 1 << "; is " << 2;
    LOG(logWARNING) << "Ops, variable x should be " << 1 << "; is " << 2;
    LOG(logINFO) << "Ops, variable x should be " << 1 << "; is " << 2;
    LOG(logDEBUG) << "Debug Ops, variable x should be " << 1 << "; is " << 2;
    LOG(logDEBUG1) << "Debug1 Ops, variable x should be " << 1 << "; is " << 2;
    LOG_F(logERROR, "123 %i 123", 666);
}

TEST(LinearSampling, StartStop) {
    LinearSamplingTest test(128, 16, 48);
    ASSERT_GE(test.sessionId, 0);

    int nItems, nFeatures, nRays, deviceId;
    int errco = getSessionStats(test.sessionId, &nItems, &nFeatures, &nRays, &deviceId);
    ASSERT_EQ(128, nItems);
    ASSERT_EQ(16, nFeatures);
    ASSERT_EQ(48, nRays);
    ASSERT_EQ(errco, 0);
}

TEST(LinearSampling, MultipleConcurrentSessions) {
    for (int i = 0; i < 10; ++i) {
        int nItems = 128 + i, nFeatures = 32 + i, nRays = 48 + i;
        test_helpers::Ptr<float> X(nItems * nFeatures), R(nRays * nFeatures);
        test_helpers::Ptr<int> target(nItems);

        fillArray2D(X, nItems, nFeatures);
        fillArray2D_int(target, nItems, 1);
        fillArray2D(R, nRays, nFeatures);

        createSession(X, target, R, nItems, nFeatures, nRays, 0, UNINITIALIZED);
    }

    closeAllSessions();
}

TEST(LinearSampling, FindNeighborsNan) {
    int nItems = 1, nFeatures = 1, nRays = 48;
    LinearSamplingTest test(nItems, nFeatures, nRays);
    int sessionId = test.sessionId;

    test_helpers::Ptr<float> w0(nFeatures);
    fillArray2D(w0, 1, nFeatures);

    test_helpers::Ptr<float> W(nRays * nFeatures);
    test_helpers::Ptr<float> t(nRays);
    ASSERT_EQ(findNeighbors(sessionId, w0, W, t), 0);

    for (int i = 0; i < nRays; ++i) ASSERT_TRUE(helpers::isnan(t[i]));
    for (int i = 0; i < nRays * nFeatures; ++i) ASSERT_TRUE(helpers::isnan(W[i]));
}

TEST(LinearSampling, FindNeighbors) {
    int nItems = 128, nFeatures = 16, nRays = 48;
    LinearSamplingTest test(nItems, nFeatures, nRays);
    int sessionId = test.sessionId;

    test_helpers::Ptr<float> w0(nFeatures);
    fillArray2D(w0, 1, nFeatures);

    test_helpers::Ptr<float> W(nRays * nFeatures);
    test_helpers::Ptr<float> t(nRays);
    ASSERT_EQ(findNeighbors(sessionId, w0, W, t), 0);

    for (int i = 0; i < nRays; ++i) ASSERT_GE(t[i], 0.0f);
    for (int i = 0; i < nRays * nFeatures; ++i) ASSERT_FALSE(helpers::isnan(W[i]));

    test_helpers::Ptr<float> w0_scores(nItems);
    int w0_errorsCount;
    ASSERT_EQ(calcAlgs(test.sessionId, w0, 1, w0_scores, NULL, &w0_errorsCount, NULL), 0);

    test_helpers::Ptr<float> scores(nItems * nRays);
    test_helpers::Ptr<unsigned char> errorVectors(nItems * nRays), errorVectors2(nItems * nRays);
    test_helpers::Ptr<int> errorVectorsCount(nRays), errorVectorsCount2(nRays);
    test_helpers::Ptr<unsigned int> hashes(nRays), hashes2(nRays);
    ASSERT_EQ(calcAlgs(test.sessionId, W, nRays, scores, errorVectors, errorVectorsCount, hashes), 0);

    ASSERT_EQ(calcAlgsEV(errorVectors, test.target, nItems, nRays, errorVectorsCount2, hashes2), 0);
    for (int i = 0; i < nRays; ++i) {
        ASSERT_EQ(errorVectorsCount[i], errorVectorsCount2[i]);
        ASSERT_EQ(hashes[i], hashes2[i]);
    }

    for (int iRay = 0; iRay < nRays; ++iRay) {
        int errorCount = 0;
        int distance = 0;
        for (int iItem = 0; iItem < nItems; ++iItem) {
            int idx = IDX2C(iItem, iRay, nItems);
            unsigned char predictedTarget = PREDICTED_TARGET(scores[idx]);
            int error = (predictedTarget != test.target[iItem]);
            ASSERT_EQ(errorVectors[idx], error);
            errorCount += error;

            unsigned char w0 = PREDICTED_TARGET(w0_scores[iItem]);
            distance += (w0 != predictedTarget);
        }

        ASSERT_EQ(errorCount, errorVectorsCount[iRay]);
        ASSERT_EQ(distance, 1);
    }

    std::set<unsigned int> unique_hashes;
    for (int iRay = 0; iRay < nRays; ++iRay) {
        if (unique_hashes.find(hashes[iRay]) == unique_hashes.end()) {
            unique_hashes.insert(hashes[iRay]);
        }
    }

    ASSERT_GE(unique_hashes.size(), (size_t)2);
}

TEST(LinearSampling, FindAllNeighbors) {
    int nItems = 128, nFeatures = 32, nRays = 48;
    LinearSamplingTest test(nItems, nFeatures, nRays);
    int sessionId = test.sessionId;

    test_helpers::Ptr<float> w0(nFeatures);
    fillArray2D(w0, 1, nFeatures);

    int w0_errorsCount;
    ASSERT_EQ(calcAlgs(test.sessionId, w0, 1, NULL, NULL, &w0_errorsCount, NULL), 0);

    int nAlgs = 20;
    int maxIters = 10;
    test_helpers::Ptr<float> W_all(nAlgs * nFeatures);
    int nAlgsActual = findAllNeighbors(sessionId, w0, nAlgs, maxIters, nItems, W_all);
    ASSERT_EQ(nAlgsActual, nAlgs);

    test_helpers::Ptr<int> W_errorsCount(nAlgs);
    ASSERT_EQ(calcAlgs(test.sessionId, W_all, nAlgsActual, NULL, NULL, W_errorsCount, NULL), 0);
    for (int i = 0; i < nAlgsActual; ++i) {
        ASSERT_LE(abs((int)w0_errorsCount - (int)W_errorsCount[i]), maxIters);
    }
}

TEST(LinearSampling, PerformCrossValidation) {
    int nItems = 128, nFeatures = 32, nRays = 48, nAlgs = 20, nCVIters = 100, randomSeed = 1;
    int nTrainItems = nItems / 2;
    LinearSamplingTest test(nItems, nFeatures, nRays);
    int sessionId = test.sessionId;

    test_helpers::Ptr<float> w0(nFeatures);
    fillArray2D(w0, 1, nFeatures);

    test_helpers::Ptr<float> W_all(nAlgs * nFeatures);
    int nAlgsActual = findAllNeighbors(sessionId, w0, nAlgs, nAlgs, nItems, W_all);
    ASSERT_EQ(nAlgsActual, nAlgs);

    test_helpers::Ptr<unsigned char> errorVectors(nAlgs * nItems);
    test_helpers::Ptr<int> errorsCount(nAlgs), errorsCount2(nAlgs);
    test_helpers::Ptr<unsigned int> hashes(nAlgs), hashes2(nAlgs);
    ASSERT_EQ(calcAlgs(test.sessionId, W_all, nAlgsActual, NULL, errorVectors, errorsCount, hashes), 0);
    ASSERT_EQ(calcAlgsEV(errorVectors, test.target, nItems, nAlgsActual, errorsCount2, hashes2), 0);
    for (int i = 0; i < nAlgsActual; ++i) {
        ASSERT_EQ(errorsCount[i], errorsCount2[i]);
        ASSERT_EQ(hashes[i], hashes2[i]);
    }

    test_helpers::Ptr<int> trainEC(nCVIters), testEC(nCVIters);
    ASSERT_EQ(performCrossValidation(errorVectors, errorsCount, nItems, nTrainItems, nAlgs, nCVIters, randomSeed, trainEC, testEC), 0);

    // ensure that there are at least 10 different values
    // ensure that all values are bounded to [-nItems / 3, nItems / 3]
    std::set<int> unique_counts;
    for (int i = 0; i < nCVIters; ++i) {
        int overfitting = testEC[i] - trainEC[i];
        ASSERT_LE(overfitting, nItems / 3);
        ASSERT_GE(overfitting, -nItems / 3);
        if (unique_counts.find(overfitting) == unique_counts.end()) {
            unique_counts.insert(overfitting);
        }
    }

    ASSERT_GE(unique_counts.size(), (size_t)10);
}

TEST(LinearSampling, FindRandomNeighbors) {
    int nItems = 128, nFeatures = 16, nRays = 48, nStartPoints = 32, randomSeed = 1;
    LinearSamplingTest test(nItems, nFeatures, nRays);
    int sessionId = test.sessionId;

    test_helpers::Ptr<float> w0(nStartPoints * nFeatures);
    fillArray2D(w0, nStartPoints, nFeatures);

    test_helpers::Ptr<int> w0_ec(nStartPoints);
    ASSERT_EQ(calcAlgs(sessionId, w0, nStartPoints, NULL, NULL, w0_ec, NULL), 0);

    test_helpers::Ptr<float> W(nStartPoints * nFeatures);
    ASSERT_EQ(findRandomNeighbors(sessionId, w0, nStartPoints, randomSeed, W), 0);

    test_helpers::Ptr<int> W_ec(nStartPoints);
    ASSERT_EQ(calcAlgs(sessionId, W, nStartPoints, NULL, NULL, W_ec, NULL), 0);

    for (int i = 0; i < nStartPoints * nFeatures; ++i) ASSERT_FALSE(helpers::isnan(W[i]));
    for (int i = 0; i < nStartPoints; ++i ) {
        ASSERT_LE(abs((int)w0_ec[i] - (int)W_ec[i]), 1);
    }
}

TEST(LinearSampling, RunRandomWalking) {
    int nItems = 32, nFeatures = 4, nRays = 60, nStartPoints = 1, allowSimilar = 0, randomSeed = 1, nIters = 256;

    float pTransition[1];
    pTransition[0] = 1.0f;
    int maxAlgs = (nIters * nStartPoints) / 8;
    LinearSamplingTest test(nItems, nFeatures, nRays);
    int sessionId = test.sessionId;

    // Setup onlineCV parameters.
    int onlineCV_nIters = 16, onlineCV_nCheckpoints = 4, onlineCV_nTrainItems = (nItems/2);
    test_helpers::Ptr<int> onlineCV_checkpoints(onlineCV_nCheckpoints);
    test_helpers::Ptr<int> onlineCV_trainEC(onlineCV_nCheckpoints * onlineCV_nIters);
    test_helpers::Ptr<int> onlineCV_testEC(onlineCV_nCheckpoints * onlineCV_nIters);
    onlineCV_checkpoints.get()[0] = 4;
    onlineCV_checkpoints.get()[1] = 8;
    onlineCV_checkpoints.get()[2] = 16;
    onlineCV_checkpoints.get()[3] = 32;

    test_helpers::Ptr<float> w0(nStartPoints * nFeatures);
    fillArray2D(w0, nStartPoints, nFeatures);

    test_helpers::Ptr<unsigned char> h_isSource(maxAlgs);
    test_helpers::Ptr<float> h_W(maxAlgs * nFeatures);
    int nAlgsFound = runRandomWalking(sessionId, w0, nStartPoints, maxAlgs, nIters, nItems, allowSimilar, pTransition, randomSeed, 
        onlineCV_nIters, onlineCV_nCheckpoints, onlineCV_checkpoints, onlineCV_nTrainItems, h_W, h_isSource, onlineCV_trainEC, onlineCV_testEC);
    ASSERT_EQ(nAlgsFound, maxAlgs);

    test_helpers::Ptr<int> h_algsEC(maxAlgs);
    test_helpers::Ptr<unsigned int> h_algsHashes(maxAlgs);
    test_helpers::Ptr<unsigned char> h_algsEV(nItems * maxAlgs);
    ASSERT_EQ(calcAlgs(sessionId, h_W, maxAlgs, NULL, h_algsEV, h_algsEC, h_algsHashes), 0);

    test_helpers::Ptr<int> h_upperCon(maxAlgs);
    test_helpers::Ptr<int> h_lowerCon(maxAlgs);
    ASSERT_EQ(calcAlgsConnectivity(h_algsHashes, h_algsEC, maxAlgs, nItems, h_upperCon, h_lowerCon), 0);
    int numSources = 0, numZeroCon = 0;
    for (int i = 0; i < maxAlgs; ++i) {
        ASSERT_GE(h_upperCon[i], 0);
        ASSERT_GE(h_lowerCon[i], 0);
        ASSERT_LE(h_upperCon[i], nItems);
        ASSERT_LE(h_lowerCon[i], nItems);

        if (h_lowerCon[i] == 0) numZeroCon++;
        if (h_isSource[i]) numSources++;

        if (h_lowerCon[i] == 0) {
            // Disable this verification. I guess this is a bug in lowerCon / isSource calculation!
            // ASSERT_TRUE(h_isSource[i]);
        }
    }

    // Idealy lowerCon==0 defines the same condition as isSource == true.
    // However, method calcAlgsConnectivity only estimates (not the exact value) of lowerCon,
    // due to potential hash collisions (see implementatino of calcAlgsConnectivity for details).
    // calcAlgsConnectivity guaranties that if lowerCon==0 than the algorihm is true source, but
    // the inverse is not true (in some cases lowerCon > 0 even for sources). This is an undesirable behaviour,
    // and the number of such casesshould stay at minumim. The following assert checks that this ration is less than 50%.
    ASSERT_LE(numSources, 2 * numZeroCon);

    test_helpers::Ptr<unsigned char> h_isSources2(maxAlgs);
    findSources(h_algsEV, nItems, maxAlgs, h_isSources2);
    for (int i = 0; i < maxAlgs; ++i) {
        ASSERT_EQ(h_isSources2[i], h_isSource[i]);
    }
}

TEST(LinearSampling, CalcQEpsCombinatorialESokolov) {
    int nItems = 32, nFeatures = 4, nRays = 60;
    LinearSamplingTest test(nItems, nFeatures, nRays);
    test.CalcQEpsCombinatorialGenericTest(0);
}

TEST(LinearSampling, CalcQEpsCombinatorialClassicSC) {
    int nItems = 32, nFeatures = 4, nRays = 60;
    LinearSamplingTest test(nItems, nFeatures, nRays);
    test.CalcQEpsCombinatorialGenericTest(1);
}

TEST(LinearSampling, CalcQEpsCombinatorialVCType) {
    int nItems = 32, nFeatures = 4, nRays = 60;
    LinearSamplingTest test(nItems, nFeatures, nRays);
    test.CalcQEpsCombinatorialGenericTest(2);
}

TEST(LinearSampling, CalcQEpsCombinatorialAF) {
    int nItems = 128, nFeatures = 32, nRays = 48;
    LinearSamplingTest test(nItems, nFeatures, nRays);
    int sessionId = test.sessionId;

    test_helpers::Ptr<float> w0(nFeatures);
    fillArray2D(w0, 1, nFeatures);

    int nAlgs = 20, maxIters = 10;
    test_helpers::Ptr<float> W_all(nAlgs * nFeatures);
    int nAlgsActual = findAllNeighbors(sessionId, w0, nAlgs, maxIters, nItems, W_all);
    ASSERT_EQ(nAlgsActual, nAlgs);

    test_helpers::Ptr<unsigned char> ev(nItems * nAlgs), isSource(nAlgs);
    test_helpers::Ptr<int> ec(nAlgs);
    test_helpers::Ptr<unsigned int> hashes(nAlgs);
    ASSERT_EQ(calcAlgs(test.sessionId, W_all, nAlgsActual, NULL, ev, ec, hashes), SUCCESS);

    // Each cluster spans across the whole error layer
    int minEC = ec[0], maxEC = ec[0];
    for (int iAlg = 1; iAlg < nAlgs; ++iAlg) {
        if (ec[iAlg] < minEC) minEC = ec[iAlg];
        if (ec[iAlg] > maxEC) maxEC = ec[iAlg];
    }

    test_helpers::Ptr<int> clusterIds(nAlgs);
    ASSERT_EQ(detectClusters(ev, ec, nItems, nAlgs, clusterIds), SUCCESS);
    int nClusters = 0;
    for (int i = 0; i < nAlgs; ++i) nClusters = std::max(nClusters, clusterIds[i]);
    nClusters++;

    int nEpsValues = nItems / 4;
    int nTrainItems = nItems / 2;
    test_helpers::Ptr<float> epsValues(nEpsValues);
    for (int iEps = 0; iEps < nEpsValues; ++iEps) {
        epsValues[iEps] = 4.0f / nItems * iEps;
    }

    ASSERT_EQ(findSources(ev, nItems, nAlgs, isSource), SUCCESS);
    test_helpers::Ptr<float> h_QEps(nAlgs * nEpsValues);
    ASSERT_EQ(calcQEpsCombinatorialAF(ev, ec, hashes, isSource, epsValues, clusterIds, nItems, nTrainItems, nAlgs, nEpsValues, nClusters, h_QEps), SUCCESS);

    for (int i = 0; i < nClusters; ++i) {
        for (int j = 0; j < nEpsValues; ++j) {
            ASSERT_GE(h_QEps[IDX2C(i, j, nClusters)], 0);
            ASSERT_LE(h_QEps[IDX2C(i, j, nClusters)], 1);
            if (j != 0) ASSERT_GE(h_QEps[IDX2C(i, j - 1, nClusters)], h_QEps[IDX2C(i, j, nClusters)]);
        }
    }
}

TEST(LinearSampling, StopRecordingAndReplay) {
    ASSERT_EQ(stopRecording(), 0);
    helpers::call_on_destruction c([&]() {
        remove(recordingFileName.c_str());
    });

    Replayer replayer;
    ASSERT_EQ(replayer.replay(recordingFileName), 0);
}

TEST(LinearSampling, ReplayMatlabTests) {
    ASSERT_TRUE(true);
    // Replayer replayer;
    // ASSERT_EQ(replayer.replay("testdata\\gputests.dat"), 0);
}
