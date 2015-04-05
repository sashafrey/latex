#include "ls_api_cpu_intl.h"

#include <cassert>
#include <cfloat>

#include <algorithm>
#include <tuple>
#include <vector>

#include "common.h"
#include "helpers.h"
#include "Log.h"
#include "Ptr.h"

int kAnalyzeClusters(
    const unsigned char* ev,
    const int* clusterIds,
    const int* sourceIds,
    int nAlgs,
    int nSources,
    int nItems,
    int nClusters,
    int* clustersInferiority,
    int* clustersCommonEC,
    int* clustersUnionEC)
{
    // re-sort cluster ids so that they are sequential.
    typedef std::tuple<int, int> Key;
    std::vector<Key> clusterVec;
    clusterVec.reserve(nAlgs);
    for (int i = 0; i < nAlgs; ++i) {
        clusterVec.push_back(Key(clusterIds[i], i));
    }

    std::sort(clusterVec.begin(), clusterVec.end(), [](const Key& a, const Key& b) -> bool { return std::get<0>(a) < std::get<0>(b); });
    helpers::Ptr<int> clusterBegin(nClusters), clusterEnd(nClusters);
    int iCluster = 0;
    clusterBegin.get()[0] = 0;
    for (int iAlg = 1; iAlg < nAlgs; ++iAlg) {
        if (std::get<0>(clusterVec[iAlg]) == std::get<0>(clusterVec[iAlg - 1])) continue;
        clusterEnd.get()[iCluster] = iAlg;
        iCluster++;
        clusterBegin.get()[iCluster] = iAlg;
    }

    clusterEnd.get()[iCluster] = nAlgs;
    helpers::Ptr<unsigned char> upperVec(nItems), lowerVec(nItems);
    for (int iCluster = 0; iCluster < nClusters; ++iCluster)
    {
        for (int iItem = 0; iItem < nItems; ++iItem) { 
            upperVec.get()[iItem] = 0; 
            lowerVec.get()[iItem] = 1; 
        }

        for (int indexAlg = clusterBegin.get()[iCluster]; indexAlg < clusterEnd.get()[iCluster]; ++indexAlg) {
            int iAlg = std::get<1>(clusterVec[indexAlg]);
            for (int iItem = 0; iItem < nItems; ++iItem) {
                unsigned char cur = ev[IDX2C(iItem, iAlg, nItems)];
                upperVec.get()[iItem] = upperVec.get()[iItem] || cur;
                lowerVec.get()[iItem] = lowerVec.get()[iItem] && cur;
            }
        }

        int upperEC = 0, lowerEC = 0;
        for (int iItem = 0; iItem < nItems; ++iItem) { 
            upperEC += upperVec.get()[iItem];
            lowerEC += lowerVec.get()[iItem];
        }
        
        clustersCommonEC[iCluster] = lowerEC;
        clustersUnionEC[iCluster] = upperEC;
        
        // The "right way" according to theoretical results:
        // clustersInferiority[iCluster] = kCalcInferiorityHelper(ev, lowerVec, sourceIds, nItems, nSources, nAlgs);

        // Heuristic that works better:
        clustersInferiority[iCluster] = kCalcInferiorityHelper(ev, upperVec.get(), sourceIds, nItems, nSources, nAlgs);
    }

    return SUCCESS;
}

void kCalcAlgs(
    const float* scores, 
    const int* target, 
    int nItems, 
    int nAlgs, 
    int* errorsCount, 
    unsigned int* hashes) 
{
    // ToDo: turn loop over iAlg into parfor
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        int curErrors = 0;
        unsigned int curHash = 17;
        for (int iItem = 0; iItem < nItems; ++iItem) {
            int predictedTarget = PREDICTED_TARGET(scores[IDX2C(iItem, iAlg, nItems)]);
            curHash = (curHash * 23 + predictedTarget);
            if (predictedTarget != target[iItem]) curErrors++;
        }

        if (errorsCount != NULL) errorsCount[iAlg] = curErrors;
        if (hashes != NULL) hashes[iAlg] = curHash;
    }
}

int kCalcAlgSourceQEpsClassicSC(
    const int* algsEC,
    const int* algsUpperCon,
    const int* algsInferiority,
    const float* logfact,
    const float* epsValues,
    int nItems,    
    int nTrainItems,
    int nAlgs, 
    int nEpsValues,
    float* QEps)
{
    assert((nTrainItems > 0) && (nTrainItems < nItems));
    
    // ToDo: Turn loop over iAlgs into parfor. 
    // WARNING: move qeps_sum insize (each thread must have its own)
    int qeps_sums_length = nItems;
    helpers::Ptr<float> qeps_sums(qeps_sums_length);
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        for (int iEps = 0; iEps < nEpsValues; ++iEps) {
            QEps[IDX2C(iAlg, iEps, nAlgs)] = FLT_MAX;
        }

        const int m = algsEC[iAlg];
        const int u = algsUpperCon[iAlg];
        const int q = algsInferiority[iAlg];
        
        const int L = nItems;
        const int ell = nTrainItems;
        const int k = L - ell;

        memset(qeps_sums.get(), 0, sizeof(float) * qeps_sums_length);

        int j_max = ell * m / L;
        double sum = 0.0;
        for (int j_ = 0; j_ <= j_max; ++j_) {
            int L_ = L - u - q;        
            int ell_ = ell - u;
            int m_ = m - q;
            if ((m_ >= j_) && (L_ >= m_) && (ell_ >= j_) && ((L_ - m_) >= (ell_ - j_))) 
            {
                if (helpers::kBinCoeffIsZero(m_, j_)) continue;
                if (helpers::kBinCoeffIsZero(L_ - m_, ell_ - j_)) continue;
                sum += exp(helpers::kBinCoeffLog(m_, j_, logfact) + helpers::kBinCoeffLog(L_ - m_, ell_ - j_, logfact) - helpers::kBinCoeffLog(L, ell, logfact));
            }

            qeps_sums.get()[j_] = (float)sum;
        }

        for (int iEps = 0; iEps < nEpsValues; ++iEps) {
            float s_max = ell * (m - epsValues[iEps] * k) / L;
            QEps[IDX2C(iAlg, iEps, nAlgs)] = (s_max >= 0) ? qeps_sums.get()[(int)s_max] : 0.0f;
        }
    }

    return 0;
}

int kCalcAlgSourceQEpsAFrey(
    const int* clustersEC,
    const int* clustersUpperCon,
    const int* clustersInferiority,
    const int* clustersCommonErrors,
    const int* clustersUnionErrors,
    const int* clustersSizes,
    const float* logfact,
    const float* epsValues,
    int nItems,
    int nTrainItems,
    int nClusters, 
    int nEpsValues,
    float* QEps)
{
    assert((nTrainItems > 0) && (nTrainItems < nItems));
    
    int L = nItems;
    int ell = nTrainItems;
    int k = L - ell;

    for (int iCluster = 0; iCluster < nClusters; ++iCluster) {
        for (int iEps = 0; iEps < nEpsValues; ++iEps) {
            QEps[IDX2C(iCluster, iEps, nClusters)] = 0.0f;
        }

        int L_prime = L - clustersUpperCon[iCluster] - clustersInferiority[iCluster];
        int ell_prime = ell - clustersUpperCon[iCluster];
        int k_prime = L_prime - ell_prime;
        
        if (helpers::kBinCoeffIsZero(L_prime, ell_prime)) continue;
        double P_cluster = exp(helpers::kBinCoeffLog(L_prime, ell_prime, logfact) - helpers::kBinCoeffLog(L, ell, logfact));

        int m = clustersCommonErrors[iCluster];
        int r = clustersUnionErrors[iCluster] - m;
        int rho = clustersEC[iCluster] - m;
        int m_prime = std::max(m - clustersInferiority[iCluster], 0);

        float epsKoef  = ((float)L_prime * (float)ell * (float)k) / ((float)L * (float)ell_prime * (float)k_prime);
        float epsConst = (((float)m_prime) - ((float)m * (float)ell * (float)L_prime)/((float)L * (float)ell_prime)) / (float)k_prime;

        for (int i = 0; i <= std::min(m_prime, ell_prime); ++i) {
            for (int j = 0; j <= std::min(r, ell_prime - i); ++j) {
                if (helpers::kBinCoeffIsZero(m_prime, i)) continue;
                if (helpers::kBinCoeffIsZero(r, j)) continue;
                if (helpers::kBinCoeffIsZero(L_prime - m_prime - r, ell_prime - i - j)) continue;
                double tau = P_cluster * 
                             exp(helpers::kBinCoeffLog(m_prime, i, logfact) + 
                                 helpers::kBinCoeffLog(r, j, logfact) + 
                                 helpers::kBinCoeffLog(L_prime - m_prime - r, ell_prime - i - j, logfact) - 
                                 helpers::kBinCoeffLog(L_prime, ell_prime, logfact));
                
                float trainErr = (float)(i + std::max(0, rho - r - j));
                float testErr = (float)(m_prime + rho - trainErr);
                float eps_max = testErr / (float)k_prime - trainErr / (float)ell_prime;
                eps_max = (eps_max - epsConst) / epsKoef;
                
                for (int iEps = 0; iEps < nEpsValues; ++iEps) {
                    if ((epsValues[iEps] <= eps_max) ||             // normal comparison
                        (abs(epsValues[iEps] - eps_max) < 1e-6))    // hack to heal rounding errors
                    { 
                        QEps[IDX2C(iCluster, iEps, nClusters)] += (float)tau;
                    }
                }
            }
        }
    }

    return 0;
}

void kCalcAlgSourceQEpsESokolov(
    const int* algsEC,
    const int* algsUpperCon,
    const int* ecAlgsNotSources,
    const int* ecSourcesNotAlgs,
    const float* logfact,
    const float* epsValues,
    int nItems,    
    int nTrainItems,
    int nAlgs, 
    int nSources, 
    int nEpsValues,
    float* QEps)
{
    // ToDo: Turn both loops (over iAlg and over iSource) into parfor. 
    //  WARNING: take care about atomic_min (!)
    //  WARNING: move qeps_sum insize (each thread must have its own)
    int qeps_sums_length = nItems;
    helpers::Ptr<float> qeps_sums(qeps_sums_length);
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        for (int iEps = 0; iEps < nEpsValues; ++iEps) {
            QEps[IDX2C(iAlg, iEps, nAlgs)] = FLT_MAX;
        }

        for (int iSource = 0; iSource < nSources; ++iSource) {
            const int m = algsEC[iAlg];
            const int u = algsUpperCon[iAlg];
            const int q = ecAlgsNotSources[IDX2C(iAlg, iSource, nAlgs)];
            const int T = std::min(q, ecSourcesNotAlgs[IDX2C(iAlg, iSource, nAlgs)]);
            const int L = nItems;
            const int ell = nTrainItems;
            const int k = L - ell;

            memset(qeps_sums.get(), 0, sizeof(float) * qeps_sums_length);

            int j_max = ell * m / L;
            for (int j_ = 0; j_ <= j_max; ++j_) {
                double sum = 0.0f;
                int t = 0;
                for (int t = 0; t <= T; ++t) {
                    int L_ = L - u - q;        
                    int ell_ = ell - u - t;
                    int m_ = m - q;
                    int j = j_ - t;
                    if (helpers::kBinCoeffIsZero(q, t)) continue;
                    if (helpers::kBinCoeffIsZero(m_, j)) continue;
                    if (helpers::kBinCoeffIsZero(L_ - m_, ell_ - j)) continue;
                    sum += exp(helpers::kBinCoeffLog(q, t, logfact) +
                               helpers::kBinCoeffLog(m_, j, logfact) +
                               helpers::kBinCoeffLog(L_ - m_, ell_ - j, logfact) -
                               helpers::kBinCoeffLog(L, ell, logfact));
                }

                qeps_sums.get()[j_] = (float)sum;
            }

            for (int j = 1; j <= j_max; ++j) {
                qeps_sums.get()[j] += qeps_sums.get()[j - 1];
            }

            for (int iEps = 0; iEps < nEpsValues; ++iEps) {
                float* targetLocation = &QEps[IDX2C(iAlg, iEps, nAlgs)];
                float s_max = ell * (m - epsValues[iEps] * k) / L;
                float potentialValue = (s_max >= 0) ? qeps_sums.get()[(int)s_max] : 0.0f;

                // atomicMin((int*)targetLocation, *(int*)(&potentialValue));
                *targetLocation = std::min(*targetLocation, potentialValue);
            }
        }
    }
}

void kCalcEV(
    const float* scores, 
    const int *target, 
    int nItems, 
    int nAlgs, 
    unsigned char *errorVectors)
{
    // ToDo: turn loop over iAlg into parfor
    for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
        for (int iItem = 0; iItem < nItems; ++iItem) {
            int currentTarget = target[iItem];
            int index = IDX2C(iItem, iAlg, nItems);
            int predictedTarget = PREDICTED_TARGET(scores[index]);
            errorVectors[index] = (predictedTarget != currentTarget);
        }
    }
}

int kCalcInferiorityHelper(
    const unsigned char *evSources,
    const unsigned char *evAlg,
    const int *sourceIds,
    int nItems,
    int nSources,
    int nAlg)
{
    helpers::Ptr<unsigned char> evAlgsAdjusted(nItems);
    memcpy(evAlgsAdjusted.get(), evAlg, sizeof(unsigned char) * nItems);
    int inferiority = 0;
    for (int idxSource = 0; idxSource < nSources; ++idxSource) {
        int iSource = sourceIds[idxSource];
        
        bool sourcePrecAlg = true;
        for (int iItem = 0; iItem < nItems; ++iItem) {
            unsigned char eAlg = evAlg[iItem];
            unsigned char eSource = evSources[IDX2C(iItem, iSource, nItems)];
            if (eAlg == 0 && eSource == 1) {
                sourcePrecAlg = false;
                break;
            }
        }

        if (sourcePrecAlg) {
            for (int iItem = 0; iItem < nItems; ++iItem) {
                unsigned char eAlg = evAlgsAdjusted.get()[iItem];
                unsigned char eSource = evSources[IDX2C(iItem, iSource, nItems)];
                if (eAlg == 1 && eSource == 0) {
                    inferiority++;
                    evAlgsAdjusted.get()[iItem] = 0;
                }
            }
        }
    }

    return inferiority;
}

void kCompareAlgsToSources(
    const unsigned char* ev,
    const int* algIds,
    const int* sourceIds,
    int nAlgs,
    int nSources,
    int nItems,
    int nEV,
    int* ecAlgsNotSources,
    int* ecSourcesNotAlgs,
    int* ecAlgsInferiority)
{
    // ToDo: turn loop over iAlgs into parfor.
    for (int idxAlg = 0; idxAlg < nAlgs; ++idxAlg) {
        int iAlg = algIds[idxAlg];
        int algInferiority = 0;
        helpers::Ptr<unsigned char> algErrors(nItems);
        for (int iItem = 0; iItem < nItems; ++iItem) {
            algErrors.get()[iItem] = ev[IDX2C(iItem, iAlg, nItems)];
        }

        for (int idxSource = 0; idxSource < nSources; ++idxSource) {
            int iSource = sourceIds[idxSource];

            int algNotSource = 0;
            int sourceNotAlg = 0;
            for (int iItem = 0; iItem < nItems; ++iItem) {
                unsigned char eAlg = ev[IDX2C(iItem, iAlg, nItems)];
                unsigned char eSource = ev[IDX2C(iItem, iSource, nItems)];
                if (eAlg > eSource) algNotSource++;
                if (eSource > eAlg) sourceNotAlg++;
            }

            if ((ecAlgsInferiority != NULL) && (sourceNotAlg == 0)) {
                // Source is "pure" for iAlg and might contribute into inferiority.
                for (int iItem = 0; iItem < nItems; ++iItem) {
                    unsigned char eAlg = algErrors.get()[iItem];
                    unsigned char eSource = ev[IDX2C(iItem, iSource, nItems)];
                    if ((eSource == 0) && (eAlg == 1)) {
                        algErrors.get()[iItem] = 0;
                        algInferiority++;
                    }
                }
            }
            
            if (ecAlgsNotSources != NULL)  ecAlgsNotSources[IDX2C(idxAlg, idxSource, nAlgs)] = algNotSource;
            if (ecSourcesNotAlgs != NULL)  ecSourcesNotAlgs[IDX2C(idxAlg, idxSource, nAlgs)] = sourceNotAlg;
            if (ecAlgsInferiority != NULL) ecAlgsInferiority[idxAlg] = algInferiority;
        }
    }
}

void kFindNeighbors( 
    const float *d_X, 
    const float *d_R, 
    const float *d_XR, 
    const float *d_w0,
    const float *d_Xw0,
    const int *d_rayId,
    const int *d_wId,
    int nItems, 
    int nFeatures, 
    int nTargets, 
    int nW,
    float *d_t ) 
{
    // ToDo: turn look over iTarget into parfor
    for (int iTarget = 0; iTarget < nTargets; ++iTarget) {
        const int iRay = d_rayId[iTarget];
        const int iW = d_wId[iTarget];

        float result = 0.0f;
        float ignore = FLT_MAX;

        // First iteration finds the smallest, second iteration - second smallest "t".
        for (int iIter = 0; iIter < 2; iIter++) {
            float minT = FLT_MAX;
            for (int iItem = 0; iItem < nItems; ++iItem) {
                float curT = - d_Xw0[IDX2C(iItem, iW, nItems)] / d_XR[IDX2C(iItem, iRay, nItems)];
                if ((curT > 0) && (curT < minT) && (curT != ignore)) {
                    minT = curT;
                }        
            }

            if (minT != FLT_MAX) {
                result = result + minT / 2.0f;
                ignore = minT;
            } else {
                result = helpers::get_qnan();
                break;
            }
        }

        d_t[iTarget] = result;
    }
}

void kPerformCrossValidation(
    const unsigned char* errorVectors, 
    const int* nTotalErrorsPerAlg, 
    int nItems, 
    int nTrainItems,
    int nAlgs, 
    int nIters, 
    int randomSeed,
    int* trainEC,
    int* testEC) 
{
    // ToDo: turn the loop over iters into parfor (take care of different seeds!)
    for (int iIter = 0; iIter < nIters; ++iIter) {
        if (randomSeed != 0) {
            srand(randomSeed + iIter);
        }

        // Generate train mask as random permutation of boolean vector.
        helpers::Ptr<char> trainMask(nItems);
        for (int iItem = 0; iItem < nItems; ++iItem) trainMask.get()[iItem] = (iItem < nTrainItems);
        for (int iItem = nItems - 1; iItem > 0; --iItem) {
            int pos = ((int)rand() + (int)RAND_MAX * (int)rand()) % (iItem + 1);
            char tmp = trainMask.get()[iItem];
            trainMask.get()[iItem] = trainMask.get()[pos];
            trainMask.get()[pos] = tmp;
        }
        
        static const int Uninitialized = -1;
        int bestId = Uninitialized;
        int nMinTrainErrors = nItems + 1;
        int nMaxTotalErrors = Uninitialized;
        for (int iAlg = 0; iAlg < nAlgs; ++iAlg) {
            int nTrainErrors = 0;
            for (int iItem = 0; iItem < nItems; ++iItem) {
                nTrainErrors += errorVectors[IDX2C(iItem, iAlg, nItems)] * trainMask.get()[iItem];
            }

            int nTotalErrors = nTotalErrorsPerAlg[iAlg];
            if ((nTrainErrors < nMinTrainErrors) || ((nTrainErrors == nMinTrainErrors) && (nTotalErrors >= nMaxTotalErrors))) {
                nMinTrainErrors = nTrainErrors;
                nMaxTotalErrors = nTotalErrors;
                bestId = iAlg;
            }
        }

        if (trainEC != NULL) trainEC[iIter] = nMinTrainErrors;
        if (testEC != NULL) testEC[iIter] = nTotalErrorsPerAlg[bestId] - nMinTrainErrors;
    }
}