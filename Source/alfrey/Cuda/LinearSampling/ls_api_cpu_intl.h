#ifndef __LINEAR_SAMPLING_CPU_INTERNAL_H
#define __LINEAR_SAMPLING_CPU_INTERNAL_H

#include "common.h"

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
    int* clustersUnionEC);

void kCalcAlgs(
    const float* scores, 
    const int* target, 
    int nItems, 
    int nAlgs, 
    int* errorsCount, 
    unsigned int* hashes);

int kCalcAlgSourceQEpsClassicSC(
    const int* algsEC,                  // nAlgs, errors count 
    const int* algsUpperCon,            // nAlgs, errors upper connectivity
    const int* algsInferiority,         // nAlgs, algs inveriority
    const float* logfact,               // nItems + 1, table of e-based logarithms of factorials. ln(n!) = logFactorisl[n].
    const float* epsValues,             // nEpsValues, the values of threshold "eps" to calculate the bound QEps(eps).
    int nItems,    
    int nTrainItems,                    // \ell --- number of items in train sample
    int nAlgs, 
    int nEpsValues,                     // length of epsValues
    float* QEps);                       // nAlgs * nEpsValues. QEps(iAlg, iEps) is contribution of iAlg to QEps with eps = epsValues(iEps)

int kCalcAlgSourceQEpsAFrey(
    const int* clustersEC,              // nClusters, errors count,
    const int* clustersUpperCon,        // nClusters, errors upper connectivity,
    const int* clustersInferiority,     // nClusters, number of proper errors of algs,
    const int* clustersCommonErrors,    // nClusters, number of common errors,
    const int* clustersUnionErrors,     // nClusters, number of union errors
    const int* clustersSizes,           // nClusters, number of algorithms per cluster
    const float* logfact,               // nItems + 1, table of e-based logarithms of factorials. ln(n!) = logFactorisl[n].
    const float* epsValues,             // nEpsValues, the values of threshold "eps" to calculate the bound QEps(eps).
    int nItems,    
    int nTrainItems,                    // \ell --- number of items in train sample
    int nClusters, 
    int nEpsValues,                     // length of epsValues
    float* QEps);                       // nClusters * nEpsValues. QEps(iCluster, iEps) is contribution of iCluster to QEps with eps = epsValues(iEps)

void kCalcAlgSourceQEpsESokolov(
    const int* algsEC,         // nAlgs, errors count 
    const int* algsUpperCon,            // nAlgs, errors upper connectivity
    const int* ecAlgsNotSources,        // nAlgs * nSources, number of proper errors of algs comparing to sources
    const int* ecSourcesNotAlgs,        // nAlgs * nSources, number of proper errors of sources comparing to algs
    const float* logfact,               // nItems + 1, table of e-based logarithms of factorials. ln(n!) = logFactorisl[n].
    const float* epsValues,             // nEpsValues, the values of threshold "eps" to calculate the bound QEps(eps).
    int nItems,    
    int nTrainItems,                    // \ell --- number of items in train sample
    int nAlgs, 
    int nSources, 
    int nEpsValues,                     // length of epsValues
    float* QEps);                       // nAlgs * nEpsValues. QEps(iAlg, iEps) is contribution of iAlg to QEps with eps = epsValues(iEps)

void kCalcEV(
    const float* scores, 
    const int *target, 
    int nItems, 
    int nAlgs, 
    unsigned char *errorVectors);

int kCalcInferiorityHelper(
    const unsigned char *evSources,         // nItems * nAlgs
    const unsigned char *evAlg,             // nItems
    const int *sourceIds,                   // nSources
    int nItems,
    int nSources,
    int nAlg);

// For each algorithm and source calculates the number of items where a) alg is better b) source is better
// Input:
//    - ev        - matrix of error vectors, nItems * nEV,
//    - algIds    - vector of algs ids, of length nAlgs, elements must be withing range 0 .. (nEV - 1)
//    - sourceIds - vector of source ids, of length nSources, elements must be withing range 0 .. (nEV - 1)
// Output:
//    - ecAlgsNotSources - matrix of size nAlgs * nSources, containing the number of errors of the alg which are not errors of the source
//    - ecSourcesNotAlgs - matrix of size nAlgs * nSources, containing the number of errors of the source which are not errors of the alg
//    - ecAlgsInferiority - matrix of size nAlgs * 1, containing the inferiority of each algorithm 
//                          (see http://www.machinelearning.ru/wiki/images/7/7b/Voron11colt-submission-eng.pdf for the definition of inferiority)
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
    int* algsInferiority);

// Searches for adjecent cells of multiple starting points in multiple walking dirrections.
// Input:
// d_X, d_R, d_XR --- standard things (object-feature matrix, rays of random dirrections, and their matrix product)
// d_w0 -- matrix of size (nW * nFeatures). Defines the list of potential starting points.
// d_Xw0 -- matrix product of w0 and X
// d_rayId, d_wId -- arrays of the length nTargets, that define the actual starting point and the actual ray dirrection
// d_t -- output array of length nTargets.
void kFindNeighbors( 
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

void kPerformCrossValidation(
    const unsigned char* errorVectors, 
    const int* nTotalErrorsPerAlg, 
    int nItems, 
    int nTrainItems,
    int nAlgs, 
    int nIters, 
    int randomSeed,
    int* trainEC,
    int* testEC);

#endif __LINEAR_SAMPLING_CPU_INTERNAL_H