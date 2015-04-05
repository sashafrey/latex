// This file contains declaration of CPU-functions, which operates with GPU data.

#ifndef __LINEAR_SAMPLING_DEVICE_CUH
#define __LINEAR_SAMPLING_DEVICE_CUH

#include "ls_api.h"
#include "ls_api_gpu_kernels.cuh"

int transposeMatrix_device(
    unsigned char *odata, 
    const unsigned char *idata, 
    int input_width, 
    int input_height);

int calcAlgs_device(
    int sessionId,
    const float *d_W, 
    int nAlgs,
    float* d_scores,
    unsigned char* d_EV,
    int* d_EC, 
    unsigned int* d_hashes);

int compareAlgsToSources_device(
    unsigned char* d_evAlgs,
    unsigned char* d_evSources,
    int nAlgs,
    int nSources,
    int nItems,
    int* d_ecAlgsNotSources,
    int* d_ecSourcesNotAlgs);

int calcAlgSourceQEps_device(
    const int* algsEC,            // nAlgs, errors count 
    const int* algsConnectivity,        // nAlgs, errors upper connectivity
    const int* ecAlgsNotSources,        // nAlgs * nSources, number of proper errors of algs comparing to sources
    const int* ecSourcesNotAlgs,        // nAlgs * nSources, number of proper errors of sources comparing to algs
    const float* logFactorial_global,    // nItems + 1, table of e-based logarithms of factorials. ln(n!) = logFactorisl[n].
    const float* epsValues,
    int nItems,    
    int nTrainItems,
    int nAlgs, 
    int nSources, 
    int nEpsValues,
    float* QEps);

#endif