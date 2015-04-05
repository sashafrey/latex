#ifndef __CUDA_EXCEPTION_H
#define __CUDA_EXCEPTION_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "Log.h"

extern cublasStatus_t _lastCublasCallResult;

inline static const char *_cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        default:
            return "<unknown>";
    }
}

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        LOG_F(logERROR, "CUDA error calling \""#call"\", code is '%i', error message: %s; at %s, line %i, function %s.", cudaGetLastError(), cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__, __FUNCTION__); \
        return LINEAR_SAMPLING_ERROR; \
    }

#define CUDA_CHECK_SWALLOW(call) \
    if((call) != cudaSuccess) { \
        LOG_F(logERROR, "CUDA error calling \""#call"\", code is '%i', error message: %s; at %s, line %i, function %s.", cudaGetLastError(), cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__, __FUNCTION__); \
    }

#define CUBLAS_CHECK(call) \
    if ((_lastCublasCallResult = (call)) != CUBLAS_STATUS_SUCCESS) { \
        LOG_F(logERROR, "CUDA error calling \""#call"\", code is '%i', error message: %s; at %s, line %i, function %s.", _lastCublasCallResult, _cublasGetErrorString(_lastCublasCallResult), __FILE__, __LINE__, __FUNCTION__); \
        return LINEAR_SAMPLING_ERROR; \
    }

#define CUBLAS_CHECK_SWALLOW(call) \
    if ((_lastCublasCallResult = (call)) != CUBLAS_STATUS_SUCCESS) { \
        LOG_F(logERROR, "CUDA error calling \""#call"\", code is '%i', error message: %s; at %s, line %i, function %s.", _lastCublasCallResult, _cublasGetErrorString(_lastCublasCallResult), __FILE__, __LINE__, __FUNCTION__); \
    }

#endif // __CUDA_EXCEPTION_H