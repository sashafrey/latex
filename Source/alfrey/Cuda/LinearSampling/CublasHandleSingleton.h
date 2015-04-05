#ifndef __CUBLAS_HANDLE_SINGLETON_H
#define __CUBLAS_HANDLE_SINGLETON_H

#include <cublas_v2.h>
#include "Log.h"
#include "CudaException.h"

class CublasHandleSingleton
{
private:
    CublasHandleSingleton() { 
        LOG_F(logINFO, "Initializing cublas");
        CUBLAS_CHECK_SWALLOW(cublasCreate(&cublasHandle));
    }

    ~CublasHandleSingleton() {
        LOG_F(logINFO, "Deinitializing cublas");
        CUBLAS_CHECK_SWALLOW(cublasDestroy(cublasHandle));
    }

    cublasHandle_t cublasHandle;
public:
    static cublasHandle_t getInstance() {
        static CublasHandleSingleton instance;
        return instance.cublasHandle;
    }
};

#endif // __CUBLAS_HANDLE_SINGLETON_H