#ifndef __CUDA_PTR_2D_H
#define __CUDA_PTR_2D_H

#include <memory>

#include <cuda_runtime.h>
#include "Log.h"
#include "SessionManager.h"
#include "CudaException.h"
#include "helpers.h"

template<class T>
class HostPtr2D;

template<class T>
class CudaPtr2D
{
public:
    CudaPtr2D() : nRows_(Uninitialized), nColumns_(Uninitialized), d_ptr_(NULL) {
    }

    CudaPtr2D(int nRows, int nColumns, const T* h_ptr = NULL) : nRows_(Uninitialized), nColumns_(Uninitialized), d_ptr_(NULL) {
        reset(nRows, nColumns, h_ptr);
    }

    CudaPtr2D(const HostPtr2D<T>& h) : nRows_(Uninitialized), nColumns_(Uninitialized), d_ptr_(NULL) {
        reset(h.nRows(), h.nColumns(), h);
    }

    virtual ~CudaPtr2D() {
        dispose();
    }

    void reset(int nRows, int nColumns, const T* h_ptr = NULL) {
        dispose();
        
        CUDA_CHECK_SWALLOW(cudaGetDevice(&deviceId));
        CUDA_CHECK_SWALLOW(cudaMalloc(&d_ptr_, nRows * nColumns * sizeof(T)));
        if (cudaGetLastError() != cudaSuccess) {
            LOG_F(logINFO, "Last cudaMalloc called with nRows = %i, nColumns = %j.", nRows, nColumns);
        }

        nRows_ = nRows;
        nColumns_ = nColumns;

        if (h_ptr != NULL) {
            CUDA_CHECK_SWALLOW(cudaMemcpy(d_ptr_, h_ptr, nRows * nColumns * sizeof(T), cudaMemcpyHostToDevice));
            if (cudaGetLastError() != cudaSuccess) {
                // to avoid partially initialized object (e.g. memory was allocated, but data hasn't been copied).
                dispose();
            }
        }
    }

    void dispose() {
        if (d_ptr_ == NULL) {
            return;
        }

        // we have to make sure that we cudaFree on correct device.
        int tmp; 
        cudaGetDevice(&tmp);
        if (tmp != deviceId) cudaSetDevice(deviceId); 

        cudaFree(d_ptr_);
        d_ptr_ = NULL;
        nRows_ = Uninitialized;
        nColumns_ = Uninitialized;

        if (tmp != deviceId)  cudaSetDevice(tmp);
    }

    operator T*() { return d_ptr_; }
    operator const T*() const { return d_ptr_; }
    T* get() { return d_ptr_; }
    const T* get() const { return d_ptr_; }
    typedef std::tr1::shared_ptr<CudaPtr2D<T>> ptr;

    int nRows() const {
        return nRows_;
    }

    int nColumns() const {
        return nColumns_;
    }
private:
    DISALLOW_COPY_AND_ASSIGN(CudaPtr2D);

    static const int Uninitialized = -1;
    int nRows_;
    int nColumns_;
    int deviceId;
    T* d_ptr_;
};

typedef Session<CudaPtr2D<float>::ptr, CudaPtr2D<int>::ptr> CudaSession;
typedef SessionManager<CudaSession> CudaSessionManager;

#endif // CUDA_PTR_2D_H