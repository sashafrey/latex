#ifndef __HOST_PTR_2D_H
#define __HOST_PTR_2D_H

#include <cuda_runtime.h>

#include <memory>
#include <iostream>

#include "Log.h"
#include "CudaException.h"
#include "helpers.h"

template<class T>
class CudaPtr2D;

template<class T>
class HostPtr2D
{
public:
    HostPtr2D() : nRows_(Uninitialized), nColumns_(Uninitialized), h_ptr_(NULL) {
    }

    HostPtr2D(int nRows, int nColumns, const T* d_ptr = NULL)  : nRows_(Uninitialized), nColumns_(Uninitialized), h_ptr_(NULL) {
        reset(nRows, nColumns, d_ptr);
    }

    HostPtr2D(const CudaPtr2D<T>& d) : nRows_(Uninitialized), nColumns_(Uninitialized), h_ptr_(NULL) {
        reset(d.nRows(), d.nColumns(), d);
    }

    virtual ~HostPtr2D() {
        dispose();    
    }

    void dispose() {
        if (h_ptr_ != NULL) {
            free(h_ptr_);
            h_ptr_ = NULL;
            nRows_ = Uninitialized;
            nColumns_ = Uninitialized;
        }
    }

    void reset(int nRows, int nColumns, const T* d_ptr = NULL) {
        dispose();

        nRows_ = nRows;
        nColumns_ = nColumns;

        h_ptr_ = (T*) malloc(nRows * nColumns * sizeof(T));
        if (d_ptr != NULL) {
            CUDA_CHECK_SWALLOW(cudaMemcpy(h_ptr_, d_ptr, nRows * nColumns * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    operator T*() { return h_ptr_; }
    operator const T*() const { return h_ptr_; }
    T* get() { return h_ptr_; }
    const T* get() const { return h_ptr_; }
    typedef std::tr1::shared_ptr<HostPtr2D<T>> ptr;

    int nRows() const {
        return nRows_;
    }

    int nColumns() const {
        return nColumns_;
    }
private:
    DISALLOW_COPY_AND_ASSIGN(HostPtr2D);

    static const int Uninitialized = -1;
    int nRows_;
    int nColumns_;
    T* h_ptr_;
};

template<typename T>
std::ostream& operator<< (std::ostream& stream, const HostPtr2D<T>& matrix) 
{
    for (int i = 0; i < matrix.nRows(); ++i) {
        for (int j = 0; j < matrix.nColumns(); ++j) {
            stream << matrix.get()[IDX2C(i, j,  matrix.nRows())] << " ";
        }

        stream << std::endl;
    }

    stream << std::endl;
    return stream;
}

#endif // __HOST_PTR_2D_H