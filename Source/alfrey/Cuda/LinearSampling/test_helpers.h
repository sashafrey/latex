#ifndef __TEST_HELPERS_H
#define __TEST_HELPERS_H

#include <cstdlib>

namespace test_helpers {

void fillArray2D(float* p, int nRows, int nColumns)
{
    int iRow, iColumn;
    for (iRow = 0; iRow < nRows; ++iRow) {
        for (iColumn = 0; iColumn < nColumns; ++iColumn) {
            p[iRow * nColumns + iColumn] = (float)rand() / (float) RAND_MAX - 0.5f;
        }
    }
}

void fillArray2D_int(int* p, int nRows, int nColumns)
{
    int iRow, iColumn;
    for (iRow = 0; iRow < nRows; ++iRow) {
        for (iColumn = 0; iColumn < nColumns; ++iColumn) {
            p[iRow * nColumns + iColumn] = rand() % 2;
        }
    }
}

template<class T>
class Ptr
{
public:
    Ptr() : ptr_(NULL) {
    }

    Ptr(int size) : ptr_(NULL) {
        resize(size);
    }

    Ptr(int nRows, int nColumns) : ptr_(NULL) {
        resize(nRows * nColumns);
    }

    ~Ptr() {
        reset();
    }

    void resize(int size) {
        if (ptr_) free(ptr_);
        ptr_ = (T*)malloc(sizeof(T) * size);
    }

    void resize(int nRows, int nColumns) {
        resize(nRows * nColumns);
    }

    void reset() {
        if (ptr_) free(ptr_);
        ptr_ = NULL;
    }
    
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    typedef std::tr1::shared_ptr<Ptr<T>> ptr;
private:
    DISALLOW_COPY_AND_ASSIGN(Ptr);
    T* ptr_;
};

}

#endif