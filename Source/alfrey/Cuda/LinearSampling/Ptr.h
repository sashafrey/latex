#ifndef __PTR
#define __PTR

#include <memory>

#include "helpers.h"
#include "SessionManager.h"

namespace helpers {

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
        reset();
        if (size > 0) {
            ptr_ = (T*)malloc(sizeof(T) * size);
        }
    }

    void resize(int nRows, int nColumns) {
        resize(nRows * nColumns);
    }

    void reset() {
        if (ptr_) free(ptr_);
        ptr_ = NULL;
    }
    
    // Pure evil, don't use it.
    // operator T*() { return ptr_; }
    // operator const T*() const { return ptr_; }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    typedef std::tr1::shared_ptr<Ptr<T>> ptr;
private:
    DISALLOW_COPY_AND_ASSIGN(Ptr);
    T* ptr_;
};

}

typedef Session<helpers::Ptr<float>::ptr, helpers::Ptr<int>::ptr> CpuSession;
typedef SessionManager<CpuSession> CpuSessionManager;

#endif // __PTR