#ifndef __HELPERS_H
#define __HELPERS_H

#include <cassert>
#include <ctime>
#include <functional>
#include <string>

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
// http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&);             \
    void operator=(const TypeName&)        \

namespace helpers {

// An object that accepts a lambda expression end executes it in destructor
class call_on_destruction
{
private:
    std::function<void()> f_;
    DISALLOW_COPY_AND_ASSIGN(call_on_destruction);
public:
    call_on_destruction(std::function<void()> f) : f_(f) {}
    ~call_on_destruction() { f_(); }
};

// Returns true if agrument is NaN (either QNan or SNan)
// Explicitly checks bitmask according to IEEE standard. 
// Check http://en.wikipedia.org/wiki/NaN for more details.
inline bool isnan(float value) {
    int& bits = *((int *)&value);
    return ((bits & 0x7F800000) == 0x7F800000);
}

// Returns a QNan value.
inline float get_qnan() {
    int bits = 0x7fc00000;
    return *((float *)&bits);
}

inline std::string getDateTime()
{
    const int MaxDateTimeStrLength = 26;
    char buffer[MaxDateTimeStrLength];

    time_t rawtime;
    time(&rawtime);

    // use secure version of ctime
    ctime_s(buffer, MaxDateTimeStrLength, &rawtime);

    // overwrite '\r' (char 10) at the end of the date time string.
    buffer[MaxDateTimeStrLength - 2] = '\0';  
    return std::string(buffer);
}

// Returns true if binomial coef (n, k) is not defined.
inline bool kBinCoeffIsZero(int n, int k) {
    return ((n < 0) || (k < 0) || (k > n));    
}

// Returns e-based logarithm of binomial coef (n, k).
inline float kBinCoeffLog(int n, int k, const float* logFactorial) {
    return logFactorial[n] - logFactorial[k] - logFactorial[n - k];
}

} // namespace helpers

#endif