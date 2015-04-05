// This file is from http://stackoverflow.com/questions/5028302/small-logger-class
#ifndef __LOG_H__
#define __LOG_H__

#include <cstdarg>
#include <cstdio>
#include <sstream>
#include <string>

#include "common.h"
#include "helpers.h"

inline std::string NowTime();

enum TLogLevel {logSILENT, logERROR, logWARNING, logINFO, logDEBUG, logDEBUG1, logDEBUG2, logDEBUG3, logDEBUG4};

template <typename T>
class Log
{
public:
    Log();
    virtual ~Log();
    std::ostringstream& Get(TLogLevel level = logINFO);
public:
    static TLogLevel& ReportingLevel();
    static std::string ToString(TLogLevel level);
    static TLogLevel FromString(const std::string& level);
    static TLogLevel FromInt(int level);

protected:
    std::ostringstream os;
private:
    Log(const Log&);
    Log& operator =(const Log&);
};

template <typename T>
Log<T>::Log()
{
}

template <typename T>
std::ostringstream& Log<T>::Get(TLogLevel level)
{
    os << helpers::getDateTime() << " " << ToString(level) << " ";
    return os;
}

template <typename T>
Log<T>::~Log()
{
    os << std::endl;
    T::Output(os.str());
}

template <typename T>
TLogLevel& Log<T>::ReportingLevel()
{
    static TLogLevel reportingLevel = logDEBUG4;
    return reportingLevel;
}

template <typename T>
std::string Log<T>::ToString(TLogLevel level)
{
	static const char* const buffer[] = {
        "SILENT ",
        "ERROR  ", 
        "WARNING", 
        "INFO   ", 
        "DEBUG  ", 
        "DEBUG1 ", 
        "DEBUG2 ",
        "DEBUG3 ", 
        "DEBUG4 "};

    return buffer[level];
}

template <typename T>
TLogLevel Log<T>::FromString(const std::string& level)
{
    if (level == "DEBUG4")
        return logDEBUG4;
    if (level == "DEBUG3")
        return logDEBUG3;
    if (level == "DEBUG2")
        return logDEBUG2;
    if (level == "DEBUG1")
        return logDEBUG1;
    if (level == "DEBUG")
        return logDEBUG;
    if (level == "INFO")
        return logINFO;
    if (level == "WARNING")
        return logWARNING;
    if (level == "ERROR")
        return logERROR;
    if (level == "SILENT")
        return logSILENT;

    Log<T>().Get(logWARNING) << "Unknown logging level '" << level << "'. Using INFO level as default.";
    return logINFO;
}

template <typename T>
TLogLevel Log<T>::FromInt(int level)
{
    if (level == logDEBUG4)
        return logDEBUG4;
    if (level == logDEBUG3)
        return logDEBUG3;
    if (level == logDEBUG2)
        return logDEBUG2;
    if (level == logDEBUG1)
        return logDEBUG1;
    if (level == logDEBUG)
        return logDEBUG;
    if (level == logDEBUG)
        return logINFO;
    if (level == logWARNING)
        return logWARNING;
    if (level == logERROR)
        return logERROR;
    if (level == logSILENT)
        return logSILENT;
    Log<T>().Get(logWARNING) << "Unknown logging level '" << level << "'. Using INFO level as default.";
    return logINFO;
}

class Output2FILE
{
public:
    static void Output(const std::string& msg);
    static void Output2FILE::OutputFormat(const char* level, const char* format, ...);
};

inline void Output2FILE::Output(const std::string& msg)
{   
    FILE* pStream;
    if (fopen_s( &pStream, LOG_FILENAME, "a" ) != 0) {
        fprintf(stderr, "Unable to open log %s for write.", LOG_FILENAME);
        return;
    }
    
    fprintf(pStream, "%s", msg.c_str());
    fclose(pStream);
}

inline void Output2FILE::OutputFormat(const char* level, const char* format, ...)
{   
    FILE* pStream;
    if (fopen_s( &pStream, LOG_FILENAME, "a" ) != 0) {
        fprintf(stderr, "Unable to open log %s for write.", LOG_FILENAME);
        return;
    }
    
    fprintf(pStream, "%s %s ", helpers::getDateTime().c_str(), level);
    
    va_list args;
    va_start(args, format);
    vfprintf(pStream, format, args);
    va_end(args);

    fprintf(pStream, "\n");
    fclose(pStream);
}

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#   if defined (BUILDING_FILELOG_DLL)
#       define FILELOG_DECLSPEC   __declspec (dllexport)
#   elif defined (USING_FILELOG_DLL)
#       define FILELOG_DECLSPEC   __declspec (dllimport)
#   else
#       define FILELOG_DECLSPEC
#   endif // BUILDING_DBSIMPLE_DLL
#else
#   define FILELOG_DECLSPEC
#endif // _WIN32

class FILELOG_DECLSPEC FILELog : public Log<Output2FILE> {};
//typedef Log<Output2FILE> FILELog;

#ifndef FILELOG_MAX_LEVEL
#define FILELOG_MAX_LEVEL logDEBUG4
#endif

#define LOG(level) \
    if (level > FILELOG_MAX_LEVEL) ;\
    else if (level > FILELog::ReportingLevel()) ; \
    else FILELog().Get(level)

#define LOG_F(level, format, ...) \
    if (level > FILELOG_MAX_LEVEL) ;\
    else if (level > FILELog::ReportingLevel()) ; \
    else Output2FILE::OutputFormat(FILELog::ToString(level).c_str(), format, __VA_ARGS__)    

#define IS_NULL_STR(ptr) ((ptr == NULL) ? "==NULL" : "!=NULL")

#endif //__LOG_H__
