#ifndef __SESSION_MANAGER_H
#define __SESSION_MANAGER_H

#include <cstdlib>
#include <map>

#include "common.h"
#include "Log.h"

// Generic struct to store data of a session (X, target, R, matrix product X * R, and log-factorial table.
// Template type parameters FloatPtr and IntPtr must satisfy the following:
//  - be smart-pointers with reference counting (e.g. be own their memory and be copiable)
//  - have .get() method that returns underlying pointer
template<class FloatPtr, class IntPtr>
struct Session
{
    typedef FloatPtr FloatPtrT;
    typedef IntPtr IntPtrT;

    Session() : id(Uninitialized), nItems(Uninitialized), nFeatures(Uninitialized), nRays(Uninitialized), deviceId(Uninitialized) { ; }

    int id;
    FloatPtr d_X;
    IntPtr d_target;
    FloatPtr d_R;
    FloatPtr d_XR;
    FloatPtr d_logFact;
    
    static const int Uninitialized = -1;
    int nItems;
    int nFeatures;
    int nRays;
    int deviceId;

    bool isInitialized() {
        return id != Uninitialized;
    }

    const float* get_X() { return (*d_X).get(); }
    const int*   get_target() { return (*d_target).get(); }
    const float* get_R() { return (*d_R).get(); }
    const float* get_XR() { return (*d_XR).get(); }
    const float* get_logFact() { return (*d_logFact).get(); }
};

// Generic SessionManager class, which keeps track of active sessions.
// Template type parameter SessionT is expected to be an instance of generic Session.
template<class SessionT>
class SessionManager
{
private:
    SessionManager() { }

    int next_sessionId;
    std::map<int, SessionT> session_map;
public:
    static SessionManager<SessionT>& getInstance() {
        static SessionManager<SessionT> sessionManager;
        return sessionManager;
    }

    int createSession(
        typename SessionT::FloatPtrT d_X, 
        typename SessionT::IntPtrT d_target, 
        typename SessionT::FloatPtrT d_R, 
        typename SessionT::FloatPtrT d_XR, 
        typename SessionT::FloatPtrT d_logFact, 
        int nItems, 
        int nFeatures, 
        int nRays, 
        int deviceId, 
        int sessionId) 
    {
        SessionT session;

        if (sessionId < 0) {
            // iterate next_sessionId until reaching some available slot
            while(session_map.find(next_sessionId) != session_map.end()) {
                next_sessionId++;
            }

            sessionId = next_sessionId++;
        }

        if (session_map.find(sessionId) != session_map.end()) {
            closeSession(sessionId);
        }

        session.id = sessionId;
        session.d_X = d_X;
        session.d_target = d_target;
        session.d_R = d_R;
        session.d_XR = d_XR;
        session.d_logFact = d_logFact;
        session.nItems = nItems;
        session.nFeatures = nFeatures;
        session.nRays = nRays;
        session.deviceId = deviceId;

        LOG_F(logINFO, "Created session: id=%i, nItems=%i, nFeatures=%i, nRays=%i, deviceId=%i", session.id, nItems, nFeatures, nRays, deviceId);
        session_map.insert(std::pair<int, SessionT>(session.id, session));

        return session.id;
    }

    SessionT getSession(int sessionId) {
        auto iter = session_map.find(sessionId);
        if (iter == session_map.end()) 
        {
            LOG_F(logWARNING, "Unable to get session with id=%i", sessionId);
            return SessionT();
        }
        else 
        {
            return iter->second;
        }
    }

    void closeSession(int sessionId) 
    {
        session_map.erase(sessionId);
    }

    void closeAllSessions() 
    {
        session_map.clear();
    }

    ~SessionManager()
    {
    }
};

#endif // __SESSION_MANAGER_H