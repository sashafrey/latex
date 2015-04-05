// Common implementation-specific definitions 
#ifndef __COMMON_H
#define __COMMON_H

// Follow fortran layout for matrices
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

// Mapping between score and target class label
#define PREDICTED_TARGET(score) ((score) < 0) ? 0 : 1

// Return values
const int LINEAR_SAMPLING_ERROR = -1;
const int UNINITIALIZED = -1;
const int SUCCESS = 0;

// Checks function call
#define CHECK(call) \
    if ((call) < 0) { \
        LOG_F(logERROR, "Error calling \""#call"\""); \
        return LINEAR_SAMPLING_ERROR; \
    }

#define LOG_FILENAME "LinearSampling.log"

// List of all operations to use in replayer/recorder serialization
enum OperationType {
    CalcAlgs,
    CalcAlgsEV,
    CalcAlgsConnectivity,
    CalcQEpsCombinatorial,
    CalcQEpsCombinatorialAF,
    CalcQEpsCombinatorialEV,
    CloseSession,
    CloseAllSessions,
    CreateSession,
    FindAllNeighbors,
    FindNeighbors,
    FindRandomNeighbors,
    FindSources,
    GetSessionStats,
    PerformCrossValidation,
    RunRandomWalking
};

// Params of GPU configuration for CUDA code
#define MAX_GRID_SIZE 65535
#define COPY_BLOCK_SIZE 16
#define GRID_SIZE 32
#define BLOCK_SIZE 128

#endif // common.h