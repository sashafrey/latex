// ======================================================================================
// Global remarks
//  - Each API function returns 0 on success and a negative value on failure, unless otherwise stated in the comments.
//  - Some API functions accept pointers to output buffers. Each such pointer is allowed to be zero (to disable output transport).
//    This is especially useful for functions with more than one output buffer.
//  - Withing this file API functions are sorted alphabeticaly.
//
//  - Order of arguments in API functions is:
//      1. const pointers (input data arrays)
//      2. scalar values (sizes of input data arrays, and other parameters)
//      3. non-const pointeres (output data buffers)
//
//  - Common abbreviations (concepts)
//      ERM         - Empirical risk minimization.
//
//  - Common scalar abbreviations (and corresponding $TeX$ abbreviations from http://www.machinelearning.ru/wiki/images/7/7b/Voron11colt-submission-eng.pdf)
//      nItems      - number of items in the dataset ($L$)
//      nTrainItems - number of items to select for cross-validation ($\ell$)
//      nFeatures   - number of features in the dataset
//      nAlgs       - number of algorithms
//      nEpsValues  - number of points to calculate QEps value
//      nRays       - number of pre-defined directions to search for neighbours of classifiers
//      sessionId   - identifier of a session (which in turn contains X, target and R)
//      randomSeed  - random seed. Use 0 to initialize from timer within the library.
//
//  - Common vector abbreviations, their types and size
//      w0              - vector of length nFeatures, representing weight coefficients of a linear classifier
//      EC, int         - vector of length nAlgs, representing number of errors of each classifiers on some dataset
//      target, int     - vector of length nItems, representing target labels of items in the dataset.
//                        Target labels must be 0 or 1.
//      hashes, uint    - vector of length nAlgs, representing hashes of a set of classifiers.
//                        Hash is based on predicted labels. Distinct hashes imply distinct classifiers. The inverse is not guarantied (hash collisions).
//      upperCon, int   - vector of length nAlgs, representing upper connectivity of classifiers within some set W
//      lowerCon, int   - vector of length nAlgs, representing lower connectivity of classifiers within some set W
//      isSource, uchar - vector of length nAlgs, indicating whether classifiers is a source of some set W
//      epsValues       - vector of length nEpsValues, representing $\eps$ thresholds to use for QEps calculation
//
//  - Common matrix abbreviations, their types and sizes
//      X, float        - "feature matrix" of size nItems * nFeatures, representing items in the dataset
//      R, float        - matrix of size nRays * nFeatures, representing pre-defined directions to search for neighbours of classifiers
//      W, float        - matix of size nFeatures * nAlgs, representing set of classifiers
//                        (in combinatorial overfitting theory "classifier" and "algorithm" means the same thing)
//      EV, uchar       - matrix of size nItems * nAlgs, representing error vectors of some set of classifiers on some dataset
//      scores, float   - matrix of size nItems * nAlgs, representing classifications scores of classifiers set W on dataset (X, target).
//      QEps, float     - matrix of size nAlgs * nEpsValues, representing contribution of individual classifiers into overal overfitting bound.
// ======================================================================================

#ifndef __LINEAR_SAMPLING_H
#define __LINEAR_SAMPLING_H

#ifdef LINEARSAMPLING_EXPORTS
#define LINEARSAMPLING_API __declspec(dllexport)
#else
#define LINEARSAMPLING_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =========== Core routines ============

// Applies given set of algorithms to the data and reports scores, error vectors, error counts, and hashes.
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
LINEARSAMPLING_API int calcAlgs(
    int                 sessionId,
    const float         *W,
    int                 nAlgs,
    float               *scores,
    unsigned char       *EV,
    int                 *EC,
    unsigned int        *hashes);

// Calculates hashes and error counts based on error vectors.
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
LINEARSAMPLING_API int calcAlgsEV(
    const unsigned char *EV,
    const int           *target,
    int                 nItems,
    int                 nAlgs,
    int                 *EC,
    unsigned int        *hashes);

// Calculates lower and upper algs connectivity.
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
// Remark: There is small probability of non-accurate results, because this API function is implemented based on
// hash-based technique, and ignores hash collisions.
LINEARSAMPLING_API int calcAlgsConnectivity(
    const unsigned int  *hashes,
    const int           *EC,
    int                 nAlgs,
    int                 nItems,
    int                 *upperCon,
    int                 *lowerCon);

// Calculates overfitting based on combinatorial formulas.
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
// Utilize different algorithm based on the bound type:
//  0 - ESokolov bound,
//  1 - Splitting-connectivity bound,
//  2 - VC-type bound.
LINEARSAMPLING_API int calcQEpsCombinatorial(
    int                 sessionId,
    const float         *W,
    const unsigned char *isSource,
    const float         *epsValues,
    int                 nTrainItems,
    int                 nAlgs,
    int                 nEpsValues,
    int                 boundType,
    float               *QEps);

// Calculates overfitting based on combinatorial formulas.
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
// Uses Alexander Frey agorithm
// Remark: note differentiation between nAlgs and nClusters in sizes of input arguments.
LINEARSAMPLING_API int calcQEpsCombinatorialAF(
    const unsigned char *EV,                // nItems * nAlgs         - error vectors
    const int           *EC,                // nAlgs                  - error counts
    const unsigned int  *hashes,            // nAlgs                  - hashes
    const unsigned char *isSource,          // nAlgs                  - flags indicating whether algs are sources
    const float         *epsValues,         // nEpsValues             - vector of thresholds "eps" to calculate the bound of QEps(eps)
    const int           *clusterIds,        // nAlgs                  - ids of clusters
    int                 nItems,             //                        - number of items,
    int                 nTrainItems,        //                        - ell, number of items for train sample
    int                 nAlgs,              //                        - number of algorithms
    int                 nEpsValues,         //                        - length of epsValues
    int                 nClusters,          //                        - length of clusterIds
    float               *QEps);             // nClusters * nEpsValues - overfitting per algorithm and thresholds epsValues

// Calculates overfitting based on combinatorial formulas.
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
// Utilize different algorithm based on the bound type:
//  0 - ESokolov bound,
//  1 - Splitting-connectivity bound,
//  2 - VC-type bound.
LINEARSAMPLING_API int calcQEpsCombinatorialEV (
    const unsigned char *EV,
    const int           *EC,
    const unsigned int  *hashes,
    const unsigned char *isSource,
    const float         *epsValues,
    int                 nItems,
    int                 nTrainItems,
    int                 nAlgs,
    int                 nEpsValues,
    int                 boundType,
    float               *QEps);

// Closes the session with given ID, and releases memory hold by session.
LINEARSAMPLING_API int closeSession(int sessionId);

// Closes all sessions.
LINEARSAMPLING_API int closeAllSessions();

// Creates session for a X, target and R.
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
// Returns: - the ID of the session, or negative value in case of error.
// Remarks: - when input sessionId parameter is set to negative value, session manager automatically assigns an ID to this session and returs it.
//          - when input sessionId parameter is set to custom non-negative value, session manager closes curent session with correspondent id (if it exists), and uses provided sessionId.
LINEARSAMPLING_API int createSession(
    const float         *X,
    const int           *target,
    const float         *R,
    int                 nItems,
    int                 nFeatures,
    int                 nRays,
    int                 deviceId,   // GPU device id (specific for GPU implementation)
    int                 sessionId);

// Auto-detect clusters. Current implementation: group pairs of classifiers (a, b) when \rho(a, b) = 2 and m(a, \XX) = m(b, \XX).
LINEARSAMPLING_API int detectClusters(
    const unsigned char *h_algsEV,
    const int           *h_algsEC,
    int                 nItems,
    int                 nAlgs,    
    int                 *clusterIds);

// Finds "all" neighbors of hyperplane w0 (breadth-first search on the SC-graph).
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
// Returns: actual count of neighbors found, or negative value if error happens during evaluation.
// Stop conditions: either reached given number of iterations, or found required number of algorithms.
LINEARSAMPLING_API int findAllNeighbors(
    int                 sessionId,
    const float         *w0,
    int                 maxAlgs,        //                      - max number of algorithms to sample
    int                 maxIterations,  //                      - max number of iterations to perform
    int                 nErrorsLimit,   //                      - limit for algorithms error count
    float               *W);

// Finds neighbors of hyperplane w0 along each ray R (stored in sessionId).
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
// Remarks:
// - output may containt NaN's. This indicates that there is no neighbour along corresponding ray.
LINEARSAMPLING_API int findNeighbors(
    int                 sessionId,
    const float         *w0,
    float               *W,         // nRays * nFeatures    - neighbours
    float               *t);        // nRays                - step along each ray.

// Finds random neighbors of hyperplanes w0.
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
LINEARSAMPLING_API int findRandomNeighbors(
    int                 sessionId,
    const float         *W_input,           // nAlgs * nFeatures     - algorithms (starting points)
    int                 nAlgs,              //                       - number of starting points
    int                 randomSeed,         //
    float               *W_output);         // nAlgs * nFeatures     - neighbours (one neighbour per starting point)

// Determines list of source algoriths.
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
LINEARSAMPLING_API int findSources(
    unsigned char       *EV,
    int                 nItems,
    int                 nAlgs,
    unsigned char       *isSource);

// Returs session statistics.
LINEARSAMPLING_API int getSessionStats(
    int                 sessionId,
    int                 *nItems,        // array of length 1 to store scalar value.
    int                 *nFeatures,     // array of length 1 to store scalar value.
    int                 *nRays,         // array of length 1 to store scalar value.
    int                 *deviceId);     // array of length 1 to store scalar value.

// Performs monte-carlo cross-validation (50% train, 50% test) over the set of error vectors.
// Returns 0 in case of success, or negative value if error happens during evaluation.
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
LINEARSAMPLING_API int performCrossValidation(
    const unsigned char *EV,
    const int           *EC,
    int                 nItems,
    int                 nTrainItems,
    int                 nAlgs,
    int                 nIters,               //                  - number of monte-carlo iterations to perform.
    int                 randomSeed,
    int                 *trainEC,             // nIters * 1       - for each monte-carlo iteration reports train errors count of the best ERM alg.
    int                 *testEC);             // nIters * 1       - for each monte-carlo iteration reports test errors count of the best ERM alg.

// Performs random walk, starting from the set of hyperplanes w0.
// Returns a new set of algorithms and the list of "sources" (algorithms with no incoming edges).
// Sizes of all input/output arrays must be consistent with associated sessionId and sizes.
// See "Global remarks" in ls_api.h for description of common abreviations.
// Stop conditions: either reached given number of iterations, or found required number of algorithms.
// Remark: only returns algorithms with no more than nErrorsLimit errors.
// Returns: actual count of neighbors found, or negative value if error happens during evaluation.
LINEARSAMPLING_API int runRandomWalking(
    int                 sessionId,
    const float         *W_input,           // nAlgsInput * nFeatures - algorithms (starting points)
    int                 nAlgsInput,         //                        - number of starting points
    int                 nAlgsOutput,        //                        - max number of algorithms to sample
    int                 nIters,             //                        - max number of iterations to perform
    int                 nErrorsLimit,       //                        - limit for algorithms error count; doesn't search beyound this threshold.
    int                 allowSimilar,       //                        - flag indicatin whether random walking should allow similar algorithms or exclude them from result.
    const float         *pTransition,       //                        - probability of transition from M to M+1 layer (M+1 to M is always 100%).
    int                 randomSeed,
    float               *W_output,          // nAlgsOutput * nFeatures    - neighbours
    unsigned char       *isSource);         // nAlgsOutput * 1            - flags indicating whether algs are sources

// =========== Debugging functions ===========

LINEARSAMPLING_API int getXR(int sessionId, float *XR);

// Replays recorded blob
LINEARSAMPLING_API int replayRecording(const char* filename);

// Sets log level
LINEARSAMPLING_API int setLogLevel(int logLevel);

// Starts recording of all input data
LINEARSAMPLING_API int startRecording(const char* filename);

// Stops recording of all input data
LINEARSAMPLING_API int stopRecording();

// ===========================================

#ifdef __cplusplus
}
#endif

#endif // __LINEAR_SAMPLING_H