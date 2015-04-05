function [scores, ev, ec, hashes] = GPU_CalcAlgs(sessionId, W)
    % GPU_CalcAlgs
    % Calculates error vectors, error counts and hashes of classifiers
    % from the set W.
    %
    % Usage:
    %  [ev, ec, hashes] = GPU_CALCALGS(sessionId, W), where 
    %   - sessionId is an identificator of session, produced by
    %   GPU_CreateSession() call,
    %   - W  is a matrix of size [nAlgs x nFeatures] that describes the
    %   classifiers,
    %   - ev is a matrix of size [nItems * nAlgs] that describes error
    %   vectors,
    %   - ec is a vector of length [nAlgs] that contains error counts,
    %   - hashes is a vector of size [nAlgs] that contains hashes,
    %   calculated based on error vectors.
    % Typical usage of hashes is to identify classifiers with distinct
    % error vectors. Classifiers with distinct hashes are guarantied to
    % have distinct error vectors, but not verse versus.

GPU_Initialize;
[nItems, nFeatures, nRays, deviceId] = GPU_GetSessionStats(sessionId);
nAlgs = size(W, 1);
Check(size(W, 2) == nFeatures);
scores = zeros(nItems, nAlgs, 'single');
ec = zeros(nAlgs, 1, 'int32');
ev = zeros(nItems, nAlgs, 'uint8');
hashes = zeros(nAlgs, 1, 'uint32');
[errco, W1, scores, ev, ec, hashes] = calllib(GPU_LibName, 'calcAlgs', sessionId, W, nAlgs, scores, ev, ec, hashes);
GPU_CheckErrCode(errco);
end
