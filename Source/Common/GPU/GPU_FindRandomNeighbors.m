function W = GPU_FindRandomNeighbors(sessionId, W0, seed)
    % GPU_FindRandomNeighbors
    % Finds a random neighbour for each classifier from the given set.
    %
    % WARNING: method may in some cases return NaN's for some of
    % classifiers. Retry with different seed may produce other results.
    %
    % Usage:
    %  [ec, hashes] = GPU_FINDRANDOMNEIGHBORS(sessionId, W, seed), where 
    %   - sessionId is an identificator of session, produced by
    %   GPU_CreateSession() call,
    %   - W0  is a matrix of size [nAlgs x nFeatures] that describes the
    %   classifiers,
    %   - seed is an number defining behaviour of random number generator.
    %   Optional, default = 0 (time-initialized random number).
    %   - W is a matrix of size [nAlgs x nFeatures] that describes
    %   neighbours of the original classifiers.
    
    if (~isa(W0, 'single'))
        W0 = single(W0);
    end
    
    if (exist('seed', 'var'))
        seed = 0;
    end
    
    nW = size(W0, 1);
    [nItems, nFeatures, nRays, deviceId] = GPU_GetSessionStats(sessionId);
    Check(size(W0, 2) == nFeatures);
    
    W = zeros(nW, nFeatures, 'single');
    [err, w1, W] = calllib(GPU_LibName, 'findRandomNeighbors', sessionId, W0, nW, seed, W);
    GPU_CheckErrCode(err);
end

