function W = GPU_FindAllNeighbors(sessionId, w0, maxAlgs, maxIters, maxErrors)
    % GPU_FindRandomNeighbors
    % Performs breadth-first traversal of SC-graph (also known as
    % 1-inclusion graph)
    %
    % Usage:
    %  [ec, hashes] = GPU_FINDALLNEIGHBORS(sessionId, w0, maxAlgs, maxIters, maxErrors), where 
    %   - sessionId is an identificator of session, produced by
    %   GPU_CreateSession() call;
    %   - w0 is a vector of size [1 x nFeatures] that describes starting
    %   point of the traversal,
    %   - maxAlgs limits the number of algorithms produced by this
    %   operation,
    %   - maxIters limits the number of internal iterations allowed for
    %   this operation. Optional, default = 1e9 (no limitation).
    %   - maxErrors limits the number of error. No classifiers with error
    %   count greather than maxErrors will be considered during traversal.
    %   Optional, default = nItems (no limitation).
    %   - W  is a matrix of size [nAlgs x nFeatures] that describes
    %   classifiers found during traversal.
    
    if (~isa(w0, 'single'))
        w0 = single(w0);
    end
    
    [nItems, nFeatures, nRays, deviceId] = GPU_GetSessionStats(sessionId);
    Check(length(w0) == nFeatures);
    
    if (~exist('maxErrors', 'var'))
        maxErrors = nItems + 1;
    end
    
    if (~exist('maxIters', 'var'))
        maxIters = 1024 * 1024 * 1024;
    end
    
    W = zeros(maxAlgs, length(w0), 'single');
    [nAlgs, w1, W] = calllib(GPU_LibName, 'findAllNeighbors', sessionId, w0, maxAlgs, maxIters, maxErrors, W);
    GPU_CheckErrCode(nAlgs);
    if (nAlgs < maxAlgs)
        W = W(1:nAlgs, :);
    end
end
