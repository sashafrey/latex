function [W, t] = GPU_FindNeighbors(sessionId, w0)
    % GPU_FindNeighbors
    % Finds neighbours along all rays starting from single classifier w0.
    %
    % WARNING: method may return NaN's for some rays, implying that search
    % from w0 along corresponding ray fails to locate a neighbour.
    %
    % Usage:
    %  [W, t] = GPU_FINDNEIGHBORS(sessionId, W), where 
    %   - sessionId is an identificator of session, produced by
    %   GPU_CreateSession() call;
    %   - w0 is a vector of size [1 x nFeatures] that describes the
    %   classifier to locate neighbours for;
    %   - W  is a matrix of size [nRays x nFeatures] that describes the
    %   classifiers being found.
    %   - t is a vector of length [nRays]. All elements t(i) from t
    %   satisfies the following equation: W(i, :) = w0 + t(i) * R(i, :).

    [nItems, nFeatures, nRays, deviceId] = GPU_GetSessionStats(sessionId);
    W = zeros(nRays, nFeatures, 'single');
    t = zeros(nRays, 1, 'single');
    
    if (~isa(w0, 'single'))
        w0 = single(w0);
    end
    
    [errco, w1, W, t] = calllib(GPU_LibName, 'findNeighbors', sessionId, w0, W, t);
    GPU_CheckErrCode(errco);
end