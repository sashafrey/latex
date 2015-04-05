function [QEps, eps] = GPU_CalcQEpsCombinatorial(sessionId, W, isSource, boundType, eps, nTrainItems)
    % GPU_CalcQEpsCombinatorial
    % Estimates overfitting based on E. Sokolov formula.
    %
    % Usage:
    %  [QEps, eps] = GPU_CalcQEpsCombinatorial(sessionId, W, isSource)
    %   - sessionId is an identificator of session, produced by
    %   GPU_CreateSession() call,
    %   - W  is a matrix of size [nAlgs x nFeatures] that describes the
    %   classifiers,
    %   - isSource is a vector of size [nAlgs] that contains flags
    %   defining which classifiers are "sources" in terms of E. Sokolov
    %   bound.
    %   - boundType - optional, type of the bound to calculate. 
    %       0 - ESokolov bound (default),
    %       1 - Classic SC-bound,
    %       2 - Simple VC-type bound.
    %   - eps is a vector with eps-thresholds to calculate QEps at.
    %   Optional, default = 0, 4/L, 8/L, ...
    %   - nTrainItems is a number corresponding to size of train subsample 
    %   (\ell). Optional, default = L/2.
    %   - QEps is a vector of the same size as eps. QEps(i) gives the
    %   tail probability of the deviation on test and train sample larger
    %   than eps(i).
   
    [nItems, nFeatures, nRays, deviceId] = GPU_GetSessionStats(sessionId);
    Check(size(W, 2) == nFeatures);
    
    if (~isa(W, 'single'))
        W = single(W);
    end
    
    nAlgs = size(W, 1);
    Check(length(isSource) == nAlgs);
    
    if (~exist('eps', 'var'))
        nEpsValues = floor(nItems / 4);
        eps = zeros(1, nEpsValues);
        for i=1:nEpsValues
            eps(i) = double(i-1) * 4 / double(nItems);
        end
    else
        nEpsValues = length(eps);
    end 
    
    if (~exist('nTrainItems', 'var'))
        nTrainItems = nItems / 2;
    end
    
    if (~exist('boundType', 'var'))
        boundType = 0;
    end
    
    Check(ismember(boundType, [0 1 2]))
    QEps = zeros(nAlgs, nEpsValues, 'single');
    [errCode, W1, isSource1, eps1, QEps] = calllib(GPU_LibName, 'calcQEpsCombinatorial', ...
        sessionId, W, isSource, single(eps), int32(nTrainItems), nAlgs, int32(nEpsValues), int32(boundType), QEps);
    
    GPU_CheckErrCode(errCode);    
end
