function [QEps, eps] = GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, isSource, boundType, eps, nTrainItems)
    % GPU_CalcQEpsCombinatorialEV
    % Estimates overfitting based on E. Sokolov formula.
    %
    % Usage:
    %  [QEps, eps] = GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, isSource, boundType, eps, nTrainItems)
    %   - ev is a matrix of size [nItems * nAlgs] that describes error
    %   vectors,
    %   - ec is a vector of length [nAlgs] that contains error counts,
    %   - hashes is a vector of size [nAlgs] that contains hashes,
    %   calculated based on error vectors.
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

    if (~isa(ev, 'uint8'))
        ev = uint8(ev);
    end
    
    if (~isa(isSource, 'uint8'))
        isSource = uint8(isSource);
    end
    
    if (~isa(ec, 'int32'))
        ec = int32(ec);
    end
    
    if (~isa(hashes, 'uint32'))
        hashes = uint32(hashes);
    end
    
    nItems = size(ev, 1);
    nAlgs = size(ev, 2);
    Check(length(isSource) == nAlgs);
    Check(length(ec) == nAlgs);
    Check(length(hashes) == nAlgs);
    
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
    [errCode, ev1, ec1, hashes1, isSource1, eps1, QEps] = calllib(GPU_LibName, 'calcQEpsCombinatorialEV', ...
        ev, ec, hashes, isSource, single(eps), nItems, int32(nTrainItems), nAlgs, int32(nEpsValues), int32(boundType), QEps);
    GPU_CheckErrCode(errCode);
end