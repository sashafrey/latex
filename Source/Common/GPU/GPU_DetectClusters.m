function clusterIds = GPU_DetectClusters(ev, ec)
    % GPU_DetectClusters
    % Detects clusters (in current "weak" implementation cluster is a pair
    % of two algorithms from the same layer, different only at two objects.
    %   - ev is a matrix of size [nItems * nAlgs] that describes error
    %   vectors,
    %   - ec is a vector of length [nAlgs] that contains error counts,

    if (~isa(ev, 'uint8'))
        ev = uint8(ev);
    end
    
    if (~isa(ec, 'int32'))
        ec = int32(ec);
    end
    
    nItems = size(ev, 1);
    nAlgs = size(ev, 2);
    Check(length(ec) == nAlgs);
    clusterIds = zeros(nAlgs, 1, 'int32');
    
    [errCode, ev1, ec2, clusterIds] = calllib(GPU_LibName, 'detectClusters', ...
        ev, ec, nItems, nAlgs, clusterIds);
    
    GPU_CheckErrCode(errCode);    
end
