function [QEps, eps] = GPU_CalcQEpsCombinatorialAF(ev, ec, hashes, isSource, clusterIds, eps, nTrainItems)
    % GPU_CalcQEpsCombinatorialAF
    % Estimates overfitting based on A. Frey formula.
    %
    % Usage:
    %  [QEps, eps] = GPU_CalcQEpsCombinatorialAF(ev, ec, hashes, isSource, clusterIds, eps, nTrainItems)
    %   - ev is a matrix of size [nItems * nAlgs] that describes error
    %   vectors,
    %   - ec is a vector of length [nAlgs] that contains error counts,
    %   - hashes is a vector of size [nAlgs] that contains hashes,
    %   calculated based on error vectors.
    %   - isSource is a vector of size [nAlgs] that contains flags
    %   defining which classifiers are "sources" in terms of E. Sokolov
    %   bound.
    %   - clusterIds - ids of clusters.
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
    
    if (~isa(clusterIds, 'int32'))
        clusterIds = int32(clusterIds);
    end
    
    nItems = size(ev, 1);
    nAlgs = size(ev, 2);
    Check(length(isSource) == nAlgs);
    Check(length(ec) == nAlgs);
    Check(length(hashes) == nAlgs);
    Check(length(clusterIds) == nAlgs);
    
    nClusters = length(unique(clusterIds));
    Check(all(unique(clusterIds)' == 0:(nClusters-1))); % clusters enumerated 0 .. nClusters-1
    for i = 0 : (nClusters - 1)
        Check(length(unique(ec(clusterIds == i))) == 1); % within cluster all algs has the same number of errors.
    end
        
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
    
    QEps = zeros(nClusters, nEpsValues, 'single');
    [errCode, ev1, ec1, hashes1, isSource1, eps1, clusterIds1, QEps] = calllib(GPU_LibName, 'calcQEpsCombinatorialAF', ...
        ev, ec, hashes, isSource, single(eps), clusterIds, nItems, int32(nTrainItems), nAlgs, int32(nEpsValues), int32(nClusters), QEps);
    GPU_CheckErrCode(errCode);
end

% Unfortunately, matlab routine 'linkage' crashes from time to time, so
% this function has being retired, and replaced with ad-hoc implementation
% in LinearSampling library.
function clusterIds = performClustering(ev, ec, maxClusterDiameter)
    guid = mexCreateGUID;
    guid(guid=='{' | guid=='}' | guid=='-') = [];
    filename = [datestr(now, 'yyyymmddTHHMMSS_FFF'), guid, '.mat'];
    save(filename, 'ev', 'ec', 'maxClusterDiameter');
    nAlgs = length(ec);
    if (maxClusterDiameter == 0)
        clusterIds = (1:nAlgs) - 1;
    else
        maxid = 0;
        clusterIds = zeros(nAlgs, 1);
        for m = unique(ec)'
            evm = ev(:, ec==m);
            n = size(evm, 1);
            if (size(evm, 2) > 1)
                Z = linkage(double(evm'), 'complete', 'hamming');
                Z(:, 3) = Z(:, 3) * n;
                %dendrogram(Z, 0)
                T = cluster(Z,'cutoff',maxClusterDiameter,'criterion','distance');
                Check(all(unique(T)' == 1:length(unique(T))));
            else
                T = 1;
            end
            T = T + maxid;
            maxid = max(T);
            clusterIds(ec == m) = T;
            %fprintf('%i %i\n', length(unique(T)), size(evm, 2))
        end
        clusterIds = clusterIds' - 1;
    end
    
    delete(filename )
end
   
    