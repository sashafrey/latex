function [w_list, w_hash] = GPU_FindNeighborsMatlab(w, X, R, XR)
    % Matlab implementation of GPU_FindNeighbors.
    % XR is expected to be a matrix product of feature matrix and matrix of random directions.
    Check(size(w, 1) == 1);
    
    nFeatures = size(X, 2);
    nRays = size(R, 1);
    
    w_list = zeros(nRays, nFeatures);
    w_hash = int32(zeros(nRays, 1));
    
    Xw = X * w';
    
    for iRay = 1:nRays
        t = - Xw ./ XR(:, iRay);
        % sum((w + t(1) * R(1, :)) .* X(1, :))
        t = t ( t > 0 );
        if (size(t) < 2)
            continue;
        end
        
        [tMin1, tMin1id] = min(t);
        t(tMin1id) = NaN;
        tMin2 = min(t);
        tStar = (tMin1 + tMin2) / 2;
        w_new = (w + R(iRay, :) * tStar);
        w_list(iRay, :) = w_new;
        w_hash(iRay) = getLC_hash(w_new, X);
    end   

    % w_list = w_list(w_hash ~= 0, :);
    % w_hash = w_hash(w_hash ~= 0);

    %[~, ids] = unique(w_hash);
    
    % w_list = w_list(ids, :);
    % w_hash = w_hash(ids);
end

function hash = getLC_hash(w, X)
    ev = (sign(X * w') < 0);
    max = 64*1024*1024;
    hash = int32(17);
    for i=1:length(ev)
        hash = mod(hash * 23 + int32(ev(i)), max);
    end
 end
