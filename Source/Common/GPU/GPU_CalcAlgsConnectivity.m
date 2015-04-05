function [upperCon, lowerCon] = GPU_CalcAlgsConnectivity(hashes, ec, nItems)
    % GPU_CalcAlgsConnectivity
    % Calculates lower and upper connectivity of classifiers.
    %
    % Usage:
    %  [upperCon, lowerCon] = GPU_CalcAlgsConnectivity(hashes, errorsCount, nItems),
    %   - hashes is a vector of hashes produced by GPU_CalcAlgs or GPU_CalcAlgsEV,
    %   - ec is a vector that contains error counts of the classifiers.
    %   Must be of the same length as hashes.
    %   - nItems is a number of items (objects) in the task.
    %   - upperCon and lowerCon are both vectors of the same length as
    %   hashes and ec. upperCon(i) (lowerCon(i)) give an estimation for the
    %   number of classifiers in upper (lower) neighbourhood of classifier
    %   i.
    %
    % Due to internal trick the calculation is feasible based on hashes and
    % error counts, no other information such as full error vectors is
    % required. This is because hash is calculated as follows:
    %   hash(w) = (signn(<w, x1>) + 23 * sign(<w, x2>) + 23^2 * sign(<w, x3>) + ... ) mod 2^32.
    % When two classifiers have similar classification on all objects
    % except one their hashes are guarantied to be different by (23^k mod
    % 2^32) for some k < nItems. We utilize this fact to estimate upperCon
    % and lowerCon. The result is only guarantied to be an upper bound of
    % actual connectivities due to potential hash collisions.

    nAlgs = length(hashes);
    Check(length(ec) == nAlgs);
    Check(nItems > 0);
    
    Check(isa(hashes, 'uint32'));
    Check(isa(ec, 'int32'));
    
    lowerCon = zeros(nAlgs, 1, 'int32');
    upperCon = zeros(nAlgs, 1, 'int32');
    
    [errco, hashes1, errorsCount1, upperCon, lowerCon] = calllib(GPU_LibName, 'calcAlgsConnectivity', hashes, ec, nAlgs, nItems, upperCon, lowerCon);
    GPU_CheckErrCode(errco);
end
