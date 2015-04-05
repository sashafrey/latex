function [ec, hashes] = GPU_CalcAlgsEV(ev, target)
    % GPU_CalcAlgsEV
    % Calculates error counts and hashes classifiers based on their error
    % vectors and vector of target class labels.
    %
    % Usage:
    %  [ev, ec, hashes] = GPU_CALCALGSEV(sessionId, W), where 
    %   - ev is a matrix of size [nItems * nAlgs] that describes error
    %   vectors,
    %   - target is a vector of size [nItems] that describes labels of
    %   target classes. Labels must be either 0 or 1.
    %   - ec is a vector of length [nAlgs] that contains error counts,
    %   - hashes is a vector of size [nAlgs] that contains hashes,
    %   calculated based on error vectors.
    % Typical usage of hashes is to identify classifiers with distinct
    % error vectors. Classifiers with distinct hashes are guarantied to
    % have distinct error vectors, but not verse versus.    
    
GPU_Initialize;

if (~isa(ev, 'uint8'))
    ev = uint8(ev);
end

if (~isa(target, 'int32'))
    target = int32(target);
end

nItems = size(ev, 1);
nAlgs = size(ev, 2);
Check(length(target) == nItems);
Check(all(ismember(target, [0, 1])));

ec = zeros(nAlgs, 1, 'int32');
hashes = zeros(nAlgs, 1, 'uint32');
[errco, ev1, target1, ec, hashes] = calllib(GPU_LibName, 'calcAlgsEV', ev, target, nItems, nAlgs, ec, hashes);
GPU_CheckErrCode(errco);
end
