function isSource = GPU_FindSources(ev)
    % GPU_FIND_SOURCES
    % Selects classifiers with no incoming edged in SC-graph (known  also
    % as 1-inclusion graph).
    % 
    % Usage:
    % isSource = GPU_FINDSOURCES(ev)
    % - ev is a matrix of size [nItems * nAlgs] that describes error
    % vectors,
    % - isSource is a vector of size [nAlgs] that contains flags
    % defining which classifiers are "sources" in terms of E. Sokolov
    % bound.
    
    if (~isa(ev, 'uint8'))
        ev = uint8(ev);
    end
    
    nItems = size(ev, 1);
    nAlgs = size(ev, 2);
    isSource = zeros(nAlgs, 1, 'uint8');
    [errCode, ev1, isSource] = calllib(GPU_LibName, 'findSources', ev, nItems, nAlgs, isSource);
    GPU_CheckErrCode(errCode);
end
