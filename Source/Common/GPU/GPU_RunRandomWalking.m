function [W, isSource] = GPU_RunRandomWalking(sessionId, W0, maxAlgs, maxIters, maxErrors, pTransition, seed, allowSimilar)
    % GPU_RunRandomWalking
    % Performs random walk on SC-graph (also known as 1-inclusion graph)
    % starting from multiple classifiers.
    %
    % Usage: 
    % [W, isSource] = GPU_RunRandomWalking(sessionId, W0, maxAlgs, maxIters, maxErrors, seed),
    % where
    %   - sessionId is an identificator of session, produced by
    %   GPU_CreateSession() call;
    %   - W0  is a matrix of size [nAlgs x nFeatures] that describes the
    %   classifiers to start traversal from,
    %   - maxAlgs limits the number of algorithms produced by this
    %   operation,
    %   - maxIters limits the number of internal iterations allowed for
    %   this operation. Optional, default = 1e9 (no limitation).
    %   - maxErrors limits the number of error. No classifiers with error
    %   count greather than maxErrors will be considered during traversal.
    %   Optional, default = nItems (no limitation).
    %   - pTransition defines the probability of transition from M to M+1 error layer.
    %   Whenever pTransition < 1, it creates an "force" that tries to keeps 
    %   search in lower layers of the graph. Optional, default = 0.
    %   - seed is an number defining behaviour of random number generator.
    %   Optional, default = 0 (time-initialized random number).
    %   - allowSimilar - boolean flag (0 or 1), defines whether to allow
    %   duplicates algorithms in the output. Optional, default = 0.
    %   - isSource is a vector of size [nAlgs] that contains flags
    %   defining which classifiers are "sources" in terms of E. Sokolov
    %   bound.

    [nItems, nFeatures, nRays, deviceId] = GPU_GetSessionStats(sessionId);
    
    if (~exist('seed', 'var'))
        seed = 0;
    end
    
    if (~exist('maxErrors', 'var')) 
        maxErrors = nItems + 1;
    end
    
    if (~exist('maxIters', 'var'))
        maxIters = int32(1024 * 1024 * 1024);
    end
    
    if (~exist('allowSimilar', 'var'))
        allowSimilar = 0;
    end
    allowSimilar = int32(allowSimilar);
    
    if (~exist('pTransition', 'var'))
        pTransition = 1.0;
    end
    
    if (~isa(pTransition, 'single'))
        pTransition = single(pTransition);
    end
    
    Check(~isempty(W0));
    Check(size(W0, 2) == nFeatures);
    n_w0 = size(W0, 1);
    Check(n_w0 > 0);
    
    
    W = zeros(maxAlgs, nFeatures, 'single');
    isSource = zeros(maxAlgs, 1, 'uint8');
    
    if (~isa(W0, 'single'))
        W0 = single(W0);
    end
    
    [nAlgs, W1, pTransition1, W, isSource] = calllib(GPU_LibName, 'runRandomWalking', ...
        sessionId, W0, n_w0, maxAlgs, maxIters, maxErrors, allowSimilar, pTransition, seed, W, isSource);
    GPU_CheckErrCode(nAlgs);
    if (nAlgs < maxAlgs)
        W = W(1:nAlgs, :);
        isSource = isSource(1:nAlgs, :);
    end

    % ToDo: investigate further if encouter this problem.
    if (nAlgs == 0) 
        throw(MException('GPU:RunRandomWalking', 'GPU_RunRandomWalking failed'));
    end;
end
