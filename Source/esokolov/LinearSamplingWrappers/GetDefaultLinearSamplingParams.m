function params = GetDefaultLinearSamplingParams()
    params.nItersMCC = 1024;             % monte-carlo cross-validation
    params.nItersRandomWalk = 4096;      % random walk 
    params.nAlgsToSample = 4096;
    params.nRays = 128;                  % number of rays
    params.randomSeed = 0;               % random seed
    params.nStartAlgs = 256;        
    params.linearSamplingDll_path = 'D:\Science\vtf11ccas\Source\Common\GPU';
    params.nLayers = 15;
    params.pTransition = 0.8;
end