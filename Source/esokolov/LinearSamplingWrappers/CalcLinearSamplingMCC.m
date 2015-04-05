function [trainErrorMCC, testErrorMCC, algsErrCnt, ev] = ...
        CalcLinearSamplingMCC(train_set, w_start, params)
    
    if ~exist('params', 'var')
        params = GetDefaultLinearSamplingParams;
    end
    
    if ~isfield(params, 'allowSimilar')
        params.allowSimilar = 0;
    end
    
    Check(all(unique(train_set.target) == [1; 2]));

    GPU_Initialize(params.linearSamplingDll_path);
    GPU_SetLogLevel(0);
    
    randomRays = 2 * (rand(params.nRays, train_set.nFeatures) - 0.5);
    session_train = GPU_CreateSession(train_set.objects, train_set.target - 1, randomRays);

    [~, ~, trainErrorCount, ~] = GPU_CalcAlgs(session_train, w_start');

    % сэмплируем и считаем оценку
    W0 = ones(params.nStartAlgs, 1) * w_start';
    [algs_rw, sources] = GPU_RunRandomWalking(session_train, ...
        W0, params.nAlgsToSample, params.nItersRandomWalk, ...
        trainErrorCount + params.nLayers, params.pTransition, params.randomSeed, ...
        params.allowSimilar);
    if (size(algs_rw, 1) == 0) 
        fprintf('randomWalkFailed\n'); 
        trainErrorMCC = NaN;
        testErrorMCC = NaN;
        return;
    end;

    % оцениваем переобучение МЭР с помощью папы карло
    [~, ev, ec, ~] = GPU_CalcAlgs(session_train, algs_rw);
    [~, ~, trainErrorMCCOneIter, testErrorRateMCCOneIter] = ...
        GPU_PerformCrossValidation(ev, ec, params.nItersMCC, params.randomSeed);

    trainErrorMCC = mean(trainErrorMCCOneIter);
    testErrorMCC = mean(testErrorRateMCCOneIter);

    GPU_CloseAllSessions();
    
    algsErrCnt = ec;
end