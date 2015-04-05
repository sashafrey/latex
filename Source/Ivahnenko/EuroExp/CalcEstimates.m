function result=CalcEstimates(res)
params.nItersMCC = 512; %4096;             % monte-carlo cross-validation
params.nItersRandomWalk = 512; %2*4096;    % random walk 
params.nAlgsToSample = 512; %2*4096;
params.nRays = 512; %1024;                 % number of rays
params.randomSeed = 0;               % random seed
params.nStartAlgs = 16;        
params.linearSamplingDll_path = 'd:\!Study\TNOP\vft11ccas-2\Source\Common\GPU';
params.nLayers = 15;
params.pTransition = 0.8;
params.ellRatio = 0.95;

GPU_Initialize(params.linearSamplingDll_path);
trainErr = zeros(length(res.algs),1);
CombVCInverse = zeros(length(res.algs),1);
CombSCInverse = zeros(length(res.algs),1);
CombESInverse = zeros(length(res.algs),1);
parfor i=1:length(res.algs)
    fprintf('%d\n', i);
    ts = GetSubFeaturesSpace(res.task, res.algs{i});
    ts = AddConstantFeature(ts);
    randomRays = 2 * (rand(params.nRays, ts.nFeatures) - 0.5);
    sessionId = GPU_CreateSession(ts.objects, ts.target - 1, randomRays);

    warning off
    [w] = glmfit(ts.objects(:,1:end-1), ts.target - 1, 'binomial', 'link', 'logit');
    warning on
    w = [w(2:end); w(1)];
    w = w ./ sqrt(w'* w);

    trainErr(i) = mean((2 * ts.target-3) .* (ts.objects * w) <= 0);

    % Combinatorial overfitting
    [~, ~, trainErrorCount, ~] = GPU_CalcAlgs(sessionId,  w');

    % сэмплируем и считаем оценку
    W0 = ones(params.nStartAlgs, 1) * w';
    [algs_rw, sources] = GPU_RunRandomWalking(sessionId, ...
        W0, params.nAlgsToSample, params.nItersRandomWalk, trainErrorCount + params.nLayers, ...
        params.pTransition, params.randomSeed);

    [~, ev, ec, hashes] = GPU_CalcAlgs(sessionId, algs_rw);
    
    epsStep = min(0.0001, 0.05 * 4 / ts.nItems);
    CombEps = 0 : epsStep : 1; 
    CombVC = sum(GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 2, CombEps, int32(params.ellRatio * ts.nItems)));
    CombVCInverse(i) = min(CombEps(CombVC < 0.5));

    CombSC = sum(GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 1, CombEps, int32(params.ellRatio * ts.nItems)));
    CombSCInverse(i) = min(CombEps(CombSC < 0.5));

    CombES = sum(GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 0, CombEps, int32(params.ellRatio * ts.nItems)));
    CombESInverse(i) = min(CombEps(CombES < 0.5));

    GPU_CloseAllSessions()
end;
result.trainErr = trainErr;
result.CombVCInverse = CombVCInverse;
result.CombSCInverse = CombSCInverse;
result.CombESInverse = CombESInverse;

end