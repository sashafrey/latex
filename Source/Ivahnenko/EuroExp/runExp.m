function res = runExp(task)
N=500;

task = NormalizeFeatures(task, false, 2);
task = ConvertNominalFeatures(task);
task = GenerateNewFeatures(task);
if (task.nItems>300)
    ids = randsampleStratified(task.target, 300);
    task = GetTaskSubsample(task,ids);
end;
%idx = 1:task.nFeatures;
[~, p] = corrcoef([task.objects, task.target]);
[~, idx] = sort(p(1:task.nFeatures, task.nFeatures+1));

for i=1:N
    fCount = 1+randi(task.nFeatures-1);
    perm = randperm(task.nFeatures);
    randP{i} = perm(idx(1:fCount));
end;

trainErr = zeros(task.nFeatures,1);
testErr = zeros(task.nFeatures,1);
parfor i=1:N
    
    curTrainSet = GetSubFeaturesSpace(task, randP{i});
    fprintf('%d \n ',i);
    %try
        r = GetSubCalcFast(curTrainSet);
        trainErr(i) = r.trainErr;
        testErr(i) = r.testErr;

    %catch err
    %    continue;
    %end;
    
end;


for i=2:N
    if (trainErr(i-1)<trainErr(i))
        trainErr(i) = trainErr(i-1);
        testErr(i) = testErr(i-1);
    elseif (trainErr(i-1)==trainErr(i) && testErr(i-1)>testErr(i))
        trainErr(i) = trainErr(i-1);
        testErr(i) = testErr(i-1);
    end;
end;

res.trainErr = trainErr;
res.testErr = testErr;


end

function result = GetSubCalcFast(ts)
nFold = 2;
cvFolds = GenerateNFoldCVIndexes(ts.target, nFold);
trainErr = zeros(nFold,1);
testErr = zeros(nFold,1);
for i=1:nFold
    train_set = GetTaskSubsample(ts, find(cvFolds ~=i));
    warning off
    [w] = glmfit(train_set.objects, train_set.target - 1, 'binomial', 'link', 'logit');
    warning on
    w = [w(2:end); w(1)];
    w = w ./ sqrt(w'* w);    
    train_set = AddConstantFeature(train_set);
    trainErr(i) = mean((2 * train_set.target-3) .* (train_set.objects * w) <= 0);
    
    test_set = GetTaskSubsample(ts, find(cvFolds ==i));
    test_set = AddConstantFeature(test_set);
    testErr(i) = mean((2 * test_set.target-3) .* (test_set.objects * w) <= 0);
end;

result.trainErr = mean(trainErr);
result.testErr = mean(testErr);
end


function result = GetSubCalc(ts)
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

warning off
        [w] = glmfit(ts.objects, ts.target - 1, 'binomial', 'link', 'logit');
warning on
w = [w(2:end); w(1)];
w = w ./ sqrt(w'* w);

train_set = AddConstantFeature(ts);

result.trainError = mean((2 * train_set.target-3) .* (train_set.objects * w) <= 0);

%% Combinatorial overfitting
GPU_Initialize(params.linearSamplingDll_path);
randomRays = 2 * (rand(params.nRays, train_set.nFeatures) - 0.5);
sessionId = GPU_CreateSession(train_set.objects, train_set.target - 1, randomRays);

[~, ~, trainErrorCount, ~] = GPU_CalcAlgs(sessionId,  w');

% сэмплируем и считаем оценку
W0 = ones(params.nStartAlgs, 1) * w';
[algs_rw, sources] = GPU_RunRandomWalking(sessionId, ...
    W0, params.nAlgsToSample, params.nItersRandomWalk, trainErrorCount + params.nLayers, ...
    params.pTransition, params.randomSeed);

[~, ev, ec, ~] = GPU_CalcAlgs(sessionId, algs_rw);
[~, ~, trainErrorMCCOneIter, testErrorRateMCCOneIter] = GPU_PerformCrossValidation(ev, ec, params.nItersMCC, params.randomSeed);

result.trainErrorMCC = mean(trainErrorMCCOneIter);
result.testErrorMCC = mean(testErrorRateMCCOneIter);
result.OverfitMCC = result.testErrorMCC - result.trainErrorMCC;
[~, ev,ec,hashes] = GPU_CalcAlgs(sessionId, algs_rw);

epsStep = min(0.0001, 0.05 * 4 / train_set.nItems);
CombEps = 0 : epsStep : 1; 
CombVC = sum(GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 2, CombEps, int32(params.ellRatio * train_set.nItems)));
result.CombVCInverse = min(CombEps(CombVC < 0.5));
                    
CombSC = sum(GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 1, CombEps, int32(params.ellRatio * train_set.nItems)));
result.CombSCInverse = min(CombEps(CombSC < 0.5));

CombES = sum(GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 0, CombEps, int32(params.ellRatio * train_set.nItems)));
result.CombESInverse = min(CombEps(CombES < 0.5));

GPU_CloseAllSessions()

    end