function res = runVExp(task)
N=4000;
nInt = 20;
res.trainErr = [];
res.testErr = [];
task = ConvertNominalFeatures(task);
task = NormalizeFeatures(task, false, 2);
nFold = 10;
cvFolds = GenerateNFoldCVIndexes(task.target, nFold);

%task = GenerateNewFeatures(task);
%if (task.nItems>300)
%    ids = randsampleStratified(task.target, 300);
%    task = GetTaskSubsample(task,ids);
%end;
%idx = 1:task.nFeatures;

rng('default') % for reproducibility
[B,FitInfo] = lasso(task.objects,task.target,'NumLambda',25,'CV',10);
fSet = find(B(:, FitInfo.Index1SE));
ts = GetSubFeaturesSpace(task, fSet);
r = GetSubCalcFast(ts, cvFolds, nFold);
res.lassoSet = fSet;
res.lTrainErr = r.trainErr;
res.lTestErr = r.testErr;

t1 = task;
t1.objects(isnan(t1.objects)) = 0;
[~, p] = corrcoef([t1.objects, t1.target]);
[b, idx] = sort(abs(p(1:t1.nFeatures, t1.nFeatures+1)),'descend');
pdfs = PDFsampler(b);
%pdfs = PDFsampler(linspace(0,1,length(idx)));
trainErr = zeros(N,1);
testErr = zeros(N,1);

randP = GenerateRandomFeaturesSubsets(task, pdfs, idx, N);
%randP = GenerateFullFeaturesSubset(fs, idx);
parfor i=1:N
    curTrainSet = GetSubFeaturesSpace(task, randP{i});
    fprintf('%d \n',i);
    %try
        r = GetSubCalcFast(curTrainSet, cvFolds, nFold);
        trainErr(i) = r.trainErr;
        testErr(i) = r.testErr;
    %catch err
    %    continue;
    %end;
end;
res.trainErr = trainErr';
res.testErr = testErr';
res.algs = randP;
[~, i] = min(res.trainErr);
res.bestSet = randP{i};

[~, idx] = sort(res.trainErr, 'descend');
intervals = floor(linspace(1, N, nInt));
intervals(end) = N;
intervals(1) = 0;
for i=2:length(intervals)
    int = intervals(i-1)+1:intervals(i);
    [~, idx2] = sort(res.testErr(idx(int)), 'descend');
    t = idx(int);
    idx(int) = t(idx2);
end
res.intervals = intervals;
res.sortIdx = idx;
res.trainErr = res.trainErr(idx);
res.testErr = res.testErr(idx);
res.task = task;
plotResult(res);
end

function full = GenerateFullFeaturesSubset(n, idx)
full = cell(2^n-1,1);

for i=1:length(full)
    full{i} = idx(de2bi(i, n)==1);
end;

end

function randP = GenerateRandomFeaturesSubsets(task, pdfs, idx, N)
randP = cell(N,1);
for i=1:floor(N/2)
    fCount = 1+randi(task.nFeatures-1);
    perm = zeros(fCount,1);
    for j=1:fCount
        nr = pdfs.nextRandom;
        number = 1 + floor((1-nr)*task.nFeatures);
        perm(j) = number;
    end;
    perm(perm==0) = [];
    randP{i} = unique(idx(perm));
end;

for i=floor(N/2)+1:N
    fCount = randi(task.nFeatures-1);
    perm = zeros(fCount,1);
    for j=1:fCount
        nr = pdfs.nextRandom;
        number = task.nFeatures - floor((1-nr)*task.nFeatures);
        perm(j) = number;
    end;
    perm(perm==0) = [];
    randP{i} = setdiff(1:task.nFeatures, unique(idx(perm)));
end;
end

function result = GetSubCalcFast(ts, cvFolds, nFold)
trainErr = zeros(nFold,1);
testErr = zeros(nFold,1);
ts = AddConstantFeature(ts);
for i=1:nFold
    train_set = GetTaskSubsample(ts, cvFolds~=i);
    w = TrainOnSample(train_set);
    trainErr(i) = CalcErrorOnSample(train_set, w);
    test_set = GetTaskSubsample(ts, cvFolds==i);
    testErr(i) = CalcErrorOnSample(test_set, w);
end;

result.trainErr = mean(trainErr);
result.testErr = mean(testErr);
end