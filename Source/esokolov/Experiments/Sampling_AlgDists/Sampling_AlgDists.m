clear;

task = LoadTask('wine');
task = NormalizeFeatures(task, false, 2);
[train_set, test_set] = SplitTask(task, 0.2102);

params.nItersMCC = 1024;  % monte-carlo cross-validation
params.nItersRandomWalk = 10000; % random walk 
params.nAlgsToSample = 2000;
params.nRays = 128;             % number of rays
params.randomSeed = 0;   % random seed
params.nStartAlgs = 1;
params.linearSamplingDll_path = 'D:\Science\vtf11ccas\Source\Common\GPU';
params.nLayers = 100;
params.pTransition = 0.5;
params.allowSimilar = 1;

w_start = trainLogisticRegression(train_set, test_set);
train_set = AddConstantFeature(train_set);
test_set = AddConstantFeature(test_set);

[~, ~, errCnt, ev] = CalcLinearSamplingMCC(train_set, w_start, params);

evd = CalcEvPairwiseDist(ev, 0);

plot(errCnt, 'LineWidth', 2);
set(gcf, 'Position', [500, 500, 500, 200]);
return;

%subplot(2, 1, 1);
subplot('Position', [0.1 0.4 0.8 0.5]);
imagesc(evd);
colormap(flipud(gray));

%h = subplot(2, 1, 2);
subplot('Position', [0.1 0.1 0.8 0.2]);
plot(errCnt, 'LineWidth', 2);

set(gcf, 'Position', [500, 500, 400, 500]);