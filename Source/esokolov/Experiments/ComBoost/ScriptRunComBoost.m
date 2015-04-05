%%
clear;


%%
%fprintf('Don''t forget to change back featureSelectionSubsetRatio!!!');
%return;

%%
resultFilename = './esokolov/ComBoost_Experiments/waveform_res_new.mat';
resultFilename = fullfile(resultFilename);

%% убеждаемся, что такого файла не существует
if exist(resultFilename, 'file') == 2
    userAns = ...
        input('This file already exists.\nAre you sure you want to rewrite it? [y/n]: ', ...
        's');
    if ~strcmp(userAns, 'y')
        return;
    end
end    

%%
tasks = LoadTasks('./DataUCI');
task = tasks.waveform;

%% параметры
trainSize = 250;
cvFoldsCnt = 5;
maxEnsembleLength = 10;
ell_0 = 5;
ell_1 = 150;
maxFeatureSubsetSize = 5;
baseClassifierLearnerFunc = @logisticRegressionLearner;

compositionFuncArr = {
    @comBoost;
    @comBoost;
    @comBoost;
    @comBoost;
    @comBoost;
    @comBoost;
    @comBoost;
    @comBoost;
    @comBoost_oneBaseClassifier;
    @comBoost_oneBaseClassifier;
};

%sampleGradingFunc = @(X, Y) getSampleGrade_empirical(X, Y, ...
%    baseClassifierLearnerFunc);
%sampleGradingFunc = @(X, Y) getSampleGrade_subgraph(X, Y, 'mc');
%sampleGradingFunc = @(X, Y) getSampleGrade_subgraph(X, Y, 'qeps_sources');
%sampleGradingFunc = @(X, Y) getSampleGrade_PacBayes(X, Y, ...
%    baseClassifierLearnerFunc, 'di');
sampleGradingFuncArr = {
    @(X, Y) getSampleGrade_subgraph(X, Y, 'qeps_af');
    @(X, Y) getSampleGrade_subgraph(X, Y, 'mc');
    @(X, Y) getSampleGrade_subgraph(X, Y, 'qeps_sources');
    @(X, Y) getSampleGrade_subgraph(X, Y, 'qeps_classic');
    @(X, Y) getSampleGrade_empirical(X, Y, baseClassifierLearnerFunc);
    @(X, Y) getSampleGrade_PacBayes(X, Y, baseClassifierLearnerFunc, 'dd');
    @(X, Y) getSampleGrade_cv(X, Y, 1000, floor(trainSize / 2), baseClassifierLearnerFunc);
    [];
    @(X, Y) getSampleGrade_cv(X, Y, 1000, floor(trainSize / 2), baseClassifierLearnerFunc);;
    [];
};
featureSelectionLegend = {'Comb AF', 'Comb MC', 'Comb QEps ES', 'Comb QEps SC', ...
    'Empirical', 'PAC-Bayes DD', 'CV', 'No selection', ...
    'LogReg with selection', 'LogReg without selection'};

Check(length(compositionFuncArr) == length(sampleGradingFuncArr));
Check(length(compositionFuncArr) == length(featureSelectionLegend));

%featureSelectionFunc = @(X, Y, maxFeaturesCnt) ...
%    selectFeaturesGreedy(X, Y, maxFeaturesCnt, sampleGradingFunc);

featureSelectionFuncArr = cell(length(sampleGradingFuncArr), 1);
for i = 1:length(sampleGradingFuncArr)
    featureSelectionFuncArr{i} = @(X, Y, maxFeaturesCnt) ...
        selectFeaturesGreedy(X, Y, maxFeaturesCnt, sampleGradingFuncArr{i});
end
featureSelectionFuncArr{end} = @selectFeaturesNot;
featureSelectionFuncArr{end - 2} = @selectFeaturesNot;

randomizeFeatureSelection = true;
featureSelectionSubsetRatio = 0.6;
%featureSelectionSubsetRatio = 1;

runParameters = [];
runParameters.task = task;
runParameters.trainSize = trainSize;
runParameters.cvFoldsCnt = cvFoldsCnt;
runParameters.maxEnsembleLength = maxEnsembleLength;
runParameters.ell_0 = ell_0;
runParameters.ell_1 = ell_1;
runParameters.maxFeatureSubsetSize = maxFeatureSubsetSize;
runParameters.baseClassifierLearnerFunc = baseClassifierLearnerFunc;
runParameters.sampleGradingFuncArr = sampleGradingFuncArr;
runParameters.featureSelectionLegend = featureSelectionLegend;
runParameters.featureSelectionFuncArr = featureSelectionFuncArr;
runParameters.randomizeFeatureSelection = randomizeFeatureSelection;
runParameters.featureSelectionSubsetRatio = featureSelectionSubsetRatio;

%%
[testQual, trainQual, cvLog, learningStats] = comBoost_CV(task, ...
    trainSize, cvFoldsCnt, ...
    maxEnsembleLength, ell_0, ell_1, maxFeatureSubsetSize, ...
    featureSelectionFuncArr, baseClassifierLearnerFunc, ...
    randomizeFeatureSelection, featureSelectionSubsetRatio);

%%
save(resultFilename, 'featureSelectionLegend', 'runParameters', ...
    'testQual', 'trainQual', 'cvLog', 'learningStats');
