function [testQual, trainQual, cvLog, learningStats] = comBoost_CV(task, ...
        trainSize, cvFoldsCnt, ...
        maxEnsembleLength, ell_0, ell_1, maxFeatureSubsetSize, ...
        featureSelectionFuncArr, baseClassifierLearnerFunc, ...
        randomizeFeatureSelection, featureSelectionSubsetRatio)

    funcCnt = length(featureSelectionFuncArr);
    
    cvLog = [];
    cvLog.testQuals = nan(funcCnt, cvFoldsCnt);
    cvLog.trainQuals = nan(funcCnt, cvFoldsCnt);
    cvLog.ensembleWeights = cell(funcCnt, cvFoldsCnt);
    cvLog.featureSelectionLogs = cell(funcCnt, cvFoldsCnt);
    cvLog.trainSamples = cell(1, cvFoldsCnt);
    cvLog.testSamples = cell(1, cvFoldsCnt);
    
    for currFold = 1:cvFoldsCnt
        [train_set, test_set] = SplitTask(task, trainSize / task.nItems);
        [X, Y] = ConvertTaskToESokolovFormat(train_set);
        [X_test, Y_test] = ConvertTaskToESokolovFormat(test_set);
        
        cvLog.trainSamples{currFold}.X = X;
        cvLog.trainSamples{currFold}.Y = Y;
        cvLog.testSamples{currFold}.X = X_test;
        cvLog.testSamples{currFold}.Y = Y_test;
            
        for currFuncIdx = 1:funcCnt
            currFeatureSelectionFunc = featureSelectionFuncArr{currFuncIdx};

            [ensembleWeightsCurr, learningLogCurr] = ...
                comBoost(X, Y, maxEnsembleLength, ell_0, ell_1, maxFeatureSubsetSize, ...
                currFeatureSelectionFunc, baseClassifierLearnerFunc, ...
                randomizeFeatureSelection, featureSelectionSubsetRatio);

            Y_predicted = comBoost_classify(X, ensembleWeightsCurr);
            Y_test_predicted = comBoost_classify(X_test, ensembleWeightsCurr);

            testQualCurr = sum(Y_test == Y_test_predicted) / length(Y_test);
            trainQualCurr = sum(Y == Y_predicted) / length(Y);

            cvLog.testQuals(currFuncIdx, currFold) = testQualCurr;
            cvLog.trainQuals(currFuncIdx, currFold) = trainQualCurr;
            cvLog.ensembleWeights{currFuncIdx, currFold} = ensembleWeightsCurr;
            cvLog.featureSelectionLogs{currFuncIdx, currFold} = learningLogCurr;
        end
        
        save('./esokolov/ComBoost_Experiments/comboost_cv_backup.mat', ...
            'cvLog');
    end
    
    testQual = mean(cvLog.testQuals, 2);
    trainQual = mean(cvLog.trainQuals, 2);
    
    learningStats = {};
end
