function cmp = CompareGradesAndTestErrors(cvLog, iFunc, iCvFold, iBaseAlg, ...
        iFeatureSelectionStep)
    
    gradesLog = ...
        cvLog.featureSelectionLogs{iFunc, iCvFold}{iBaseAlg}.featureSelectionLog.gradesLog(iFeatureSelectionStep);
    gradesLog = gradesLog{1};
    X = cvLog.trainSamples{iCvFold}.X;
    Y = cvLog.trainSamples{iCvFold}.Y;
    X_test = cvLog.testSamples{iCvFold}.X;
    Y_test = cvLog.testSamples{iCvFold}.Y;

    testError = zeros(length(gradesLog.grades), 1);

    for i = 1:length(gradesLog.grades)
        X_curr = X(:, gradesLog.features(i, :));
        X_test_curr = X_test(:, gradesLog.features(i, :));

        w = logisticRegressionLearner(X_curr, Y);
        X_test_curr = [X_test_curr, ones(size(X_test_curr, 1), 1)];
        Y_predicted = sign(X_test_curr * w);

        testError(i) = mean(Y_test ~= Y_predicted);
    end

    cmp = [testError, gradesLog.grades];

    % заменяем числа в cmp на ранги
    for i = 1:size(cmp, 2)
        [~, ix] = sort(cmp(:, i));
        ranks = 1:length(ix);
        ranks(ix) = ranks;
        cmp(:, i) = ranks;
    end
end