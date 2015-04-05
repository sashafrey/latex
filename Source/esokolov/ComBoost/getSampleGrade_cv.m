function grade = getSampleGrade_cv(X, Y, nIters, trainSize, baseClassifierLearner)
    L = size(X, 1);

    testError = 0;
    
    for iIter = 1:nIters
        currTrain = randsample(L, trainSize);
        currTest = setdiff(1:L, currTrain);
        
        X_train = X(currTrain, :);
        Y_train = Y(currTrain);
        X_test = X(currTest, :);
        Y_test = Y(currTest);
        
        w = baseClassifierLearner(X_train, Y_train);

        X_test = [X_test, ones(size(X_test, 1), 1)];
        Y_predicted = sign(X_test * w);

        testError = testError + mean(Y_test ~= Y_predicted);
    end
    
    grade = testError / nIters;
end