function bound = getCvBound_Linear(X, Y, iterCnt, classifierLearner)

    task = ConvertESokolovFormatToTask(X, Y);
    trainSize = floor(size(X, 1) / 2);

    bound = 0;
    for currIter = 1:iterCnt
        [train_set, test_set] = SplitTask(task, trainSize / task.nItems);
        [X, Y] = ConvertTaskToESokolovFormat(train_set);
        [X_test, Y_test] = ConvertTaskToESokolovFormat(test_set);
        
        w = classifierLearner(X, Y);
        
        X_1 = [X, ones(size(X, 1), 1)];
        X_test_1 = [X_test, ones(size(X_test, 1), 1)];
        trainErr = sum(sign(X_1 * w) ~= Y) / size(X, 1);
        testErr = sum(sign(X_test_1 * w) ~= Y_test) / size(X_test, 1);
        
        bound = bound + testErr;
    end
    
    bound = bound / iterCnt;
end
