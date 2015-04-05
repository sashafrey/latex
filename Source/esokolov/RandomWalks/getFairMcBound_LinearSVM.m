function bound = getFairMcBound_LinearSVM(task, ell, nSplits)

    bound = 0;
    
    for currSplit = 1:nSplits
        if mod(currSplit, 100) == 0
            fprintf('\t%d\n', currSplit);
        end
        
        [train_set, test_set] = SplitTask(task, ell / task.nItems);
        [X, Y] = ConvertTaskToESokolovFormat(train_set);
        [X_test, Y_test] = ConvertTaskToESokolovFormat(test_set);
        X(:, end) = [];
        X_test(:, end) = [];
        
        params = '-t 0 -c 1 -q';
        model_curr = svmtrain2(Y, X, params);
        Y_model = svmpredict(Y, X, model_curr, '-q');
        Y_test_model = svmpredict(Y_test, X_test, model_curr, '-q');
        
        err_train = sum(Y ~= Y_model) / length(Y);
        err_test = sum(Y_test ~= Y_test_model) / length(Y_test);
        
        bound = bound + err_test;
    end
    
    bound = bound / nSplits;
end
