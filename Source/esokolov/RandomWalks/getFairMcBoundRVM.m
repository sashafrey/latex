function bound = getFairMcBoundRVM(task, sigma, ell, nSplits)

    bound = 0;
    
    for currSplit = 1:nSplits
        if mod(currSplit, 100) == 0
            fprintf('\t%d\n', currSplit);
        end
        
        [train_set, test_set] = SplitTask(task, ell / task.nItems);
        
        params = struct('width', sigma);
        model_curr = rvmtrain(train_set, params);
        Y_model = rvmpredict(train_set, model_curr);
        Y_test_model = rvmpredict(test_set, model_curr);
        
        err_train = sum(train_set.target ~= Y_model) / length(Y_model);
        err_test = sum(test_set.target ~= Y_test_model) / length(Y_test_model);
        
        bound = bound + (err_test - err_train);
    end
    
    bound = bound / nSplits;
end
