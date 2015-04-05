function [w, trainError, testError] = trainLogisticRegression(train_set, test_set)
    try
        warning off;
        [w] = glmfit(train_set.objects, train_set.target - 1, ...
            'binomial', 'link', 'logit');
        warning on;
    catch
        w = nan(size(train_set.objects, 2), 1);
        trainError = NaN;
        testError = NaN;
        warning on;               
        return;
    end            

    w = [w(2:end); w(1)];
    w = w ./ sqrt(w' * w);

    train_set = AddConstantFeature(train_set);
    test_set = AddConstantFeature(test_set);

    trainError = mean((2*train_set.target - 3) .* (train_set.objects * w) <= 0);
    testError = mean((2*test_set.target - 3) .* (test_set.objects * w) <= 0);
end
