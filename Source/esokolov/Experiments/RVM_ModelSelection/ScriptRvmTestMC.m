%%
tasks = LoadTasks('./DataUCI');

%%
sigma = 10;
nSplits = 10;

err_train_all = zeros(nSplits, 1);
err_test_all = zeros(nSplits, 1);
bounds_all = zeros(nSplits, 1);

for currSplit = 1:nSplits
    fprintf('%d\n', currSplit);
    
    [train_set, test_set] = SplitTask(tasks.wine, 200 / tasks.wine.nItems);
    [X, Y] = ConvertTaskToESokolovFormat(train_set);
    [X_test, Y_test] = ConvertTaskToESokolovFormat(test_set);

    X(:, end) = [];
    X_test(:, end) = [];
    
    params = struct('width', sigma);
    model_curr = rvmtrain(train_set, params);
    Y_model = rvmpredict(train_set, model_curr);
    Y_test_model = rvmpredict(test_set, model_curr);
    
    err_train_all(currSplit) = sum(train_set.target ~= Y_model) / length(Y);
    err_test_all(currSplit) = sum(test_set.target ~= Y_test_model) / length(Y_test);
    
    ell = floor(size(X, 1) / 2);
    nSplits_mc = 1000;
    bounds_all(currSplit) = getFairMcBoundRVM(train_set, sigma, ell, nSplits_mc);
end

%%
save('./esokolov/Experiments/RVM_ModelSelection/fairmc_sigma_10.mat', ...
    'sigma', 'err_train_all', 'err_test_all', 'bounds_all');

%%
clc;
for i = 1:nSplits
    fprintf('Train error = %.4f, Test error = %.4f, Expected delta = %.4f\n', ...
        err_train_all(i), err_test_all(i), bounds_all(i));
end
