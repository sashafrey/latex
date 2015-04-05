%%
clear

%%
tasks = LoadTasks('./DataUCI');

%%
L = 10;
sigma = 0.5;
nSplits = 10;
nStartAlgs = 10;
subsetEll = 5;
fullSearch = 0;

err_train_all = zeros(nSplits, 1);
err_test_all = zeros(nSplits, 1);
bounds_all = zeros(nSplits, 1);

for currSplit = 1:nSplits
    fprintf('%d\n', currSplit);
    
    [train_set, test_set] = SplitTask(tasks.wine, L / tasks.wine.nItems);
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
    
    X_kernel = SB1_KernelFunction(X, X, 'gauss', sigma);
    X_kernel = [X_kernel, ones(size(X_kernel, 1), 1)];
    
    w_total = zeros(size(X_kernel, 1), 1);
    w_total(model_curr.used) = model_curr.weights;
    w_total(end + 1) = model_curr.bias;
    
    w_arr = cell(nStartAlgs, 1);
    for currBaseAlg = 1:nStartAlgs
        [train_subset, ~, subset_idx] = ...
            SplitTask(train_set, subsetEll / train_set.nItems);
        params = struct('width', sigma);
        model_subset = rvmtrain(train_subset, params);
        
        subset_idx = find(subset_idx);
        
        w_curr = zeros(size(X_kernel, 1), 1);
        w_curr(subset_idx(model_subset.used)) = model_subset.weights;
        w_curr(end + 1) = model_subset.bias;
        
        w_arr{currBaseAlg} = w_curr;
    end
    
    w_arr{end + 1} = w_total;
    
    ell = floor(size(X, 1) / 2);
    nSplits_mc = 1000;
    bounds_all(currSplit) = getMcBoundRwFs(X_kernel, Y, w_total, w_arr, ell, nSplits_mc);
end

%%
save('./esokolov/Experiments/RVM_ModelSelection/mc_fs_sigma_05.mat', ...
    'sigma', 'err_train_all', 'err_test_all', 'bounds_all');

%%
clc;
for i = 1:nSplits
    fprintf('Train error = %.4f, Test error = %.4f, Expected delta = %.4f\n', ...
        err_train_all(i), err_test_all(i), bounds_all(i));
end
