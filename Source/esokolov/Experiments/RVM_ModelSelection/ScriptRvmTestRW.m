%%
tasks = LoadTasks('./DataUCI');

%%
sigma = 0.5;
nSplits = 10;
fullSearch = 1;

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
    
    X_kernel = SB1_KernelFunction(X, X, 'gauss', sigma);
    X_kernel = [X_kernel, ones(size(X_kernel, 1), 1)];
    
    w = zeros(size(X_kernel, 1), 1);
    w(model_curr.used) = model_curr.weights;
    w(end + 1) = model_curr.bias;
    
%     w = sum(repmat(model_curr.weights, 1, size(X_kernel, 2)) .* ...
%         X_kernel(model_curr.used, :), 1)';
%     w(end + 1) = model_curr.bias;
    
    ell = floor(size(X, 1) / 2);
    nSplits_mc = 1000;
    bounds_all(currSplit) = getMcBoundRw(X_kernel, Y, w, ell, nSplits_mc);
end

%%
save('./esokolov/Experiments/RVM_ModelSelection/mc_sigma_05.mat', ...
    'sigma', 'err_train_all', 'err_test_all', 'bounds_all');