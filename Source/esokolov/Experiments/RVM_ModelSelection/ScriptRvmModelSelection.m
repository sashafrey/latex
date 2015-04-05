%%
tasks = LoadTasks('./DataUCI');
[train_set, test_set] = SplitTask(tasks.wine, 200 / tasks.wine.nItems);
[X, Y] = ConvertTaskToESokolovFormat(train_set);
[X_test, Y_test] = ConvertTaskToESokolovFormat(test_set);

X(:, end) = [];
X_test(:, end) = [];

%%
sigma_all = 2 .^ (-10:7);
err_train_all = zeros(length(sigma_all), 1);
err_test_all = zeros(length(sigma_all), 1);
bounds_all = zeros(length(sigma_all), 1);
for sigma_idx = 14:length(sigma_all)
    fprintf('%d\n', sigma_idx);
    
    sigma_curr = sigma_all(sigma_idx);
    
    params = struct('width', sigma_curr);
    model_curr = rvmtrain(train_set, params);
    Y_model = rvmpredict(train_set, model_curr);
    Y_test_model = rvmpredict(test_set, model_curr);
    
    err_train_all(sigma_idx) = sum(train_set.target ~= Y_model) / length(Y);
    err_test_all(sigma_idx) = sum(test_set.target ~= Y_test_model) / length(Y_test);
    
    X_kernel = SB1_KernelFunction(X, X, 'gauss', sigma_curr);
    X_kernel = [X_kernel, ones(size(X_kernel, 1), 1)];
    
    w = zeros(size(X_kernel, 1), 1);
    w(model_curr.used) = model_curr.weights;
    w(end + 1) = model_curr.bias;
    
%     w = sum(repmat(model_curr.weights, 1, size(X_kernel, 2)) .* ...
%         X_kernel(model_curr.used, :), 1)';
%     w(end + 1) = model_curr.bias;
    
    ell = floor(size(X, 1) / 1);
    eps = 0.1;
    boundType = 'SC_sources';
    bounds_all(sigma_idx) = getBoundRw(X_kernel, Y, w, ell, eps, boundType);
end
