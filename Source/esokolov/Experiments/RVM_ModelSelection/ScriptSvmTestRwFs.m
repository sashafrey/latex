%%
clear

%%
tasks = LoadTasks('./DataUCI');

%%
L = 200;
sigma = 0.5;
nSplits = 10;
nStartAlgs = 2000;
subsetEll = 100;
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
    
    params = sprintf('-t 2 -c 1 -g %f', sigma);
    model_curr = svmtrain2(Y, X, params);
    Y_model = svmpredict(Y, X, model_curr);
    Y_test_model = svmpredict(Y_test, X_test, model_curr);
    
    err_train_all(currSplit) = sum(Y ~= Y_model) / length(Y);
    err_test_all(currSplit) = sum(Y_test ~= Y_test_model) / length(Y_test);
    
    X_kernel = SB1_KernelFunction(X, X, 'gauss', sigma);
    X_kernel = [X_kernel, ones(size(X_kernel, 1), 1)];
    
    w_total = zeros(size(X_kernel, 1), 1);
    w_total(model_curr.sv_indices) = model_curr.sv_coef;
    w_total(end + 1) = -model_curr.rho;
    
    w_arr = cell(nStartAlgs, 1);
    for currBaseAlg = 1:nStartAlgs
        [train_subset, ~, subset_idx] = ...
            SplitTask(train_set, subsetEll / train_set.nItems);
        [X_subset, Y_subset] = ConvertTaskToESokolovFormat(train_subset);

        params = sprintf('-t 2 -c 1 -g %f', sigma);
        model_subset = svmtrain2(Y_subset, X_subset, params);
        
        subset_idx = find(subset_idx);
        
        w_curr = zeros(size(X_kernel, 1), 1);
        w_curr(subset_idx(model_subset.sv_indices)) = model_subset.sv_coef;
        w_curr(end + 1) = -model_subset.rho;
        
%         % почему-то иногда надо брать с обратным знаком
%         if sum(sign(X_kernel * w_curr) ~= Y) > L / 2
%             w_curr = -w_curr;
%         end
        
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
