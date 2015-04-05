%%
clear;

%%
L = 200;
ell = 180;
sigma_all = 2 .^ (-2:5);
nStartAlgs = 5;
subsetEll = 190;

%%
% tasks = LoadTasks('./DataUCI');
% [train_set, test_set] = SplitTask(tasks.waveform, L / tasks.waveform.nItems);
% [X, Y] = ConvertTaskToESokolovFormat(train_set);
% [X_test, Y_test] = ConvertTaskToESokolovFormat(test_set);
% 
% X(:, end) = [];
% X_test(:, end) = [];
% 
% save('./esokolov/Experiments/RVM_ModelSelection/splitted_task_waveform.mat', ...
%     'tasks', 'train_set', 'test_set', 'X', 'Y', 'X_test', 'Y_test');

load('./esokolov/Experiments/RVM_ModelSelection/splitted_task_waveform.mat');

%%
err_train_all = zeros(length(sigma_all), 1);
err_test_all = zeros(length(sigma_all), 1);
bounds_all = zeros(length(sigma_all), 1);
parfor sigma_idx = 1:length(sigma_all)
% for sigma_idx = [1, 5]
    fprintf('%d\n', sigma_idx);
    
    sigma_curr = sigma_all(sigma_idx);
    
    params = struct('width', sigma_curr);
    model_curr = rvmtrain(train_set, params);
    Y_model = rvmpredict(train_set, model_curr);
    Y_test_model = rvmpredict(test_set, model_curr);
    
    err_train_all(sigma_idx) = sum(train_set.target ~= Y_model) / length(Y);
    err_test_all(sigma_idx) = sum(test_set.target ~= Y_test_model) / length(Y_test);
    
    X_kernel = SB1_KernelFunction(X, X, 'gauss', sigma_curr);
    X_kernel = X_kernel(:, model_curr.used);
    X_kernel = [X_kernel, ones(size(X_kernel, 1), 1)];
    
    w_total = model_curr.weights;
    w_total(end + 1) = model_curr.bias;
    
    w_total = w_total ./ sqrt(sum(w_total .^ 2));
    
    train_rvm = [];
    train_rvm.nItems = L;
    train_rvm.nFeatures = L;
    train_rvm.nClasses = 2;
    train_rvm.target = train_set.target;
    train_rvm.objects = X_kernel(:, 1:end-1);
    train_rvm.isnominal = false(L, 1);
    
    w_arr = cell(nStartAlgs, 1);
    for currBaseAlg = 1:nStartAlgs
        currSubsetEll = subsetEll - 20 * (currBaseAlg - 1);
        [train_subset, ~, subset_idx] = ...
            SplitTask(train_rvm, currSubsetEll / train_set.nItems);
        [X_subset, Y_subset] = ConvertTaskToESokolovFormat(train_subset);

        params = struct('kernel', 'poly1');
        model_subset = rvmtrain(train_subset, params);
        
        w_curr = model_subset.X(model_subset.used, :)' * model_subset.weights;
        w_curr(end + 1) = model_subset.bias;
        
        w_curr = w_curr ./ sqrt(sum(w_curr .^ 2));
        
        w_arr{currBaseAlg} = w_curr;
    end
    
    w_arr{end + 1} = w_total;
    
    nSplits_mc = 1000;
%     bounds_all(sigma_idx) = getMcBoundRwFs(X_kernel, Y, w_total, w_arr, ell, nSplits_mc);
    bounds_all(sigma_idx) = getCombBoundRwFs(X_kernel, Y, w_total, w_arr, ell);
end

%%
save('./esokolov/Experiments/RVM_ModelSelection/waveform_rvm_model_selection_comb.mat', ...
    'sigma_all', 'err_train_all', 'err_test_all', 'bounds_all');

%%
clc;
for i = 1:length(sigma_all)
    fprintf('sigma = %.6f, Train error = %.4f, Test error = %.4f, Expected delta = %.4f\n', ...
        sigma_all(i), err_train_all(i), err_test_all(i), bounds_all(i));
end

%%
%plot(log2(sigma_all), err_test_all - err_train_all, 'b', 'LineWidth', 2);
plot(log2(sigma_all), bounds_all, 'r', 'LineWidth', 2);
hold on;
plot(log2(sigma_all), err_train_all, 'g', 'LineWidth', 2);
plot(log2(sigma_all), err_test_all, 'm', 'LineWidth', 2);
grid on;
legend('ERM overfitting', 'Train error', 'Test error');
xlabel('log2(sigma)');