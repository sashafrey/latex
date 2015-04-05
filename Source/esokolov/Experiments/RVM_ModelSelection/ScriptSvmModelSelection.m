%%
clear;

%%
L = 200;
ell = 100;
sigma_all = 2 .^ (-16:0);
nStartAlgs = 50;
subsetEll = 100;

%%
tasks = LoadTasks('./DataUCI');
[train_set, test_set] = SplitTask(tasks.waveform, L / tasks.waveform.nItems);
[X, Y] = ConvertTaskToESokolovFormat(train_set);
[X_test, Y_test] = ConvertTaskToESokolovFormat(test_set);

X(:, end) = [];
X_test(:, end) = [];

%%
err_train_all = zeros(length(sigma_all), 1);
err_test_all = zeros(length(sigma_all), 1);
bounds_all = zeros(length(sigma_all), 1);
for sigma_idx = 1:length(sigma_all)
    fprintf('%d\n', sigma_idx);
    
    sigma_curr = sigma_all(sigma_idx);
    
    params = sprintf('-t 2 -c 1 -g %f', sigma_curr);
    model_curr = svmtrain2(Y, X, params);
    Y_model = svmpredict(Y, X, model_curr);
    Y_test_model = svmpredict(Y_test, X_test, model_curr);
    
    err_train_all(sigma_idx) = sum(Y ~= Y_model) / length(Y);
    err_test_all(sigma_idx) = sum(Y_test ~= Y_test_model) / length(Y_test);
    
    X_kernel = SB1_KernelFunction(X, X, 'gauss', sigma_curr);
    X_kernel = [X_kernel, ones(size(X_kernel, 1), 1)];
    
    w_total = zeros(size(X_kernel, 1), 1);
    w_total(model_curr.sv_indices) = model_curr.sv_coef;
    w_total(end + 1) = -model_curr.rho;
    
    if sum(sign(X_kernel * w_total) ~= Y) > L / 2
        w_total = -w_total;
    end
    
    w_arr = cell(nStartAlgs, 1);
    for currBaseAlg = 1:nStartAlgs
        [train_subset, ~, subset_idx] = ...
            SplitTask(train_set, subsetEll / train_set.nItems);
        [X_subset, Y_subset] = ConvertTaskToESokolovFormat(train_subset);

        params = sprintf('-t 2 -c 1 -g %f', sigma_curr);
        model_subset = svmtrain2(Y_subset, X_subset, params);
        
        subset_idx = find(subset_idx);
        
        w_curr = zeros(size(X_kernel, 1), 1);
        w_curr(subset_idx(model_subset.sv_indices)) = model_subset.sv_coef;
        w_curr(end + 1) = -model_subset.rho;
        
        if sum(sign(X_kernel * w_curr) ~= Y) > L / 2
            w_curr = -w_curr;
        end
        
        w_arr{currBaseAlg} = w_curr;
    end
    
    w_arr{end + 1} = w_total;
    
    nSplits_mc = 1000;
    bounds_all(sigma_idx) = getMcBoundRwFs(X_kernel, Y, w_total, w_arr, ell, nSplits_mc);
end

%%
% save('./esokolov/Experiments/RVM_ModelSelection/waveform_svm_model_selection_noreg.mat', ...
%     'sigma_all', 'err_train_all', 'err_test_all', 'bounds_all');

%%
clc;
for i = 1:length(sigma_all)
    fprintf('sigma = %.6f, Train error = %.4f, Test error = %.4f, Expected delta = %.4f\n', ...
        sigma_all(i), err_train_all(i), err_test_all(i), bounds_all(i));
end

%%
plot(log2(sigma_all), err_test_all - err_train_all, 'b', 'LineWidth', 2);
hold on;
plot(log2(sigma_all), bounds_all, 'r', 'LineWidth', 2);
grid on;
legend('SVM overfitting', 'ERM overfitting');
xlabel('log2(sigma)');