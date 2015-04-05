%%
clear;

%%
L = 200;
sigma_all = 2 .^ (-16:20);

%%
tasks = LoadTasks('./DataUCI');
[train_set, test_set] = SplitTask(tasks.wine, L / tasks.wine.nItems);
[X, Y] = ConvertTaskToESokolovFormat(train_set);
[X_test, Y_test] = ConvertTaskToESokolovFormat(test_set);

X(:, end) = [];
X_test(:, end) = [];

%%
err_train_all = zeros(length(sigma_all), 1);
err_test_all = zeros(length(sigma_all), 1);
bounds_dd = zeros(length(sigma_all), 1);
bounds_di = zeros(length(sigma_all), 1);
for sigma_idx = 1:length(sigma_all)
    fprintf('%d\n', sigma_idx);
    
    sigma_curr = sigma_all(sigma_idx);
    
    X_kernel = SB1_KernelFunction(X, X, 'gauss', sigma_curr);
%     X_kernel = bsxfun(@times, X_kernel, Y');
    % X_kernel = [X_kernel, ones(size(X_kernel, 1), 1)];
    
    X_test_kernel = SB1_KernelFunction(X_test, X, 'gauss', sigma_curr);
%     X_test_kernel = bsxfun(@times, X_test_kernel, Y');
    
    params = '-t 0 -c 1';
    model_curr = svmtrain2(Y, X_kernel, params);
    Y_model = svmpredict(Y, X_kernel, model_curr);
    Y_test_model = svmpredict(Y_test, X_test_kernel, model_curr);
    
    err_train_all(sigma_idx) = sum(Y ~= Y_model) / length(Y);
    err_test_all(sigma_idx) = sum(Y_test ~= Y_test_model) / length(Y_test);
        
    w_total = model_curr.SVs' * model_curr.sv_coef;
    w_total(end + 1) = -model_curr.rho;
    
    if sum(sign(X_kernel * w_total(1:end-1) + w_total(end)) ~= Y) > L / 2
        w_total = -w_total;
    end
    
    w_total = w_total ./ sqrt(sum(w_total .^ 2));
    
    X_kernel = [X_kernel, ones(size(X_kernel, 1), 1)];
    
    margins = X_kernel * w_total;
    margins = margins ./ sqrt(sum(X_kernel .^ 2, 2));
    margins = margins .* Y;
    
    bounds_dd(sigma_idx) = DDmargin(margins, L);
    bounds_di(sigma_idx) = DImargin(margins);
end

%%
% save('./esokolov/Experiments/RVM_ModelSelection/pacbayes_model_selection.mat', ...
%     'sigma_all', 'err_train_all', 'err_test_all', ...
%     'bounds_dd', 'bounds_di');

%%
clc;
for i = 1:length(sigma_all)
    fprintf('sigma = %.6f, Train error = %.4f, Test error = %.4f, DD = %.4f, DI = %.4f\n', ...
        sigma_all(i), err_train_all(i), err_test_all(i), ...
        bounds_dd(i), bounds_di(i));
end

%%
plot(log2(sigma_all), err_train_all, 'b', 'LineWidth', 2);
hold on;
plot(log2(sigma_all), err_test_all, 'k', 'LineWidth', 2);
plot(log2(sigma_all), bounds_dd, 'r', 'LineWidth', 2);
plot(log2(sigma_all), bounds_di, 'g', 'LineWidth', 2);
grid on;
legend('SVM train', 'SVM test', 'DD', 'DI');
xlabel('log2(sigma)');