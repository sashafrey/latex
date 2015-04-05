%%
clear;

%%
L = 200;
ell = 100;
% sigma_all = 2 .^ (-16:20);
% sigma_all = 2 .^ (-10:10);
sigma_all = 2 .^ (-2:4);
nStartAlgs = 5;
subsetEll = 190;
nSplits_mc = 1000;
iterCnt_rw = 1000;
neighSearchRaysCnt = 100;

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
bounds_mc = zeros(length(sigma_all), 1);
bounds_comb = zeros(length(sigma_all), 1);
bounds_comb_rw = zeros(length(sigma_all), 1);
bounds_asympt_approx = zeros(length(sigma_all), 1);
bounds_asympt = zeros(length(sigma_all), 1);
bounds_worst = zeros(length(sigma_all), 1);

%%
parfor sigma_idx = 1:length(sigma_all)
% for sigma_idx = [1, 5]
    fprintf('%d\n', sigma_idx);
    
    sigma_curr = sigma_all(sigma_idx);
    
    X_kernel = SB1_KernelFunction(X, X, 'gauss', sigma_curr);
%     X_kernel = bsxfun(@times, X_kernel, Y');
    % X_kernel = [X_kernel, ones(size(X_kernel, 1), 1)];
    
    X_test_kernel = SB1_KernelFunction(X_test, X, 'gauss', sigma_curr);
%     X_test_kernel = bsxfun(@times, X_test_kernel, Y');
    
    params = '-t 0 -c 1 -q';
    model_curr = svmtrain2(Y, X_kernel, params);
    Y_model = svmpredict(Y, X_kernel, model_curr, '-q');
    Y_test_model = svmpredict(Y_test, X_test_kernel, model_curr, '-q');
    
    err_train_all(sigma_idx) = sum(Y ~= Y_model) / length(Y);
    err_test_all(sigma_idx) = sum(Y_test ~= Y_test_model) / length(Y_test);
        
    w_total = model_curr.SVs' * model_curr.sv_coef;
    w_total(end + 1) = -model_curr.rho;
    
    if sum(sign(X_kernel * w_total(1:end-1) + w_total(end)) ~= Y) > L / 2
        w_total = -w_total;
    end
    
    w_arr = cell(nStartAlgs, 1);
    for currBaseAlg = 1:nStartAlgs
        subset_idx = randsample(train_set.nItems, subsetEll);
        X_subset = X_kernel(subset_idx, :);
        Y_subset = Y(subset_idx);

        params = '-t 0 -c 1 -q';
        model_subset = svmtrain2(Y_subset, X_subset, params);
        
        subset_idx = find(subset_idx);
        
        w_curr = model_subset.SVs' * model_subset.sv_coef;
        w_curr(end + 1) = -model_subset.rho;
        
        if sum(sign(X_kernel * w_curr(1:end-1) + w_curr(end)) ~= Y) > L / 2
            w_curr = -w_curr;
        end
        
        w_arr{currBaseAlg} = w_curr;
    end
    
    w_arr{end + 1} = w_total;
    
    X_kernel = [X_kernel, ones(size(X_kernel, 1), 1)];
    
    % переходим к представлению линейных классификаторов в виде структур
    alg_start_arr = initLinearAlgSimpleStructure();
    alg_start_arr(length(w_arr)) = initLinearAlgSimpleStructure();
    for i = 1:length(w_arr)
        alg_start_arr(i) = convertLinearAlgToSimpleStructure(w_arr{i}, X_kernel, Y);
    end
    
    maxLevel_rw = 0;
    for i = 1:length(w_arr)
        maxLevel_rw = max(maxLevel_rw, sum(sign(X_kernel * w_arr{i}) ~= Y) + 20);
    end
    
    % сэмплируем
    [algs_rw, corrections_rw] = random_walk_fs(X_kernel, Y, ...
        alg_start_arr, iterCnt_rw, maxLevel_rw, ...
        @(alg, X, Y) getNeighboursLC_rays(alg, X, Y, neighSearchRaysCnt), ...
        false);
    
    sources = findSourcesInSample(algs_rw);
    sourcesVects = getSourcesVects(algs_rw, sources);
    
    bounds_mc(sigma_idx) = getMcBound(algs_rw, L, ell, nSplits_mc);
    
    boundType = 'CCV_classic';
    bounds_comb(sigma_idx) = getCombBound_ManyAlgs(algs_rw, sourcesVects, ...
        L, ell, 0.1, boundType);
    bounds_comb_rw(sigma_idx) = getLayeredBoundEstimate(algs_rw, corrections_rw, ...
        sources, L, ell, 0.1, boundType);
    
    bounds_asympt_approx(sigma_idx) = asymptoticApproxScUpperBound(algs_rw, sources, L, ell);
    
    [bounds_asympt(sigma_idx), bounds_worst(sigma_idx)] = ...
        asymptoticScUpperBound(algs_rw, sources, L, ell);
    
%     bounds_all(sigma_idx) = getMcBoundRwFs(X_kernel, Y, w_total, w_arr, ell, nSplits_mc);
%     bounds_all(sigma_idx) = getCombBoundRwFs(X_kernel, Y, w_total, w_arr, ell);
end

%%
save('./esokolov/Experiments/RVM_ModelSelection/waveform_svm_model_selection_comb.mat', ...
    'sigma_all', 'err_train_all', 'err_test_all', 'bounds_mc', 'bounds_comb', ...
    'bounds_asympt', 'bounds_worst');

% %%
% clc;
% for i = 1:length(sigma_all)
%     fprintf('sigma = %.6f, Train error = %.4f, Test error = %.4f, Expected delta = %.4f\n', ...
%         sigma_all(i), err_train_all(i), err_test_all(i), bounds_all(i));
% end

%%
figure;
hold on;
plot(log2(sigma_all), err_train_all, 'r', 'LineWidth', 2);
plot(log2(sigma_all), err_test_all, 'b', 'LineWidth', 2);
plot(log2(sigma_all), bounds_mc, 'm', 'LineWidth', 2);
plot(log2(sigma_all), bounds_asympt, 'g', 'LineWidth', 2);
plot(log2(sigma_all), bounds_worst, 'c', 'LineWidth', 2);
grid on;
legend('SVM train error', 'SVM test error', 'MC bound', 'Asymptotic bound', ...
    'Worst bound');
xlabel('log2(sigma)');