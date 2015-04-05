%%
clear;

%%
nStartAlgs = 10;
% L_all = [10:10:100, 150:50:300, 400:100:1000];
L_all = [10:10:100, 150:50:300, 400:100:800];
nSplits_mc = 1000;
iterCnt_rw = 1000;
neighSearchRaysCnt = 50;
innerItersCnt = 10;

%%
tasks = LoadTasks('./DataUCI');
task_curr = tasks.waveform;

d = task_curr.nFeatures;

%%
if max(L_all) > task_curr.nItems
    error('L is too large');
end

%%
err_train_all = zeros(length(L_all), 1);
err_test_all = zeros(length(L_all), 1);
bounds_mc = zeros(length(L_all), 1);
bounds_ccv = zeros(length(L_all), 1);
bounds_ccv_rw = zeros(length(L_all), 1);
bounds_qeps = zeros(length(L_all), 1);
bounds_qeps_rw = zeros(length(L_all), 1);
bounds_asympt_approx = zeros(length(L_all), 1);
bounds_asympt = zeros(length(L_all), 1);
bounds_worst = zeros(length(L_all), 1);
bounds_asympt_test = zeros(length(L_all), 1);
bounds_asympt_distr = zeros(length(L_all), 1);
bounds_pacbayes = zeros(length(L_all), 1);

%%
parfor L_idx = 1:length(L_all)
    fprintf('%d\n', L_idx);
    
    L = L_all(L_idx);
    
    % формируем выборку
    [train_set, test_set] = SplitTask(task_curr, L / task_curr.nItems);
    [X, Y] = ConvertTaskToESokolovFormat(train_set);
    [X_test, Y_test] = ConvertTaskToESokolovFormat(test_set);
    X(:, end) = [];
    X_test(:, end) = [];
    
    % обучаем линейный классификатор, прогоняем через него обучение и контроль
    params = '-t 0 -c 1 -q';
    model_curr = svmtrain2(Y, X, params);
    Y_model = svmpredict(Y, X, model_curr, '-q');
    Y_test_model = svmpredict(Y_test, X_test, model_curr, '-q');
    
    % находим ошибку
    err_train_all(L_idx) = sum(Y ~= Y_model) / length(Y);
    err_test_all(L_idx) = sum(Y_test ~= Y_test_model) / length(Y_test);
    
    % восстанавливаем вектор весов
    w = zeros(d + 1, 1);
    w(1:d) = model_curr.SVs' * model_curr.sv_coef;
    w(end) = -model_curr.rho;
    
    for currInnerIter = 1:innerItersCnt
        w_arr = cell(nStartAlgs + 1, 1);
        w_arr{1} = w;
        for currBaseAlg = 1:nStartAlgs
            subsetEll = ceil(L * 0.95);
            subset_idx = randsample(train_set.nItems, subsetEll);
            X_subset = X(subset_idx, :);
            Y_subset = Y(subset_idx);

            params = '-t 0 -c 1 -q';
            model_subset = svmtrain2(Y_subset, X_subset, params);

            w_curr = zeros(d + 1, 1);
            w_curr(1:d) = model_subset.SVs' * model_subset.sv_coef;
            w_curr(end) = -model_subset.rho;

            if sum(sign(X * w_curr(1:end-1) + w_curr(end)) ~= Y) > L / 2
                w_curr = -w_curr;
            end

            w_arr{currBaseAlg + 1} = w_curr;
        end

        X_withBias = [X, ones(size(X, 1), 1)];

        % переходим к представлению линейных классификаторов в виде структур
        alg_start_arr = initLinearAlgSimpleStructure();
        alg_start_arr(length(w_arr)) = initLinearAlgSimpleStructure();
        for i = 1:length(w_arr)
            alg_start_arr(i) = convertLinearAlgToSimpleStructure(w_arr{i}, X_withBias, Y);
        end

        maxLevel_rw = 0;
        for i = 1:length(w_arr)
            maxLevel_rw = max(maxLevel_rw, sum(sign(X_withBias * w_arr{i}) ~= Y) + 20);
        end

        % сэмплируем
        [algs_rw, corrections_rw] = random_walk_fs(X_withBias, Y, ...
            alg_start_arr, iterCnt_rw, maxLevel_rw, ...
            @(alg, X, Y) getNeighboursLC_rays(alg, X, Y, neighSearchRaysCnt), ...
            false);

        sources = findSourcesInSample(algs_rw);
        sourcesVects = getSourcesVects(algs_rw, sources);

        ell = floor(L / 2);

        bounds_mc(L_idx) = bounds_mc(L_idx)+ getMcBound(algs_rw, L, ell, nSplits_mc);

        boundType = 'CCV_classic';
        bounds_ccv(L_idx) = bounds_ccv(L_idx)+ ...
            getCombBound_ManyAlgs(algs_rw, sourcesVects, ...
            L, ell, 0.1, boundType);
        bounds_ccv_rw(L_idx) = bounds_ccv_rw(L_idx) + ...
            getLayeredBoundEstimate(algs_rw, corrections_rw, ...
            sources, L, ell, 0.1, boundType);

    %     boundType = 'SC_sources';
    %     bounds_qeps(L_idx) = invertBound(@(eps_arg) getCombBound_ManyAlgs(algs_rw, ...
    %         sourcesVects, L, ell, eps_arg, boundType), ...
    %         0.5);
    %     bounds_qeps_rw(L_idx) = invertBound(@(eps_arg) getLayeredBoundEstimate(algs_rw, ...
    %         corrections_rw, sources, L, ell, eps_arg, boundType), ...
    %         0.5);

    %     bounds_asympt_approx(L_idx) = asymptoticApproxScUpperBound(algs_rw, sources, L, ell);

        [r1, r2] = ...
            asymptoticScUpperBound(algs_rw, sources, L, ell);
        bounds_asympt(L_idx) = bounds_asympt(L_idx) + err_train_all(L_idx) + r1;
        bounds_worst(L_idx) = bounds_worst(L_idx) + err_train_all(L_idx) + r2;

        bounds_asympt_distr(L_idx) = bounds_asympt_distr(L_idx) + ...
            err_train_all(L_idx) + ...
            asymptoticDistrUpperBound(algs_rw, sources, L, ell);

    %     bounds_asympt_test(L_idx) = ...
    %         asymptoticTestScUpperBound(algs_rw, sources, L, ell);

        % bounds_all(L_idx) = getFairMcBound_LinearSVM(train_set, ell, nSplits_mc);

    %     bounds_all(L_idx) = getMcBoundRwFs(X, Y, w_arr, ell, ...
    %         iterCnt_rw, maxLevel_rw, nSplits_mc);
    end
    
    bounds_mc(L_idx) = bounds_mc(L_idx) / innerItersCnt;
    bounds_ccv(L_idx) = bounds_ccv(L_idx) / innerItersCnt;
    bounds_ccv_rw(L_idx) = bounds_ccv_rw(L_idx) / innerItersCnt;
    bounds_asympt(L_idx) = bounds_asympt(L_idx) / innerItersCnt;
    bounds_worst(L_idx) = bounds_worst(L_idx) / innerItersCnt;
    bounds_asympt_distr(L_idx) = bounds_asympt_distr(L_idx) / innerItersCnt;

    margins = X_withBias * w_arr{1};
    margins = margins ./ sqrt(sum(w .^ 2));
    margins = margins ./ sqrt(sum(X_withBias .^ 2, 2));
    margins = margins .* Y;
    
    bounds_pacbayes(L_idx) = DDmargin(margins, L);
end

%%
save('./esokolov/Experiments/SVM_TrainSize/waveform_svm_trainSize.mat', ...
    'L_all', 'err_train_all', 'err_test_all', 'bounds_mc', 'bounds_ccv', ...
    'bounds_ccv_rw', 'bounds_qeps', 'bounds_qeps_rw', 'bounds_asympt', ...
    'bounds_worst', 'bounds_asympt_test', 'bounds_asympt_distr');

% %%
% clc;
% for i = 1:length(L_all)
%     fprintf('L = %d, Train error = %.4f, Test error = %.4f, Predicted test error = %.4f\n', ...
%         L_all(i), err_train_all(i), err_test_all(i), bounds_all(i));
% end

%%
figure;
plot(L_all, err_train_all, 'r', 'LineWidth', 2);
hold on;
plot(L_all, err_test_all, 'b', 'LineWidth', 2);
plot(L_all, bounds_mc, 'm', 'LineWidth', 2);
% plot(L_all, bounds_ccv, 'k', 'LineWidth', 2);
plot(L_all, bounds_asympt, 'g', 'LineWidth', 2);
plot(L_all, bounds_worst, 'c', 'LineWidth', 2);
plot(L_all, bounds_asympt_distr, 'k', 'LineWidth', 2);
plot(L_all, bounds_pacbayes, 'y', 'LineWidth', 2);
grid on;
legend('SVM train error', 'SVM test error', 'MC bound', 'Asymptotic bound', ...
    'Worst bound', 'Asymptotic distr bound', 'PAC-Bayes');
xlabel('Train size');
