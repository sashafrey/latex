%%
clear;

%%
nStartAlgs = 10;
L = 200;
nSplits_mc = 1000;
iterCnt_rw = 1000;
subsetEll = 190;
neighSearchRaysCnt = 100;

noiseFeaturesCnt = 0:20:100;
% noiseFeaturesCnt = [0, 30];

%%
tasks = LoadTasks('./DataUCI');
task_curr = tasks.waveform;

d = task_curr.nFeatures;

%%
[train_set, test_set] = SplitTask(task_curr, L / task_curr.nItems);
[X_base, Y] = ConvertTaskToESokolovFormat(train_set);
[X_test_base, Y_test] = ConvertTaskToESokolovFormat(test_set);
X_base(:, end) = [];
X_test_base(:, end) = [];

%%
err_train_all = zeros(length(noiseFeaturesCnt), 1);
err_test_all = zeros(length(noiseFeaturesCnt), 1);
bounds_mc = zeros(length(noiseFeaturesCnt), 1);
bounds_ccv = zeros(length(noiseFeaturesCnt), 1);
bounds_ccv_rw = zeros(length(noiseFeaturesCnt), 1);
bounds_qeps = zeros(length(noiseFeaturesCnt), 1);
bounds_qeps_rw = zeros(length(noiseFeaturesCnt), 1);
bounds_asympt_approx = zeros(length(noiseFeaturesCnt), 1);
bounds_asympt = zeros(length(noiseFeaturesCnt), 1);
bounds_worst = zeros(length(noiseFeaturesCnt), 1);
bounds_pacbayes = zeros(length(noiseFeaturesCnt), 1);

%%
algs_all = cell(length(noiseFeaturesCnt), 1);
X_all = cell(length(noiseFeaturesCnt), 1);
X_test_all = cell(length(noiseFeaturesCnt), 1);

parfor noiseCntIdx = 1:length(noiseFeaturesCnt)
    fprintf('%d\n', noiseCntIdx);
    
    currNoiseCnt = noiseFeaturesCnt(noiseCntIdx);
    
    % добавляем шум
    X = [X_base, rand(L, currNoiseCnt)];
    X_test = [X_test_base, rand(task_curr.nItems - L, currNoiseCnt)];
    p = randperm(d + currNoiseCnt);
    X = X(:, p);
    X_test = X_test(:, p);
    
    d_full = d + currNoiseCnt;
    
    % обучаем линейный классификатор, прогоняем через него обучение и контроль
    params = '-t 0 -c 100 -q';
    model_curr = svmtrain2(Y, X, params);
    Y_model = svmpredict(Y, X, model_curr, '-q');
    Y_test_model = svmpredict(Y_test, X_test, model_curr, '-q');
    
    % находим ошибку
    err_train_all(noiseCntIdx) = sum(Y ~= Y_model) / length(Y);
    err_test_all(noiseCntIdx) = sum(Y_test ~= Y_test_model) / length(Y_test);
    
    % восстанавливаем вектор весов
    w = zeros(d_full + 1, 1);
    w(1:d_full) = model_curr.SVs' * model_curr.sv_coef;
    w(end) = -model_curr.rho;
    
    w_arr = cell(nStartAlgs + 1, 1);
    w_arr{1} = w;
    for currBaseAlg = 1:nStartAlgs
        subset_idx = randsample(L, subsetEll);
        X_subset = X(subset_idx, :);
        Y_subset = Y(subset_idx);

        params = '-t 0 -c 100 -q';
        model_subset = svmtrain2(Y_subset, X_subset, params);
        
        w_curr = zeros(d_full + 1, 1);
        w_curr(1:d_full) = model_subset.SVs' * model_subset.sv_coef;
        w_curr(end) = -model_subset.rho;
        
        if sum(sign(X * w_curr(1:end-1) + w_curr(end)) ~= Y) > L / 2
            w_curr = -w_curr;
        end
        
        w_arr{currBaseAlg + 1} = w_curr;
    end
    
    X = [X, ones(size(X, 1), 1)];
    
    % переходим к представлению линейных классификаторов в виде структур
    alg_start_arr = initLinearAlgSimpleStructure();
    alg_start_arr(length(w_arr)) = initLinearAlgSimpleStructure();
    for i = 1:length(w_arr)
        alg_start_arr(i) = convertLinearAlgToSimpleStructure(w_arr{i}, X, Y);
    end
    
    maxLevel_rw = 0;
    for i = 1:length(w_arr)
        maxLevel_rw = max(maxLevel_rw, sum(sign(X * w_arr{i}) ~= Y) + 20);
    end
    
    % сэмплируем
    [algs_rw, corrections_rw] = random_walk_fs(X, Y, ...
        alg_start_arr, iterCnt_rw, maxLevel_rw, ...
        @(alg, X, Y) getNeighboursLC_rays(alg, X, Y, neighSearchRaysCnt), ...
        false);
    
    X_all{noiseCntIdx} = X;
    X_test_all{noiseCntIdx} = X_test;
    algs_all{noiseCntIdx} = algs_rw;
    
    sources = findSourcesInSample(algs_rw);
    sourcesVects = getSourcesVects(algs_rw, sources);
    
    ell = floor(L / 2);
    
    bounds_mc(noiseCntIdx) = getMcBound(algs_rw, L, ell, nSplits_mc);
    
    boundType = 'CCV_classic';
    bounds_ccv(noiseCntIdx) = getCombBound_ManyAlgs(algs_rw, sourcesVects, ...
        L, ell, 0.1, boundType);
    bounds_ccv_rw(noiseCntIdx) = getLayeredBoundEstimate(algs_rw, corrections_rw, ...
        sources, L, ell, 0.1, boundType);
    
%     boundType = 'SC_sources';
%     bounds_qeps(noiseCntIdx) = invertBound(@(eps_arg) getCombBound_ManyAlgs(algs_rw, ...
%         sourcesVects, L, ell, eps_arg, boundType), ...
%         0.5);
%     bounds_qeps_rw(noiseCntIdx) = invertBound(@(eps_arg) getLayeredBoundEstimate(algs_rw, ...
%         corrections_rw, sources, L, ell, eps_arg, boundType), ...
%         0.5);
    
    bounds_asympt_approx(noiseCntIdx) = asymptoticApproxScUpperBound(algs_rw, sources, L, ell);

    [bounds_asympt(noiseCntIdx), bounds_worst(noiseCntIdx)] = ...
        asymptoticScUpperBound(algs_rw, sources, L, ell);
    
    margins = X * w_arr{1};
    margins = margins ./ sqrt(sum(w .^ 2));
    margins = margins ./ sqrt(sum(X .^ 2, 2));
    margins = margins .* Y;
    
    bounds_pacbayes(noiseCntIdx) = DDmargin(margins, L);
    
    % bounds_all(noiseCntIdx) = getFairMcBound_LinearSVM(train_set, ell, nSplits_mc);
    
%     bounds_all(noiseCntIdx) = getMcBoundRwFs(X, Y, w_arr, ell, ...
%         iterCnt_rw, maxLevel_rw, nSplits_mc);
end

%%
save('./esokolov/Experiments/SVM_noise/waveform_svm_noise_mid.mat', ...
    'noiseFeaturesCnt', 'err_train_all', 'err_test_all', 'bounds_mc', 'bounds_ccv', ...
    'bounds_ccv_rw', 'bounds_qeps', 'bounds_qeps_rw', 'bounds_asympt', ...
    'bounds_worst', 'bounds_pacbayes');

% %%
% clc;
% for i = 1:length(L_all)
%     fprintf('L = %d, Train error = %.4f, Test error = %.4f, Predicted test error = %.4f\n', ...
%         L_all(i), err_train_all(i), err_test_all(i), bounds_all(i));
% end

%%
figure;
plot(noiseFeaturesCnt, err_train_all, 'r', 'LineWidth', 2);
hold on;
plot(noiseFeaturesCnt, err_test_all, 'b', 'LineWidth', 2);
plot(noiseFeaturesCnt, bounds_mc, 'm', 'LineWidth', 2);
plot(noiseFeaturesCnt, bounds_ccv, 'k', 'LineWidth', 2);
plot(noiseFeaturesCnt, bounds_asympt, 'g', 'LineWidth', 2);
%plot(noiseFeaturesCnt, bounds_worst, 'c', 'LineWidth', 2);
plot(noiseFeaturesCnt, bounds_pacbayes, 'c', 'LineWidth', 2);
grid on;
legend('SVM train error', 'SVM test error', 'MC bound', 'CCV', 'Asymptotic bound', ...
    'PAC-Bayes');
xlabel('Number of noise features');
