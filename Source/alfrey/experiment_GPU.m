function experiment_GPU(task)
    if (~exist('task', 'var'))
        task = LoadTask('Liver_Disorders');
    end

    nItersMCC = 1024;             % monte-carlo cross-validation
    nItersRandomWalk = 4096;      % random walk 
    nAlgsToSample = 4096;
    nRays = 128;                  % number of rays
    randomSeed = 0;               % random seed
    nStartAlgs = 256;        
    linearSamplingDll_path = '\\alfrey-h01\vft11ccas\source\Common\GPU';
    nLayers = 15;
    pTransition = 0.8;

    % формируем выборку
    task = NormalizeFeatures(task, false, 2);
    [train_set, test_set] = SplitTask(task, 0.5);    
    
    % обучаем линейный классификатор, прогон€ем через него обучение и контроль
    %[w, trainErrorCountSVMLib] = TuneSVM(train_set.objects, 2 * (train_set.target - 1.5));
   
    tic
    w = glmfit(train_set.objects, train_set.target-1, 'binomial', 'link', 'logit');
    tuneTime=toc;
    w = [w(2:end); w(1)];
    w = w ./ sqrt(w'* w);
    
    train_set = AddConstantFeature(train_set);
    test_set = AddConstantFeature(test_set);
    
    GPU_Initialize(linearSamplingDll_path);
    randomRays = 2 * (rand(nRays, train_set.nFeatures) - 0.5);
    session_train = GPU_CreateSession(train_set.objects, train_set.target - 1, randomRays);
    session_test = GPU_CreateSession(test_set.objects, test_set.target - 1, randomRays);

    [~, ~, trainErrorCount, ~] = GPU_CalcAlgs(session_train,  w');
    trainErrorRate = single(trainErrorCount) / train_set.nItems;
    %Check(trainErrorCount == trainErrorCountSVMLib); % SVMLib is consistent with GPU linear sampling.
    
    [~, ~, ecTestW, ~] = GPU_CalcAlgs(session_test,  w');
    testErrorRate = single(ecTestW) / test_set.nItems;
    
    % сэмплируем и считаем оценку
    W0 = ones(nStartAlgs, 1) * w';
    tic
    [algs_rw, sources] = GPU_RunRandomWalking(session_train, ...
        W0, nAlgsToSample, nItersRandomWalk, trainErrorCount + nLayers, ...
        pTransition, randomSeed);
    walkingTime=toc;
    if (size(algs_rw, 1) == 0) fprintf('randomWalkFailed\n'); return; end;

    tic
    [QEps, eps] = GPU_CalcQEpsCombinatorial(session_train, algs_rw, sources);
    calcCombBouncTime = toc;
    QEps_total = sum(QEps);
    combBounc = min(eps(QEps_total < 0.5));
    %combBounc = 0;

    % оцениваем переобучение ћЁ– с помощью папы карло
    [~, ev, ec, ~] = GPU_CalcAlgs(session_train, algs_rw);
    tic
    [~, ~, trainErrorRateMCC, testErrorRateMCC] = GPU_PerformCrossValidation(ev, ec, nItersMCC, randomSeed);
    performCVTime = toc;
    mcBound = mean(testErrorRateMCC - trainErrorRateMCC);
    
    %[~, ~, ~, ~, allOverfittings]  = CalcOverfitting(AlgsetCreate(ev'==1), 0.5, nItersMCC/10, 1, 0.1, 0.01);
    %mcBound2 = mean(allOverfittings);
    mcBound2 = 0;
    
    %PAC-Bayes bounds
    gamma = (2*train_set.target - 3) .* (train_set.objects * w);
    for i=1:train_set.nItems
        vec = task.objects(i, :);
        gamma(i) = gamma(i) / sqrt(vec * vec');
    end
    plot(sort(gamma))
    tic
    dibound = DImargin(gamma);
    ddbound = DDmargin(gamma, train_set.nFeatures);
    pacBayesTime = toc;
    
    fid = fopen('C:\results_xx1.txt', 'a');
    fprintf(fid, 'Task = %s, nFeatues = %i, nItems = %i, nSources = %i, trainErrorRate = %.5f, testErrorRate = %.5f, combBounc = %.5f, mcBound = %.5f, mcBound2 = %.5f, dibound = %.5f, ddbound = %.5f, timeTune = %.5f, timeWalking = %.5f, timeCombBound = %.5f, timeCV = %.5f, timePAC = %.5f\n', ...
        task.name, task.nFeatures, task.nItems, sum(sources), trainErrorRate, testErrorRate, combBounc, mcBound, mcBound2, dibound, ddbound,tuneTime,walkingTime,calcCombBouncTime,performCVTime,pacBayesTime);        
    fprintf('Task = %s, nFeatues = %i, nItems = %i, nSources = %i, trainErrorRate = %.5f, testErrorRate = %.5f, combBounc = %.5f, mcBound = %.5f, mcBound2 = %.5f, dibound = %.5f, ddbound = %.5f, timeTune = %.5f, timeWalking = %.5f, timeCombBound = %.5f, timeCV = %.5f, timePAC = %.5f\n', ...
        task.name, task.nFeatures, task.nItems, sum(sources), trainErrorRate, testErrorRate, combBounc, mcBound, mcBound2, dibound, ddbound,tuneTime,walkingTime,calcCombBouncTime,performCVTime,pacBayesTime);        
    fclose(fid);

    GPU_CloseAllSessions();
end

function task = NormalizeFeatureMatrix(task)
    for i=1:task.nFeatures
        v = task.objects(:, i);
        vmin = min(v);
        vmax = max(v);
        if ((vmax - vmin) > 1e-3)
            v = (v - vmin) / (vmax - vmin);
        end
        v = v - mean(v);
        task.objects(:, i) = v;    
    end
end

function [w, nErrors] = TuneSVM(X, Y)
    d = size(X, 2);
    params = '-t 0 -c 0.1 -q';
    model_curr = svmtrain2(Y, X, params);
    
    % восстанавливаем вектор весов
    w = zeros(d + 1, 1);
    w(1:d) = model_curr.SVs' * model_curr.sv_coef;
    w(end) = -model_curr.rho;
    
    if (model_curr.Label(1) < 0) 
        w = -w;
    end  
    
    Y_model = svmpredict(Y, X, model_curr, '-q');
    nErrors = sum(Y ~= Y_model);
end