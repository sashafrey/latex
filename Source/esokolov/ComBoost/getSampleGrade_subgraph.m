function grade = getSampleGrade_subgraph(X, Y, boundType)
    nItersMCC = 1024;             % monte-carlo cross-validation
    nItersRandomWalk = 4096;      % random walk 
    nAlgsToSample = 4096;
    nRays = 128;                  % number of rays
    randomSeed = 0;               % random seed
    nStartAlgs = 256;        
    linearSamplingDll_path = 'D:\Science\vtf11ccas\Source\Common\GPU';
    nLayers = 15;
    pTransition = 0.8;
    
    % если все объекты сливаютс€ в один объект по данному набору признаков,
    % то сразу же завершаемс€
    uniqueObjects = unique(X, 'rows');
    if size(uniqueObjects, 1) == 1
        grade = Inf;
        return;
    end
    
    L = size(X, 1);
    
    Y(Y == 1) = 2;
    Y(Y == -1) = 1;
    task = struct('nItems', size(X, 1), ...
                  'nFeatures', size(X, 2), ...
                  'nClasses', 2, ...
                  'target', Y, ...
                  'objects', X, ...
                  'isnominal', false(size(X, 2), 1), ...
                  'name', 'noname');

    task = NormalizeFeatureMatrix(task);
    train_set = task;
    
    % обучаем линейный классификатор, прогон€ем через него обучение и контроль
    w = glmfit(train_set.objects, train_set.target-1, 'binomial', 'link', 'logit');
    w = [w(2:end); w(1)];
    w = w ./ sqrt(w'* w);
    
    train_set = addConstantFeature(train_set);
    
    GPU_Initialize(linearSamplingDll_path);
    GPU_SetLogLevel(0);
    randomRays = 2 * (rand(nRays, train_set.nFeatures) - 0.5);
    session_train = GPU_CreateSession(train_set.objects, train_set.target - 1, randomRays);

    [~, ~, trainErrorCount] = GPU_CalcAlgs(session_train,  w');
    trainErrorRate = single(trainErrorCount) / train_set.nItems;
    
    % сэмплируем и считаем оценку
    W0 = ones(nStartAlgs, 1) * w';
    [algs_rw, sources] = GPU_RunRandomWalking(session_train, ...
        W0, nAlgsToSample, nItersRandomWalk, trainErrorCount + nLayers, ...
        pTransition, randomSeed);

    % оцениваем переобучение ћЁ– с помощью папы карло
    [~, ev, ec, hashes] = GPU_CalcAlgs(session_train, algs_rw);
    
    nAlgs = size(ev, 2);
    if nAlgs == 0
        grade = Inf;
        return;
    end
    
    if strcmp(boundType, 'mc')
        [~, ~, trainErrorRate, testErrorRate] = ...
            GPU_PerformCrossValidation(ev, ec, nItersMCC, randomSeed);
        mcBound = mean(testErrorRate);
        grade = mcBound;
    else
        if strcmp(boundType, 'qeps_sources')
            [QEps, eps] = GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 0);
        elseif strcmp(boundType, 'qeps_classic')
            [QEps, eps] = GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 1);
        elseif strcmp(boundType, 'qeps_af')
            clusterIds = GPU_DetectClusters(ev, ec);
            [QEps, eps] = GPU_CalcQEpsCombinatorialAF(ev, ec, hashes, sources, clusterIds);           
        end
        %[QEps, eps] = GPU_CalcQEpsCombinatorial(session_train, algs_rw, sources);
        QEps = sum(QEps);
        medianIdx = find(QEps <= 0.5, 1, 'first');
        if isempty(medianIdx)
            medianIdx = length(QEps);
        end
        invBound = eps(medianIdx);
        grade = (min(double(ec)) / L) + invBound;
    end
    
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

function task = addConstantFeature(task)
    task.objects = [task.objects, ones(task.nItems, 1)];
    task.nFeatures = task.nFeatures + 1;
end
