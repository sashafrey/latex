function CalcFeatureSelection()
    tasks = LoadTasks();

    params.nItersMCC = 4*1024;             % monte-carlo cross-validation
    params.nItersRandomWalk = 4*1024;      % random walk 
    params.nAlgsToSample = 4*1024;
    params.nRays = 256;                  % number of rays
    params.randomSeed = 0;               % random seed
    params.nStartAlgs = 32;        
    params.linearSamplingDll_path = '\\alfrey-h01\vft11ccas\source\Common\GPU';
    params.nLayers = 15;
    params.pTransition = 0.8;
    params.ratio = 0.5;
    params.maxTrainItems = 102400;

    params.nRepeats = 16;
    params.tasknames = {
        %'Sonar'    
        'glass'
        'Liver_Disorders'
        'Ionosphere'    
        'Wdbc'
        'Australian'    
        'pima'
        'faults'
        'statlog'
        %'wine'
        %'waveform'
        %'pageblocks'
        %'Optdigits'
        %'pendigits'
        %'Magic04'    
        %^'Letter'
        };

    nTasks = length(params.tasknames);
    params.nFeatures = zeros(nTasks, 1);
    for i = 1:nTasks
        taskname = params.tasknames{i};
        tasks.(taskname) = crossFeatures(tasks.(taskname));
        params.nFeatures(i) = tasks.(taskname).nFeatures;
    end
    
    if (matlabpool('size') == 0)
        matlabpool;
    end

    parfor iRepeat = 1:params.nRepeats
    %for iRepeat = 1:params.nRepeats
        results = CellVectorCreate();

        for iTask=1:length(params.tasknames)
            taskname=params.tasknames{iTask};
            fprintf('%s-%i ', taskname, iRepeat);
            task = tasks.(taskname);
            Check(all(unique(task.target) == [1; 2]));

            % формируем выборку
            task = NormalizeFeatures(task, false, 2);
            
            nTrainItems = params.ratio * task.nItems;
            if (nTrainItems > params.maxTrainItems)
                ratio = params.maxTrainItems / task.nItems;
            else
                ratio = params.ratio;
            end
            
            [train_setFull, test_setFull] = SplitTask(task, ratio);
            corrValues = zeros(train_setFull.nFeatures, 1);
            for iFeature=1:train_setFull.nFeatures
                corrMatrix = abs(corrcoef(train_setFull.objects(:, iFeature), train_setFull.target));
                corrValues(iFeature) = corrMatrix(1, 2);
            end
            [~, featureIndex] = sort(corrValues, 1, 'descend');
            
            for iFeatures = 1:task.nFeatures
                result = [];
                result.taskname = taskname;
                result.nItems = task.nItems;
                result.nFeatures = task.nFeatures;
                result.iFeatures = iFeatures;
                result.iRepeat = iRepeat;
                result.failed = false;
                result.exception = [];
                result.startTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');
                result.trainError = NaN;
                result.testError = NaN;
                result.trainErrorMCC = NaN;
                result.testErrorMCC = NaN;
                result.tuneTime = NaN;
                result.walkingTime = NaN;
                result.performCVTime = NaN;                
                
                train_set = SelectFeatures(train_setFull, featureIndex(1:iFeatures));
                test_set = SelectFeatures(test_setFull, featureIndex(1:iFeatures));

                try
                    try
                        tic
                        warning off
                        [w] = glmfit(train_set.objects, train_set.target - 1, 'binomial', 'link', 'logit');
                        warning on
                        result.tuneTime = toc;
                    catch ex
                        warning on                
                        result.tuneTime = toc;
                        rethrow(ex)
                    end            

                    w = [w(2:end); w(1)];
                    w = w ./ sqrt(w'* w);

                    train_set = AddConstantFeature(train_set);
                    test_set = AddConstantFeature(test_set);

                    result.trainError = mean((2 * train_set.target-3) .* (train_set.objects * w) <= 0);
                    result.testError = mean((2 * test_set.target-3) .* (test_set.objects * w) <= 0);

                     %% Combinatorial overfitting
                     GPU_Initialize(params.linearSamplingDll_path);
                     if (true) 
                         randomRays = 2 * (rand(params.nRays, train_set.nFeatures) - 0.5);
                         session_train = GPU_CreateSession(train_set.objects, train_set.target - 1, randomRays);

                         [~, ~, trainErrorCount, ~] = GPU_CalcAlgs(session_train,  w');

                         % сэмплируем и считаем оценку
                         W0 = ones(params.nStartAlgs, 1) * w';
                         tic
                         [algs_rw, sources] = GPU_RunRandomWalking(session_train, ...
                             W0, params.nAlgsToSample, params.nItersRandomWalk, trainErrorCount + params.nLayers, ...
                             params.pTransition, params.randomSeed);
                         result.walkingTime = toc;

                         % оцениваем переобучение МЭР с помощью папы карло
                         [~, ev, ec, ~] = GPU_CalcAlgs(session_train, algs_rw);
                         tic
                         [~, ~, trainErrorMCCOneIter, testErrorRateMCCOneIter] = GPU_PerformCrossValidation(ev, ec, params.nItersMCC, params.randomSeed);
                         result.performCVTime = toc;

                         result.trainErrorMCC = mean(trainErrorMCCOneIter);
                         result.testErrorMCC = mean(testErrorRateMCCOneIter);
                     end
                catch ex 
                    result.failed = true;
                    result.exception = ex;
                end

                GPU_CloseAllSessions();            

                results = CellVectorAdd(results, result);
                parsave(sprintf('results%i', iRepeat), results, params);
                fprintf('.');
            end

            fprintf(' done.\n');
        end
    end
end

function parsave(fname, results, params)
    save(fname, 'results', 'params')
end

function task = crossFeatures(task) 
    nFeatures = task.nFeatures;
    newNFeatures = nFeatures * (nFeatures + 1) / 2;
    newObjects = zeros(task.nItems, newNFeatures);
    ind = 1;
    for i = 1:nFeatures
        for j=i:nFeatures
            newObjects(:, ind) = task.objects(:, i) .* task.objects(:, j);
            ind = ind + 1;
        end
    end
    
    task.nFeatures = newNFeatures;
    task.objects = newObjects;
    task.isnominal = false(newNFeatures, 1);
end