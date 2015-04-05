function CalcCombAndPacComparisonTable()
    tasks = LoadTasks();

    params.nItersMCC = 2048;           % monte-carlo cross-validation
    params.nItersRandomWalk = 2048;    % random walk 
    params.nAlgsToSample = 2048;
    params.nRays = 1024;                 % number of rays
    params.randomSeed = 0;               % random seed
    params.nStartAlgs = 64;        
    params.linearSamplingDll_path = '\\alfrey-h01\vft11ccas\source\Common\GPU';
    params.outputFileName = 'results';
    params.nLayers = 15;
    params.pTransition = 0.8;
    params.nFolds = 5;
    params.ellRatio = 0.8;

    params.nRepeats = 100;
    params.tasknames = {
        'Liver_Disorders'
        %'Sonar'    
        'glass'
        'Ionosphere'    
        %'Wdbc'
        'Australian'    
        'pima'
        'faults'
        'statlog'
        'wine'
        'waveform'
        'pageblocks'
        'Optdigits'
        %'pendigits'
        %'Letter'
        };
    
    %params.ellRatio = [0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.85 0.88 0.96 0.96 0.963 0.964 0.985 0.993];
    %params.ellRatio =  [0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8  0.8  0.8  0.8  0.8   0.8   0.8   0.8];
    params.ellRatio =  [0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8  0.8  0.8  0.8];
        
%Faults - 0.85
%Statlog - 0.88
%Wine - 0.96
%Waveform - 0.96
%Pageblocks - 0.963
%Optdigits - 0.964
%Pendigits - 0.985
%Letter - 0.993       

    GPU_Initialize(params.linearSamplingDll_path);
    GPU_SetLogLevel(0);
    
    if (matlabpool('size') == 0)
        matlabpool local 4;
    end

    parfor iRepeat = 1:params.nRepeats
    %for iRepeat = 1:params.nRepeats
        results = CellVectorCreate();

        %for iTask=1:length(params.tasknames)
        for iTask=1:length(params.tasknames)
            taskname = params.tasknames{iTask};
            fprintf('%s-%i ', taskname, iRepeat);
            task = tasks.(taskname);
            Check(all(unique(task.target) == [1; 2]));

            % формируем выборку
            task = NormalizeFeatures(task, false, 2);
            
            cvIndexes = GenerateNFoldCVIndexes(task.target, params.nFolds);
            for iFold = 1:params.nFolds
                result = [];
                result.taskname = taskname;
                result.nItems = task.nItems;
                result.nFeatures = task.nFeatures;
                result.iFold = iFold;
                result.cvIndexes = cvIndexes;
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
                
                result.CombEps = [];
                result.CombVC = [];
                result.CombSC = [];
                result.CombES = [];
                result.CombAF = [];
                result.CombVCInverse = NaN;
                result.CombSCInverse = NaN;
                result.CombESInverse = NaN;
                result.CombAFInverse = NaN;
                result.CombVCTime = NaN;
                result.CombSCTime = NaN;
                result.CombESTime = NaN;
                result.CombAFTime = NaN;
                
                result.pacBayesDIbound = NaN;
                result.pacBayesDDbound = NaN;
                result.pacBayesTime = NaN;

                try
                    train_set = GetTaskSubsample(task, cvIndexes ~= iFold);
                    test_set  = GetTaskSubsample(task, cvIndexes == iFold);
                    
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
                    if (size(algs_rw, 1) == 0) 
                        throw(MException('GPU:RunRandomWalking', 'GPU_RunRandomWalking failed'));
                    end;

                    % оцениваем переобучение МЭР с помощью папы карло
                    [~, ev, ec, ~] = GPU_CalcAlgs(session_train, algs_rw);
                    tic
                    [~, ~, trainErrorMCCOneIter, testErrorRateMCCOneIter] = GPU_PerformCrossValidation(ev, ec, params.nItersMCC, params.randomSeed);
                    result.performCVTime = toc;
                    result.trainErrorMCC = mean(trainErrorMCCOneIter);
                    result.testErrorMCC = mean(testErrorRateMCCOneIter);
                    
                    epsStep = min(0.001, 0.05 * 4 / train_set.nItems);
                    result.CombEps = 0 : epsStep : 1; 
                    
                    [~, ev,ec,hashes] = GPU_CalcAlgs(session_train, algs_rw);
                    
                    %GPU_StartRecording('C:\recording.4.dat');
                    
                    tic
                    result.CombVC = sum(GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 2, result.CombEps, int32(params.ellRatio(iTask) * train_set.nItems)));
                    result.CombVCInverse = min(result.CombEps(result.CombVC < 0.5));
                    result.CombVCTime = toc;
                    
                    tic
                    result.CombSC = sum(GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 1, result.CombEps, int32(params.ellRatio(iTask) * train_set.nItems)));
                    result.CombSCInverse = min(result.CombEps(result.CombSC < 0.5));
                    result.CombSCTime = toc;
 
                    tic
                    result.CombES = sum(GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 0, result.CombEps, int32(params.ellRatio(iTask) * train_set.nItems)));
                    result.CombESInverse = min(result.CombEps(result.CombES < 0.5));
                    result.CombESTime = toc;
                    
                    tic
                    clusterIds = GPU_DetectClusters(ev, ec);
                    result.CombAF = sum(GPU_CalcQEpsCombinatorialAF(ev, ec, hashes, sources, clusterIds, result.CombEps, int32(params.ellRatio(iTask) * train_set.nItems)));
                    result.CombAFInverse = min(result.CombEps(result.CombAF < 0.5));
                    result.CombAFTime = toc;
                    
                    % Clear - to expensive to store on disk.
                    result.CombEps = [];
                    result.CombVC = [];
                    result.CombSC = [];
                    result.CombES = [];
                    result.CombAF = [];

                    %GPU_StopRecording();
                    
                    tic
                    gamma = (2*train_set.target - 3) .* (train_set.objects * w);
                    %plot(sort(gamma))
                    for i=1:train_set.nItems
                        vec = task.objects(i, :)';
                        gamma(i) = gamma(i) / sqrt(vec' * vec);
                    end
                    result.pacBayesDIbound = DImargin(gamma);
                    result.pacBayesDDbound = DDmargin(gamma, train_set.nFeatures);
                    result.pacBayesTime = toc;
                    
                catch ex 
                    result.failed = true;
                    result.exception = ex;
                end

                GPU_CloseAllSessions();            

                results = CellVectorAdd(results, result);
                parsave(sprintf('%s%i', params.outputFileName, iRepeat), results, params);
                fprintf('.');
            end

            fprintf(' done.\n');
        end
    end
end

function parsave(fname, results, params)
    save(fname, 'results', 'params')
end