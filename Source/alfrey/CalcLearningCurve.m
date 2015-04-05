tasks = LoadTasks();

nItersMCC = 1000;%10000;            % monte-carlo cross-validation
nItersRandomWalk = 1000; %4*4096/10;    % random walk 
nAlgsToSample = 1000;%4*4096/10;
nRays = 1024;                 % number of rays
randomSeed = 0;               % random seed
nStartAlgs = 128;        
linearSamplingDll_path = '\\alfrey-h01\vft11ccas\source\Common\GPU';
nLayers = 10000;
pTransition = 0.8;


skipMCC = 0;
calcSelfLearningCurveOfMCC = 0;
ratios = 0.05 : 0.05 : 0.95;
nRepeats = 50;
    
tasknames = {
    %'Sonar'    
   %'glass'
    'Liver_Disorders'
    'Ionosphere'    
    'Wdbc'
    'Australian'    
    'pima'
    'faults'
    'statlog'
    'wine'
    'waveform'
    'pageblocks'
    'Optdigits'
    'pendigits'
    'Magic04'    
    'Letter'
    };

for iTask=1:length(tasknames)
    taskname=tasknames{iTask};
    fprintf('%s\n', taskname);
    task = tasks.(taskname);
    Check(all(unique(task.target) == [1; 2]));

    % формируем выборку
    task = NormalizeFeatures(task, false, 2);

    trainError = zeros(length(ratios), 1);
    testError = zeros(length(ratios), 1);
    trainErrorMCC = zeros(length(ratios), 1);
    testErrorMCC = zeros(length(ratios), 1);
    trainErrorMCCSelf = zeros(length(ratios), 1);
    testErrorMCCSelf = zeros(length(ratios), 1);
    trainErrorSVM = zeros(length(ratios), 1);
    testErrorSVM = zeros(length(ratios), 1);
    overfittingBound = zeros(length(ratios), 1);
    overfittingBoundSelf = zeros(length(ratios), 1);

    for iRepeat = 1:nRepeats
        for iRatio = 1:length(ratios)
            ratio = ratios(iRatio);
            [train_set, test_set] = SplitTask(task, ratio);

            %svm_model = svmtrain2(2 * train_set.target - 3, train_set.objects, '-t 0 -c 0.1 -q');
            %svm_train_predictions = 1.5 + 0.5 * svmpredict(2 * train_set.target - 3, train_set.objects, svm_model, '-q');
            %svm_test_predictions = 1.5 + 0.5 * svmpredict(2 * test_set.target - 3, test_set.objects, svm_model, '-q');               
            %trainErrorSVM(iRatio) = trainErrorSVM(iRatio) + sum(svm_train_predictions ~= train_set.target) / train_set.nItems;
            %testErrorSVM(iRatio) = testErrorSVM(iRatio) + sum(svm_test_predictions ~= test_set.target) / test_set.nItems;           
            
            try
                tic
                warning off
                [w] = glmfit(train_set.objects, train_set.target-1, 'binomial', 'link', 'logit');
                warning on
                tuneTime=toc;
            catch
                trainError(iRatio) = NaN;
                testError(iRatio) = NaN;
                warning on                
                continue;
            end            

            w = [w(2:end); w(1)];
            w = w ./ sqrt(w'* w);

            train_set = AddConstantFeature(train_set);
            test_set = AddConstantFeature(test_set);

            trainError(iRatio) = trainError(iRatio) + mean((2 * train_set.target-3) .* (train_set.objects * w) <= 0);
            testError(iRatio) = testError(iRatio) + mean((2 * test_set.target-3) .* (test_set.objects * w) <= 0);
            
            %% Combinatorial overfitting
            if (~skipMCC)
                GPU_Initialize(linearSamplingDll_path);
                randomRays = 2 * (rand(nRays, train_set.nFeatures) - 0.5);
                session_train = GPU_CreateSession(train_set.objects, train_set.target - 1, randomRays);
                %session_test = GPU_CreateSession(test_set.objects, test_set.target - 1, randomRays);

                [~, ~, trainErrorCount, ~] = GPU_CalcAlgs(session_train,  w');
                %[~, ~, ecTestW, ~] = GPU_CalcAlgs(session_test,  w');

                % сэмплируем и считаем оценку
                W0 = ones(nStartAlgs, 1) * w';
                tic
                [algs_rw, sources] = GPU_RunRandomWalking(session_train, ...
                    W0, nAlgsToSample, nItersRandomWalk, trainErrorCount + nLayers, ...
                    pTransition, randomSeed);
                walkingTime=toc;
                if (size(algs_rw, 1) == 0) 
                    fprintf('randomWalkFailed\n'); 
                    trainErrorMCC(iRatio) = NaN;
                    testErrorMCC(iRatio) = NaN;
                    continue; 
                end;

                eps = 0 : (4 / task.nItems) : 1; 
                QEpsComb = GPU_CalcQEpsCombinatorial(session_train, algs_rw, sources, 1, eps, task.nItems / 2);
                QEps_total = sum(QEpsComb);
                combBound = min(eps(QEps_total < 0.5));
                overfittingBound(iRatio) = overfittingBound(iRatio) + combBound;

                % оцениваем переобучение МЭР с помощью папы карло
                [~, ev, ec, ~] = GPU_CalcAlgs(session_train, algs_rw);
                tic
                [~, ~, trainErrorMCCOneIter, testErrorRateMCCOneIter] = GPU_PerformCrossValidation(ev, ec, nItersMCC, randomSeed);
                performCVTime = toc;
                %mcBound = mean(testErrorRateMCC - trainErrorRateMCC);

                trainErrorMCC(iRatio) = trainErrorMCC(iRatio) + mean(trainErrorMCCOneIter);
                testErrorMCC(iRatio) = testErrorMCC(iRatio) + mean(testErrorRateMCCOneIter);

                GPU_CloseAllSessions();
            end
                
            fprintf('.');
        end
        
        fprintf(' Repeat %i done.\n', iRepeat);
    end

    if (calcSelfLearningCurveOfMCC)
        for iRepeat = 1:nRepeats
            warning off
            [w] = glmfit(task.objects, task.target-1, 'binomial', 'link', 'logit');
            warning on
            task2 = AddConstantFeature(task);
            
            GPU_Initialize(linearSamplingDll_path);
            randomRays = 2 * (rand(nRays, task2.nFeatures) - 0.5);
            session = GPU_CreateSession(task2.objects, task2.target - 1, randomRays);

            [~, ~, trainErrorCount, ~] = GPU_CalcAlgs(session,  w');

            W0 = ones(nStartAlgs, 1) * w';
            [algs_rw, sources] = GPU_RunRandomWalking(session, ...
                W0, nAlgsToSample, nItersRandomWalk, trainErrorCount + nLayers, ...
                pTransition, randomSeed);
            [~, ev, ec, ~] = GPU_CalcAlgs(session, algs_rw);
            
            for iRatio = 1 : length(ratios)
                ratio = ratios(iRatio);
                [~, ~, trainErrorMCCOneIter, testErrorRateMCCOneIter] = GPU_PerformCrossValidation(...
                    ev, ec, nItersMCC, randomSeed, floor(ratio * train_set.nItems));
                trainErrorMCCSelf(iRatio) = trainErrorMCCSelf(iRatio) + mean(trainErrorMCCOneIter);
                testErrorMCCSelf(iRatio) = testErrorMCCSelf(iRatio) + mean(testErrorRateMCCOneIter);

                eps = 0 : (1 / (3*task.nItems)) : 1;
                QEpsComb = GPU_CalcQEpsCombinatorial(session, algs_rw, sources, eps, floor(ratio * task.nItems));
                QEps_total = sum(QEpsComb);
                combBound = min(eps(QEps_total < 0.5));
                overfittingBoundSelf(iRatio) = overfittingBoundSelf(iRatio) + combBound;
            end   
            
            GPU_CloseAllSessions();
        end
    end        
    
    trainError = trainError / nRepeats;
    testError = testError / nRepeats;
    trainErrorMCC = trainErrorMCC / nRepeats;
    testErrorMCC = testErrorMCC / nRepeats;
    trainErrorSVM = trainErrorSVM / nRepeats;
    testErrorSVM = testErrorSVM / nRepeats;    
    trainErrorMCCSelf = trainErrorMCCSelf / nRepeats;
    testErrorMCCSelf = testErrorMCCSelf / nRepeats;
    overfittingBoundSelf = overfittingBoundSelf / nRepeats;
    overfittingBound = overfittingBound / nRepeats;
    
    
    results.trainError.(taskname) = trainError;
    results.testError.(taskname) = testError;    
    results.trainErrorMCC.(taskname) = trainErrorMCC;
    results.testErrorMCC.(taskname) = testErrorMCC;
    results.ratios.(taskname) = ratios;
    results.task.(taskname) = task;
    save results

    subplot(4,4,iTask);
    plot(floor(results.ratios.(taskname) * results.task.(taskname).nItems), results.trainErrorMCC.(taskname), 'k:', ...
         floor(results.ratios.(taskname) * results.task.(taskname).nItems), results.testErrorMCC.(taskname), 'k-', ...
         floor(results.ratios.(taskname) * results.task.(taskname).nItems), results.trainError.(taskname), 'k:x', ...
         floor(results.ratios.(taskname) * results.task.(taskname).nItems), results.testError.(taskname), 'k-x')
     
    taskname(taskname == '_') = ' ';
    title(taskname);
    drawnow
end