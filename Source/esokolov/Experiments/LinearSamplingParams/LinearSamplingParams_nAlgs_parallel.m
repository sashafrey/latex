clear;

SaveFilename = ...
    './esokolov/Experiments/LinearSamplingParams/LinearSamplingParams_nAlgs_nStartAlgs';

% убеждаемся, что ничего не затрем
if exist(sprintf('%s.mat', SaveFilename), 'file') == 2
    userAns = ...
        input('This file already exists.\nAre you sure you want to rewrite it? [y/n]: ', ...
        's');
    if ~strcmp(userAns, 'y')
        return;
    end
end 


tasks = LoadTasks();

nItersMCC = 1024;             % monte-carlo cross-validation
nItersRandomWalk = [128, 256, 512, 1024, 4096, 8192];%, 16384, 32768, 65536];%, ...
    %65536, 131072, 524288, 1048576];      % random walk 
nRays = 128;                  % number of rays
randomSeed = 0;               % random seed
nStartAlgs = [1, 16, 64, 128, 256];%, 1024, 8192, 16384];        
linearSamplingDll_path = 'C:\Storage\vft11ccas\Source\Common\GPU';
nLayers = 15;
pTransition = 0.8;

if (matlabpool('size') == 0)
    matlabpool
end
nRepeats = matlabpool('size');

ratio = 0.5;    % ell / L

tasknames = {
    'Sonar'    
    
    'wine'
    'waveform'
    'pageblocks'
    'Optdigits'
    'pendigits'
    'Magic04'    
    'Letter'
    
    'glass'
    'Liver_Disorders'
    'Ionosphere'    
    'Wdbc'
    'Australian'    
    'pima'
    'faults'
    'statlog'
    };

trainErrorAll = cell(length(tasknames), 1);
testErrorAll = cell(length(tasknames), 1);
trainErrorMCCAll = cell(length(tasknames), 1);
testErrorMCCAll = cell(length(tasknames), 1);

h = maximizeFigure(figure);

for iTask = 1:length(tasknames)
    taskname = tasknames{iTask};
    fprintf('%s\n', taskname);
    task = tasks.(taskname);
    Check(all(unique(task.target) == [1; 2]));
    
    % формируем выборку
    task = NormalizeFeatures(task, false, 2);
    
    parfor iRepeat = 1:nRepeats
        trainError = zeros(length(nItersRandomWalk), length(nStartAlgs));
        testError = zeros(length(nItersRandomWalk), length(nStartAlgs));
        trainErrorMCC = zeros(length(nItersRandomWalk), length(nStartAlgs));
        testErrorMCC = zeros(length(nItersRandomWalk), length(nStartAlgs));

        [train_set, test_set] = SplitTask(task, ratio);
        
        % переобученность логистической регрессии по монте-карло
        [w, trainErrorCurr, testErrorCurr] = ...
            trainLogisticRegression(train_set, test_set);
        trainError(:, :) = ...
            trainError(:, :) + trainErrorCurr;
        testError(:) = ...
            testError(:, :) + testErrorCurr;

        train_set = AddConstantFeature(train_set);
        test_set = AddConstantFeature(test_set);
        
        for iItersRandomWalk = 1:length(nItersRandomWalk)
            fprintf('\t%i: ', iItersRandomWalk);
            
            currItersRandomWalk = nItersRandomWalk(iItersRandomWalk);
            for iStartAlgs = 1:length(nStartAlgs)
                fprintf('%i, ', iStartAlgs);
                
                currStartAlgs = nStartAlgs(iStartAlgs);
                
                % Combinatorial overfitting
                params = [];
                params.nItersMCC = nItersMCC;  % monte-carlo cross-validation
                params.nItersRandomWalk = currItersRandomWalk; % random walk 
                params.nAlgsToSample = currItersRandomWalk;
                params.nRays = nRays;             % number of rays
                params.randomSeed = randomSeed;   % random seed
                params.nStartAlgs = currStartAlgs;        
                params.linearSamplingDll_path = linearSamplingDll_path;
                params.nLayers = nLayers;
                params.pTransition = pTransition;
                [trainErrorMccCurr, testErrorMccCurr] = ...
                    CalcLinearSamplingMCC(train_set, w, params);

                trainErrorMCC(iItersRandomWalk, iStartAlgs) = ...
                    trainErrorMCC(iItersRandomWalk, iStartAlgs) + trainErrorMccCurr;
                testErrorMCC(iItersRandomWalk, iStartAlgs) = ...
                    testErrorMCC(iItersRandomWalk, iStartAlgs) + testErrorMccCurr;

                GPU_CloseAllSessions();

                %fprintf('%i,', iItersRandomWalk);
            end
            fprintf('\n');
        end
        fprintf(' Repeat %i done.\n', iRepeat);
        
        trainErrorParfor{iRepeat} = trainError;
        testErrorParfor{iRepeat} = testError;
        trainErrorMCCParfor{iRepeat} = trainErrorMCC;
        testErrorMCCParfor{iRepeat} = testErrorMCC;
    end
    
    trainError = zeros(length(nItersRandomWalk), length(nStartAlgs));
    testError = zeros(length(nItersRandomWalk), length(nStartAlgs));
    trainErrorMCC = zeros(length(nItersRandomWalk), length(nStartAlgs));
    testErrorMCC = zeros(length(nItersRandomWalk), length(nStartAlgs));
    for iRepeat=1:nRepeats
        trainError = trainError + trainErrorParfor{iRepeat};
        testError = testError + testErrorParfor{iRepeat};
        trainErrorMCC = trainErrorMCC + trainErrorMCCParfor{iRepeat};
        testErrorMCC = testErrorMCC + testErrorMCCParfor{iRepeat};
    end
    
    trainError = trainError / nRepeats;
    testError = testError / nRepeats;
    trainErrorMCC = trainErrorMCC / nRepeats;
    testErrorMCC = testErrorMCC / nRepeats;
    
    trainErrorAll{iTask} = trainError;
    testErrorAll{iTask} = testError;
    trainErrorMCCAll{iTask} = trainErrorMCC;
    testErrorMCCAll{iTask} = testErrorMCC;
    
    subplot(4,4,iTask);
    if length(nStartAlgs) == 1
        plot(1:length(nItersRandomWalk), trainError, 'b-', ...
             1:length(nItersRandomWalk), testError, 'r-', ...
             1:length(nItersRandomWalk), trainErrorMCC, 'b-.', ...
             1:length(nItersRandomWalk), testErrorMCC, 'r-.' ...
             );
    elseif length(nItersRandomWalk) == 1
        plot(1:length(nStartAlgs), trainError, 'b-', ...
             1:length(nStartAlgs), testError, 'r-', ...
             1:length(nStartAlgs), trainErrorMCC, 'b-.', ...
             1:length(nStartAlgs), testErrorMCC, 'r-.' ...
             );
    else
        surf(1:length(nStartAlgs), 1:length(nItersRandomWalk), testErrorMCC);
    end
    
    taskname(taskname == '_') = ' ';
    title(taskname);
    drawnow;
    
    save(SaveFilename, ...
        'nItersRandomWalk', 'nStartAlgs', 'nRepeats', 'tasknames', ...
        'trainErrorAll', 'testErrorAll', 'trainErrorMCCAll', 'testErrorMCCAll');
end

saveas(h, SaveFilename, 'fig');
saveas(h, SaveFilename, 'eps2c');
save(SaveFilename, ...
    'nItersRandomWalk', 'nStartAlgs', 'nRepeats', 'tasknames', ...
    'trainErrorAll', 'testErrorAll', 'trainErrorMCCAll', 'testErrorMCCAll');
