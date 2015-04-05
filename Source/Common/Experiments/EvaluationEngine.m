function EvaluationEngine(tasks)
    if (~exist('tasks', 'var'))
        tasks = LoadTasks;
    end
    
    paramSet = {};
    
    params.filename = 'C:\results18.txt';
    names = fieldnames(tasks)';
    %exclude = {'faults', 'Liver_Disorders', 'Mushrooms', 'Thyroid_Disease', 'wine' };  % require maxRank > 2.
    %names = {'Australian', 'Echo_cardiogram', 'Heart_Disease', 'Liver_Disorders', 'Wisconsin_Breast_Cancer', 'Congressional_Voting', 'Hepatitis', 'Labor_Relations', 'wine'};
    exclude = {};
    
    %% Parallel configuration
    params.nRepeats = 50;
    params.parforLoops = 4;
    params.parallel = true;
    %params.parallel = false;
    %params.parallelConfig = 'trehost';
    params.parallelConfig = 'local';
    params.tryCatch = true;
    params.QFold = 5;

    %% COMBoost params
    params.func = @COMBoost_Evaluation;
    params.maxClassifierLength = 25;
    params.stabilize = false;
    params.AlgName = sprintf('COMBoost-25-adjCV');
    params.adjust = true;
    %paramSet{end + 1, 1} = params;

    params.AlgName = sprintf('COMBoost-25');
    params.adjust = false;
    %paramSet{end + 1, 1} = params;

    %% LibSVM
    for cOpt = [0.001, 0.01, 0.03, 0.1, 0.25, 0.5, 1, 2, 4, 10]
        for kernel = [0, 2]
            params.AlgName = sprintf('LibSVM[t%d][c%.3f]', kernel, cOpt);
            params.LibSvmTrainOptions = sprintf('-q -t %d -c %.3f', kernel, cOpt);
            params.LibSvmPredictOptions = '-q';
            params.func = @LibSVM_Evaluation;
            %paramSet{end + 1, 1} = params;
        end
    end
    
    %% RVM
    for gamma = [0.1, 0.5, 2, 10, 30]
        params.AlgName = sprintf('RVM-%.1f', gamma);
        params.width = gamma;
        params.func = @RVM_Evaluation;
        paramSet{end + 1, 1} = params;
    end
    
    %% C5.0trees
    params.AlgName = 'C5.0trees';
    params.func = @C50_Evaluation;
    %paramSet{end + 1, 1} = params;

    %% C5.0rules
    params.AlgName = 'C5.0rules';
    params.func = @C50_Evaluation;
    params.trees = true;
    %paramSet{end + 1, 1} = params;

    %% C5.0boost
    for trials = [5, 10, 15, 25]
        params.AlgName = sprintf('C5.0boost-%d', trials);
        params.func = @C50_Evaluation;
        params.trees = true;
        params.boost = true;
        params.trials = trials;
        %paramSet{end + 1, 1} = params;
    end
    
    %%sanity-check    
    ok = true;
    for name_cell = names
        name = name_cell{1};
        if (~isfield(tasks, name))
            fprintf('tasks doesnt have %s as field. Aborting experiment.', name);
            ok = false;
        end
    end
    
    if (~ok)
        return;
    end
    
    for i=1:params.nRepeats
        for name_cell = names
            name = name_cell{1};
            if (any(strcmp(name, exclude)))
                continue;
            end
                
            for iParams = 1:length(paramSet)
                params = paramSet{iParams};
                fprintf('Repeat #%d, %s, %s...\n', i, params.AlgName, name)
                
                indexes = cell(params.parforLoops, 1);
                for j = 1:params.parforLoops
                    indexes{j} = GenerateNFoldCVIndexes(tasks.(name).target, params.QFold);
                end
                
                Run(tasks.(name), name, params, indexes);
            end
            fprintf('Repeat #%d, %s, %s done.\n', i, params.AlgName, name)
        end
        
        fprintf('Repeats: %d out of %d completed.\n', i, params.nRepeats)
    end
end

function Run(task, theTaskName, params, indexes)
    Check(params.parforLoops > 0);
    
    if (params.parallel && (matlabpool('size') == 0))
        jm = findResource('scheduler', 'configuration', params.parallelConfig);
        matlabpool(jm);
    end
    
    if (params.parallel)
        parfor j = 1:params.parforLoops
            trainErr = 0;
            testErr = 0;
            localParams = params;
            localTask = task;
            try
                [trainErr, testErr] = params.func(indexes{j}, localTask, localParams);
            catch e
                HandleException(e);
                trainErr = NaN;
                testErr = NaN;                
            end
            
            avgTrainError{j} = trainErr;
            avgTestError{j} = testErr;            
            
            fprintf('parfor loops: %d out of %d completed.\n', j, localParams.parforLoops)
        end
    else
        avgTrainError = cell(params.parforLoops, 1);
        avgTestError = cell(params.parforLoops, 1);
        for j = 1:params.parforLoops
            if (params.tryCatch)
                try
                    [avgTrainError{j}, avgTestError{j}] = params.func(indexes{j}, task, params);
                catch e
                    HandleException(e);
                    avgTrainError{j} = NaN;
                    avgTestError{j} = NaN;
                end
            else
                [avgTrainError{j}, avgTestError{j}] = params.func(indexes{j}, task, params);
            end
            
            fprintf('parfor loops: %d out of %d completed.\n', j, params.parforLoops)
        end        
    end
    
    fid = fopen(params.filename, 'a');
    for j = 1:params.parforLoops
        fprintf(fid, '%s %s %d %d\n', params.AlgName, theTaskName, avgTrainError{j}, avgTestError{j});
        fprintf('%s %s %d %d\n', params.AlgName, theTaskName, avgTrainError{j}, avgTestError{j});
    end
    fclose(fid);        
end
