function TuneRuleSetGeneratorParameters(tasks, params)
    if (~exist('tasks',  'var'))
        tasks = LoadTasks;
    end
    
    if (~exist('params', 'var'))
        params = [];
    end
    
    for iIteration = 1:10
    params = SetDefault(params, 'nIters', 8*19);
    params = SetDefault(params, 'parallel', true);
    params = SetDefault(params, 'verbose', 0);
    params = SetDefault(params, 'filename', 'C:\tunerules01.txt');
    params = SetDefault(params, 'fAdjust', @AdjustPNpnVoid);
    params = SetDefault(params, 'fInfo', @HInfoD);
    
    MatlabPoolInitialize;
    
    paramsSet = {};
    paramsSet = AddParams(paramsSet, params, false, false, false, 'stabilize=no, prune=no, postStabilize=no');
    paramsSet = AddParams(paramsSet, params, true, false, false, 'stabilize=yes, prune=no, postStabilize=no');
    paramsSet = AddParams(paramsSet, params, true, true, false, 'stabilize=yes, prune=yes, postStabilize=no');
    paramsSet = AddParams(paramsSet, params, true, true, true, 'stabilize=yes, prune=yes, postStabilize=yes');
    
    tasknames = fieldnames(tasks);
    fprintf('taskname\tmaxRank\tT1\ttargetclass\tmeanTrainInfo\tmeanTestInfo\tstdTestInfo\tRuleLength\tExtraOptions\n');
    %for maxRank = 1:3
    for maxRank = 1:6
	for T1 = [1 2 4 7 10 15 25]
    for i = 1:length(tasknames)
    for iP = 1:length(paramsSet)
        params = paramsSet{iP};
        params.maxRank = maxRank;
        params.T1 = T1;
        taskname = tasknames{i};
        task = tasks.(taskname);
        [trainInfo, testInfo, ruleLength] = CV(task, params);

        fid = fopen(params.filename, 'a');
        for iClass = 1:task.nClasses
            fprintf('%s\t %d\t %d\t %d\t %.2f\t %.2f\t %.2f\t %.2f\t %s\t\n', taskname, maxRank, T1, iClass, mean(trainInfo(:, iClass)), mean(testInfo(:, iClass)), std(testInfo(:, iClass)), mean(ruleLength(:, iClass)), params.runDescription);
            fprintf(fid, '%s\t %d\t %d\t %d\t %.2f\t %.2f\t %.2f\t %.2f\t %s\t\n', taskname, maxRank, T1, iClass, mean(trainInfo(:, iClass)), mean(testInfo(:, iClass)), std(testInfo(:, iClass)), mean(ruleLength(:, iClass)), params.runDescription);
        end
        fclose(fid);        
    end
    end
    end
    end
    end
end

function paramsSet = AddParams(paramsSet, params, stabilize, prune, postStabilize, name)
    params.stabilize = stabilize;
    params.prune = prune;
    params.postStabilize = postStabilize;    
    params.runDescription = name;
    paramsSet{end+1, 1} = params;
end
    

function [trainInfo, testInfo, ruleLength] = CV(task, params)
    trainInfo = zeros(params.nIters, task.nClasses);
    testInfo = zeros(params.nIters, task.nClasses);
    if (params.parallel)
        parfor i = 1:params.nIters
            if (params.verbose > 0)
                fprintf('Iter %i out of %i\n', i, params.nIters);
            end
            [curTrainInfo, curTestInfo, curRuleLength] = GeteInfos(task, params);
            trainInfo(i, :) = curTrainInfo;
            testInfo(i, :) = curTestInfo;
            ruleLength(i, :) = curRuleLength;
        end
    else
        for i = 1:params.nIters
            if (params.verbose > 0)
                fprintf('Iter %i out of %i\n', i, params.nIters);
            end
            [curTrainInfo, curTestInfo, curRuleLength] = GeteInfos(task, params);
            trainInfo(i, :) = curTrainInfo;
            testInfo(i, :) = curTestInfo;
            ruleLength(i, :) = curRuleLength;
        end
    end
end

function [trainInfo, testInfo, ruleLength] = GeteInfos(task, params)
    [train, test] = SplitTask(task, 0.5);
    terms = Calibrate(train);
    rules = RuleSetGeneratorTEMP(train, terms, params);
    for target = unique(rules.target)'
       curRules = tsSelect(rules, rules.target == target);           
       infos = CalcRulesInfo(curRules, terms, train, params);
       [~, id] = max(infos);
       bestRule = tsSelect(curRules, id);
       trainInfo(1, target) = CalcRulesInfo(bestRule, terms, train);       
       testInfo(1, target) = CalcRulesInfo(bestRule, terms, test);
       ruleLength(1, target) = sum(bestRule.features);
    end        
end