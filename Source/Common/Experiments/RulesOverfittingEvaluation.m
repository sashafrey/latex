function [pTrain, nTrain, pTest, nTest] = RulesOverfittingEvaluation(rules, terms, task, params)
    if (~exist('params', 'var'))    
        params = [];
    end
    
    params = SetDefault(params, 'nIters', 100);
    params = SetDefault(params, 'fInfo', @HInfoD);
    
    %adjTable = CreateAdjTableLocal(task, params);
    %params.fAdjust = @(P, N, p, n, rules)AdjustPNpn(P, N, p, n, rules, adjTable);
    
    params.fAdjust = @AdjustPNpnVoid;
    
    coverage = CalcRulesCoverage(rules, terms, task);
    
    nRules = tsLength(rules);
    nItems = task.nItems;
    nClasses = task.nClasses;
    
    correct = false(nRules, nItems);
    for i = 1:task.nClasses
        correct(rules.target == i, (task.target == i)') = true;
    end
    
    rulesPerClass = cell(nClasses, 1);
    for iClass = 1:nClasses
        rulesPerClass{iClass} = tsSelect(rules, rules.target == iClass);
    end

    pTrain = zeros(params.nIters, nClasses);
    nTrain = zeros(params.nIters, nClasses);
    pTest = zeros(params.nIters, nClasses);
    nTest = zeros(params.nIters, nClasses);

    for j = 1:nClasses
        rulesC{j} = tsSelect(rules, rules.target == j);
    end
    
    for i = 1 : params.nIters
        [~, test, trainIds, ~] = SplitTask(task, 0.5);
        for j = 1:nClasses
            coverageTrain = coverage(rules.target == j, trainIds);
            correctTrain = correct(rules.target == j, trainIds);
            curRules = tsSelect(rules, rules.target == j);
            
            [P, N, p, n] = CalcCoveragePNpn(coverageTrain, correctTrain, curRules, params);
            
            infos = params.fInfo(P, N, p, n);
            [~, id] = max(infos);
            
            pTrain(i, j) = p(id);
            nTrain(i, j) = n(id);
            
            rule = tsSelect(rulesPerClass{j}, id);
            [~, ~, p, n] = CalcRulesPNpn(rule, terms, test, params);
            pTest(i, j) = p;
            nTest(i, j) = n;
        end
    end
end

function adjTable = CreateAdjTableLocal(trainTask, params)
    params.calcCV = true;
    params.nItersCV = 1000;
    params.parallel = false;
    params.verbose = false;
    params.fAdjust = @AdjustPNpnVoid;
    adjTable = CreateAdjustmentsTable(trainTask, params);
    adjTable = adjTable.CV;
end