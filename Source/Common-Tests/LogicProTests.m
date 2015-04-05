function LogicProTests
    german = LoadTask('german');
    Check(german.nItems == 1000);
    Check(german.nFeatures == 20);
    Check(german.nClasses == 2);
    
    terms = Calibrate(german);
    
    [trainTask, testTask] = SplitTask(german, 0.5);
    
    params.maxRank = 3;
    params.T1 = 10;
    params.stabilize = true;
    params.prune = true;
    params.postStabilize = true;
    params.fInfo = @HInfoD;
    params.fAdjust = @AdjustPNpnVoid;
    
    rules = RuleSetGeneratorTEMP(trainTask, terms, params);
    infoTrain = CalcRulesInfo(rules, terms, trainTask, params);
    infoTest = CalcRulesInfo(rules, terms, testTask, params);
    
    for target = 1:german.nClasses
        Check(sum(rules.target == target) >= 10);
        Check(max(infoTrain(rules.target == target)) > 25);
        Check(max(infoTest(rules.target == target)) > 25);
    end
    
    nRules = tsLength(rules);
    for iRule = randsample(nRules, min(nRules, 25))'
        origRule = tsSelect(rules, iRule);
        stabRule = RulesStabilizer(origRule, terms, german, params);
        origInfo = CalcRulesInfo(origRule, terms, german, params);
        stabInfo = CalcRulesInfo(stabRule, terms, german, params);
        Check(stabInfo >= origInfo);
    end

    params.maxRank = 2;
    params.T1 = 5;
    params.maxClassifierLength = 6;
    
    adaB = ADABoost(german, terms, params);
    Check(tsLength(adaB) == params.maxClassifierLength);
    Check(GetErrorRate(adaB, terms, german) < 0.5);  
    
    comB = COMBoost(german, terms, params);
    Check(tsLength(comB) == params.maxClassifierLength);
    Check(GetErrorRate(comB, terms, german) < 0.5);   
end

function errorRate = GetErrorRate(composition, terms, task)
    prediction = ClassifyComposition(composition, terms, task, 0.5); 
    errorRate = sum(prediction ~= task.target) / task.nItems;
end