function [composition, params, debugInfo] = COMBoost(task, terms, params)
    Check(task.nClasses == 2);
    if (~exist('params', 'var'))
        params = [];
    end
    
    params = SetDefault(params, 'maxClassifierLength', 25);
    params = SetDefault(params, 'prohibitedFeatures', true);
    params = SetDefault(params, 'nProhibitedFeatures', floor(task.nFeatures / 3));
    params = SetDefault(params, 'confidence', 1);
    params = SetDefault(params, 'M0', -1);
    params = SetDefault(params, 'M1', 2);
    params = SetDefault(params, 'fAdjust', @AdjustPNpnVoid);
    params = SetDefault(params, 'fInfo', @HInfoD);
  
    [composition, debugInfo] = COMBoost_internal(task, terms, params);
end

function [composition, debugInfo] = COMBoost_internal(task, terms, params, debugInfo)
    if (~exist('debugInfo', 'var'))
        debugInfo = [];
    end
    
    t1count = sum(task.target == 1);
    t2count = sum(task.target == 2);
    [~, tailClass] = max([t1count, t2count]);
    
    composition = [];
    sample = true(task.nItems, 1);
    
    for i = 1:params.maxClassifierLength
        if (~any(sample)) % sample is now empty.
            break;
        end
        
        sampleTask = GetTaskSubsample(task, sample);
        Check(sampleTask.nItems > 0);
        
        if (params.prohibitedFeatures)
            prohibitedFeatures = randsample(task.nFeatures, params.nProhibitedFeatures);
            curTerms = tsSelect(terms, ~ismember(terms.feature, prohibitedFeatures));
        else
            curTerms = terms;            
        end
        
        [rules, debugInfo] = RuleSetGeneratorTEMP(sampleTask, curTerms, params, debugInfo);

        info = CalcRulesInfo(rules, [], sampleTask, params);

        errorsCount = zeros(sampleTask.nClasses, 1);
        for iTarget = 1:sampleTask.nClasses
            rule = SelectBestPerClass(rules, info, iTarget);
            
            % Tail class is the "default" class that ClassifyComposition method
            % should use for those object that were not covered by any rule of the classifier 
            % (or the margin between votes of different classes is too small)
            % Remark: currently, the same tailClass value is stored
            % multiple times --- tsLength(composition). Hense, composition 
            % is still a tsStruct. The contract is to use
            % composition.tailClass(tsLength(composition)) as the correct
            % value. This is convenient for experiments that build the
            % charts that depends on the length of the classifier.            
            rule.tailClass = tailClass;
            
            tmpComposition = tsConcat(composition, rule);
            [targetVector, ~] = ClassifyComposition(tmpComposition, [], sampleTask, params.confidence);
            errorsCount(iTarget) = sum(targetVector ~= sampleTask.target);            
        end
        
        [~, iTarget] = min(errorsCount);
        rule = SelectBestPerClass(rules, info, iTarget);
        if (isempty(rule.terms))
            break;
        end
        
        rule.tailClass = tailClass;
        composition = tsConcat(composition, rule);
        [~, margin] = ClassifyComposition(composition, [], task, params.confidence);
        sample = (params.M0 <= margin) & (margin <= params.M1);
    end
end

function rule = SelectBestPerClass(rules, info, target)
    ids = rules.target == target;

    [~, maxId] = max(info(ids));

    ids = find(ids);
    id = ids(maxId);

    rule = tsSelect(rules, id);
end
