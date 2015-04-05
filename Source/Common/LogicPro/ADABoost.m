function [composition, params] = ADABoost(task, terms, params)
    Check(task.nClasses == 2);
    
    params = SetDefault(params, 'maxRank', 3);
    params = SetDefault(params, 'T1', 15);
    params = SetDefault(params, 'remdup', true);
    params = SetDefault(params, 'maxClassifierLength', 35);
    params = SetDefault(params, 'prohibitedFeatures', true);
    params = SetDefault(params, 'nProhibitedFeatures', floor(task.nFeatures / 3));
    params = SetDefault(params, 'pEncourageCoeff', 0.005);
    params = SetDefault(params, 'pAggresivity', 0.20);
    params = SetDefault(params, 'fAdjust', @AdjustPNpnVoid);
    params = SetDefault(params, 'fInfo', @HInfoD);

    t1count = sum(task.target == 1);
    t2count = sum(task.target == 2);
    [~, tailClass] = max([t1count, t2count]);
    
    composition = [];
    task.weights = ones(task.nItems, 1);
    while(tsLength(composition) < params.maxClassifierLength)
        if (params.prohibitedFeatures)
            prohibitedFeatures = randsample(task.nFeatures, params.nProhibitedFeatures);
            curTerms = tsSelect(terms, ~ismember(terms.feature, prohibitedFeatures));
        else
            curTerms = terms;            
        end
        
        rules = RuleSetGeneratorTEMP(task, curTerms, params);
        info = CalcRulesInfo(rules, [], task, params);
        
        bestRules = GetBestPerClass(rules, info);
        bestRules.weight = CalcRulesWeight(bestRules, task, params);
        if (all(bestRules.weight <= 0))
            break;
        end
        
        bestRules = tsSelect(bestRules, bestRules.weight > 0);
        for iBestRule = 1 : tsLength(bestRules)
            curBestRule = tsSelect(bestRules, iBestRule);

            % see detailed comment about tailClass in COMBoost.
            curBestRule.tailClass = tailClass;
            
            composition = tsConcat(composition, curBestRule);
            task = ReWeightObjects(task, curBestRule, params.pAggresivity);            
        end
    end
end

function w = CalcRulesWeight(rule, task, params)
    [P, N, p, n] = CalcRulesPNpn(rule, [], task, params);
    pRatio = p ./ P;
    pRatio(isnan(pRatio)) = 0;
    nRatio = n ./ N;
    nRatio(isnan(nRatio)) = 0;
    w = log(pRatio ./ max(nRatio, params.pEncourageCoeff));
end

function bestRules = GetBestPerClass(rules, infos)
	for i = unique(rules.target)'
        targetIds = find(rules.target == i);
        [~, id] = max(infos(targetIds));
        ids(i) = targetIds(id);        
	end
    
    bestRules = tsSelect(rules, ids);
end

function task = ReWeightObjects(task, rule, pAggresivity)
    nItems = task.nItems;
    coveredCorrectScale = exp(-1 * pAggresivity * rule.weight);
    coveredWrongScale   = exp( 2 * pAggresivity * rule.weight);
    uncoveredScale      = exp( 1 * pAggresivity * rule.weight);
    covered = CalcRulesCoverage(rule, [], task);
    correct = (rule.target == task.target)';
    
    task.weights(~covered) = task.weights(~covered) * uncoveredScale;
    task.weights(covered & correct) = task.weights(covered & correct) * coveredCorrectScale;
    task.weights(covered & ~correct) = task.weights(covered & ~correct) * coveredWrongScale;
    task.weights = task.weights / sum(task.weights) * nItems;
end
