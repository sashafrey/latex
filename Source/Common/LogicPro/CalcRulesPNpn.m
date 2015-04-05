function [P, N, p, n, errorVectors, coverage] = CalcRulesPNpn(rules, terms, task, params)
    nRules = tsLength(rules);
    nItems = task.nItems;
    
    coverage = CalcRulesCoverage(rules, terms, task);
    correct = false(nRules, nItems);
    for i = 1:task.nClasses
        correct(rules.target == i, (task.target == i)') = true;
    end
    
    if (~isfield(task, 'weights'))
        [P, N, p, n] = CalcCoveragePNpn(coverage, correct, rules, params);
    else
        [P, N, p, n] = CalcCoveragePNpn(coverage, correct, rules, params, task.weights);
    end
    
    errorVectors = xor(coverage, correct);
end
