function errorVectors = CalcRulesErrorVectors(rules, terms, task)
    nRules = tsLength(rules);
    nItems = task.nItems;
    
    coverage = CalcRulesCoverage(rules, terms, task);
    correct = false(nRules, nItems);
    for i = 1:task.nClasses
        correct(rules.target == i, (task.target == i)') = true;
    end
    
    errorVectors = xor(coverage, correct);
end