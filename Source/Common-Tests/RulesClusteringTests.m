function RulesClusteringTests
    germanAll = LoadTask('german');
    for i=1:10
        german = GetTaskSubsample(germanAll, randsampleStratified(germanAll.target, 10));
        terms = Calibrate(german);
        rulesAll = RuleSetGeneratorTEMP(german, terms);
        for iClass = 1:german.nClasses
            rules = tsSelect(rulesAll, rulesAll.target == iClass);
            ev = CalcRulesErrorVectors(rules, terms, german);
            graph = BuildHasseGraphOnEV(ev);
            [P1, P11] = PrepareQEpsHasseByType(ev, graph, german.target, rules.target);        
            [P1, P11] = RulesClustering(rules, german, ev, floor(german.nItems / 2), P1, P11);
        end
    end
end