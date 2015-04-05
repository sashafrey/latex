function info = CalcRulesInfo(rules, terms, task, params)
    [P, N, p, n] = CalcRulesPNpn(rules, terms, task, params);
    info = params.fInfo(P, N, p, n);
end
