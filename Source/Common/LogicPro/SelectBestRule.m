function rule = SelectBestRule(rules, terms, task, params)
    infos = CalcRulesInfo(rules, terms, task, params);
    [~, id] = max(infos);
    
    rule = tsSelect(rules, id);    
end
