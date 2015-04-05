function [ rules ] = RulesPruner( rules, terms, task, params )
    newRules = [];
    
    for iRule = 1:tsLength(rules)
        rule = tsSelect(rules, iRule);
        origInfo = CalcRulesInfo(rule, terms, task, params);

        while(true)
            nTerms = sum(rule.features);
            if (nTerms == 1)
                break;
            end

            infos = zeros(nTerms, 1);
            for iTerm = 1:nTerms
                rule2 = RemoveTermFromRule(rule, iTerm);
                infos(iTerm) = CalcRulesInfo(rule2, terms, task, params);
            end

            if (origInfo > max(infos))
                break;
            end

            [~, toRemove] = max(infos);
            rule = RemoveTermFromRule(rule, toRemove);
            origInfo = infos(toRemove);
        end
        
        newRules = tsConcat(newRules, rule);
    end
    
    rules = newRules;
end

function rule = RemoveTermFromRule(rule, iTerm)
    explicitTerms = iscell(rule.terms);
    if (explicitTerms)
        rule.terms(iTerm) = {[]};
    else
        rule.terms(iTerm) = NaN;
    end
    
    activeFeatures = find(rule.features);
    rule.features(activeFeatures(iTerm)) = false;
end
