function rulesCoverage = CalcRulesCoverage(rules, terms, task)
    nRules = tsLength(rules);
    if (nRules == 0)
        rulesCoverage = [];
        return;
    end

    nItems = task.nItems;
    rulesCoverage = false(nRules, nItems);    

    if (iscell(rules.terms))
        for iRule = 1:nRules
            if (isnan(rules.target(iRule)))
                % to handle VOID rules
                rulesCoverage(iRule, :) = false(1, nItems);
                continue;
            end
            
            terms = rules.terms(iRule, :);
            coverage = true(1, nItems);
            for i = 1:length(terms)
                if (isempty(terms{i})) 
                    continue;
                end
                
                coverage = coverage & CalcOneTermCoverage(terms{i}.left, terms{i}.right, terms{i}.feature, terms{i}.isnot, task)';
            end
            
             rulesCoverage(iRule, :) = coverage ;
        end
    else
        termsCoverage = CalcTermsCoverage(terms, task);%todo:считать покрытия только тех термов которые используются в правилах
        for iRule = 1:nRules
            if (isnan(rules.target(iRule)))
                % to handle VOID rules
                rulesCoverage(iRule, :) = false(1, nItems);
                continue;
            end
            
            terms = rules.terms(iRule, :);
            coverage = true(1, nItems);
            for iTerm = terms(~isnan(terms))
                coverage = coverage & termsCoverage(iTerm, :);
            end

            rulesCoverage(iRule, :) = coverage ;
        end
    end
end
