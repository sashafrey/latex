function rules = ConvertRulesToExplicitTermFormat(rules, terms)
    nRules = tsLength(rules);
    retval = [];
    for i=1:nRules
        rule = tsSelect(rules, i);
        if (iscell(rule.terms))
            retval = tsConcat(retval, rule);
            continue;
        end

        nOriginalTerms = length(rule.terms);
        termIds = rule.terms(~isnan(rule.terms));
        nTerms = length(termIds);
        rule.terms = cell(1, nTerms);

        for iTerm = 1:length(termIds)
            rule.terms{iTerm} = tsSelect(terms, termIds(iTerm));
        end

        % To store rules in table struct the length of rule.terms must be aligned.
        rule.terms((end + 1) : nOriginalTerms) = {[]};
        retval = tsConcat(retval, rule);
    end
    
    rules = retval;
end