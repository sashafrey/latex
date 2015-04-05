function infos = CalcCoverageInfo( task, rulesCoverage, rules, params )
    % Calculates informativity for each target class
    
    nItems = task.nItems;
    nClasses = task.nClasses;
    nRules = size(rulesCoverage, 1);
    
    infos = NaN(nRules, nClasses);
    for iTarget = 1:nClasses
        correct = false(nRules, nItems);
        correct(:, (task.target == iTarget)') = true;
        if (~isfield(task, 'weights'))
            [P, N, p, n] = CalcCoveragePNpn(rulesCoverage, correct, rules, params);
        else
            [P, N, p, n] = CalcCoveragePNpn(rulesCoverage, correct, rules, params, task.weights);
        end

        infos(:, iTarget) = params.fInfo(P, N, p, n);
    end
end
