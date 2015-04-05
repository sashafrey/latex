function rules = RulesStabilizer(rules, terms, task, params)
    newRules = [];
    rules = ConvertRulesToExplicitTermFormat(rules, terms);
    
    for iRule=1:tsLength(rules)
        rule = tsSelect(rules, iRule);
        rule = FillInExtraStructures(rule, task);
        
        % Safaety guard. The while loop will converge anyway, because on each 
        % step we strictly increase informativity. If there is no way to 
        % increase informativity, all terms are marked as stable.
        iIter = 0; 
        while(~all(rule.isstable))
            iIter = iIter + 1;
            rule = StabilizeIteration(rule, task, params);
            if (iIter > (3 * length(rule.terms)))
                break;
            end
        end

        rule = rmfield(rule, 'isnotsupported');    
        rule = rmfield(rule, 'isstable');    
        
        
        newRules = tsConcat(newRules, rule);
    end
    
    rules = newRules;
end

function rule = FillInExtraStructures(rule, task)
    nTerms = length(rule.terms);

    rule.isstable = false(1, nTerms);
    rule.isnotsupported = false(1, nTerms);    
    for iTerm = 1:nTerms
        if (isempty(rule.terms{iTerm}))
            rule.isnotsupported(iTerm) = true;
            rule.isstable(iTerm) = true;
        else
            rule.isnotsupported(iTerm) = task.isnominal(rule.terms{iTerm}.feature) || rule.terms{iTerm}.isnot;
            rule.isstable(iTerm) = rule.isnotsupported(iTerm);
        end
    end
end

function rule = StabilizeIteration(rule, task, params)
    newRules = [];
    t = rule.target;
    nTerms = length(rule.terms);
    for iTerm = find(~rule.isstable)
        feature = rule.terms{iTerm}.feature;
        excludedCoverage = CalcExcludedCoverate(rule.terms, iTerm, task);
        excludedCoverage = excludedCoverage & ~isnan(task.objects(:, feature))';
        
        if (~any(excludedCoverage))
            continue;
        end
        
        fV = task.objects(excludedCoverage, feature);
        tV = task.target(excludedCoverage);
        
        leftT = rule.terms{iTerm}.left;
        rightT = rule.terms{iTerm}.right;
        
        if (isnan(leftT) || isnan(rightT))
            continue;
        end
        
        %VisualizeTerm(fV, tV, leftT, rightT, t)
        
        %**Stop -- object-blocker from other class. The threshold can't be extended beyond **Stop objects.
        %**NewT  -- new threshold. If NaN - then it makes no sense to adjust term in that corresponding dirrection.
        % In the pictograms 'o' --- object of the other class, 'x' --- object of own class, '|' --- term threshold
        
        %LL, extention of the term. Makes sense to include more own, and less other terms.
        % The 'oooxxx|xxxxxx|ooo' term must be extended to the left
        llStop = max(fV((fV <= leftT) & (tV ~= t))); if(isempty(llStop)) llStop = -Inf; end;
        llNewT = min(fV((fV <  leftT) & (tV == t) & (fV > llStop))); if (isempty(llNewT)) llNewT = NaN; end;
        if (~isinf(llStop)) llNewT = (llNewT + llStop) / 2; else llNewT = llNewT - 1; end;
        
        %RR, extention of the term. Makes sense to include more own, and less other terms.
        % The 'oooooo|xxxxxx|xxxooo' term must be extended to the right
        rrStop = min(fV((fV >= rightT) & (tV ~= t))); if(isempty(rrStop)) rrStop = +Inf; end;
        rrNewT = max(fV((fV >  rightT) & (tV == t) & (fV < rrStop))); if (isempty(rrNewT)) rrNewT = NaN; end;
        if (~isinf(rrStop)) rrNewT = (rrNewT + rrStop) / 2; else rrNewT = rrNewT + 1; end;
        
        %LR, shrinking of the term. Makes sense to exclude more other terms, and less own terms.
        % The 'oooooo|ooxxxx|oooooo' term must be shrinked from the left hand side
        lrStop = min(fV((fV >= leftT) & (tV == t))); if(isempty(lrStop)) lrStop = +Inf; end;
        lrNewT = max(fV((fV >  leftT) & (tV ~= t) & (fV < lrStop) & (fV < rightT))); if (isempty(lrNewT)) lrNewT = NaN; end;
        if (~isinf(lrStop)) lrNewT = (lrNewT + lrStop) / 2; end;
        
        %RL, shrinking of the term. Makes sense to exclude more other terms, and less own terms.
        % The 'oooooo|xxxxoo|oooooo' term must be shrinked from the right hand side
        rlStop = max(fV((fV <= rightT) & (tV == t))); if(isempty(rlStop)) rlStop = -Inf; end;
        rlNewT = min(fV((fV <  rightT) & (tV ~= t) & (fV > rlStop) & (fV > leftT))); if (isempty(rlNewT)) rlNewT = NaN; end;
        if (~isinf(rlStop)) rlNewT = (rlNewT + rlStop) / 2; end;
       
        termisstable = isnan(llNewT) && isnan(lrNewT) && isnan(rlNewT) && isnan(rrNewT);
        if (termisstable) 
            continue;
        end        
        
        newLeftTs = [leftT, llNewT, lrNewT];
        newLeftTs(isnan(newLeftTs)) = [];
        
        newRightTs = [rightT, rlNewT, rrNewT];
        newRightTs(isnan(newRightTs)) = [];
        
        for newLeftT = newLeftTs
            for newRightT = newRightTs
                if ((newLeftT == leftT) && (newRightT == rightT))
                    % Skip original term to avoid duplicates in newRules.
                    continue;
                end
                
                newRule = rule;
                newRule.terms{iTerm}.left = newLeftT;
                newRule.terms{iTerm}.right = newRightT;
                newRule.isstable = rule.isnotsupported; % request re-stabilized other terms.
                newRule.isstable(iTerm) = true;
                newRules = tsConcat(newRules, newRule);
                
                % VisualizeTerm(fV, tV, newLeftT, newRightT, t)
            end
        end
    end
    
    if (isempty(newRules))
        % Set all terms as stable.
        rule.isstable = true(1, nTerms);
        return; 
    end
    
    % Add original rule. If it wins -- then it is stable.
    rule.isstable(:) = true;
    newRules = tsConcat(newRules, rule);
    
    info = CalcRulesInfo(newRules, [], task, params);
    [~, id] = max(info);
    rule = tsSelect(newRules, id);
end

function coverage = CalcExcludedCoverate(terms, excludeId, task)
    coverage = true(1, task.nItems);
    ids = 1:length(terms);
    ids(excludeId) = [];
    for i = ids
        if (isempty(terms{i}))
            continue;
        end
        coverage = coverage & CalcOneTermCoverage(terms{i}.left, terms{i}.right, terms{i}.feature, terms{i}.isnot, task)';
    end
end

% VisualizeTerm function is for debugging purpose.
function VisualizeTerm(fV, tV, leftT, rightT, target)
    [fV, ids] = sort(fV);
    tV = tV(ids);
    
    leftDrawn = false;
    rightDrawn = false;
    groupLen = 0; %group stands for the group of objects with the save value of the feature
    for i=1:length(fV)
        if (tV(i) == target)
            symb = 'x';
        else
            symb = 'o';
        end
        
        if (~leftDrawn && (fV(i) >= leftT)) % >=, left inclusive
            fprintf('[');
            leftDrawn = true;
        end
        
        if (~rightDrawn && (fV(i) >= rightT)) % >=, intentionally. Right exclusive, but still >= here.
            fprintf(']');
            rightDrawn = true;
        end
        
        if (i < length(fV))
            if (fV(i+1) == fV(i))
                % Create or continue group.
                if (groupLen == 0)
                    fprintf('{');
                end
                groupLen = groupLen + 1;
            end
        end
        
        fprintf('%s', symb);
        
        if (groupLen > 0)
            %perhaps is time to close the group?
            if ((i == length(fV)) || fV(i+1)~=fV(i))
                groupLen = 0;
                fprintf('}');
            end
        end
    end
    
    if (~rightDrawn)
        fprintf(']');
    end
    fprintf('\n');
end