function [rules, params, debugInfo] = RuleSetGeneratorTEMP(task, terms, params, debugInfo)
    if (~exist('params', 'var'))
        params = [];
    end
    
    if (~exist('debugInfo', 'var'))
        debugInfo = [];
    end
    
    params = SetDefault(params, 'verbose', 0);       % do logging to console
    params = SetDefault(params, 'maxRank', 2);       % max number of terms in rules
    params = SetDefault(params, 'T1',      10);      % number of best terms to select on each iteration
    params = SetDefault(params, 'T2',      10);      % number of rules to return
    params = SetDefault(params, 'nMaxTerms', 300);   % max number of terms to use on each iteration
    params = SetDefault(params, 'shuffle', true);    % boolean flag indicating whether to shuffle rules to avoid preferences to first features
	params = SetDefault(params, 'stabilize', true);  % stabilize rules

    % Doesn't have proven impact to quality, => disabled by default.
    params = SetDefault(params, 'prune', false);            % prune rules on test sample.
    params = SetDefault(params, 'pruneTestRatio', 0.33);  % portion of sample to use for pruning.
    params = SetDefault(params, 'postStabilize', false);    % stabilize rules after pruning
    
    params = SetDefault(params, 'fAdjust', @AdjustPNpnVoid);
    params = SetDefault(params, 'fInfo', @HInfoD);
    
    params = SetDefault(params, 'GatherFeatureUsage', false);
    
    debugInfo.usedFeatures = SortedMatrixCreate();
    
    if (params.verbose > 1)
        fprintf('RSG: #terms = %d, #features = %d, #items = %d\n', length(terms), task.nFeatures, task.nItems);
    end

    % the following code assume that maxRank is lower than number of unique features.
    if (length(unique(terms.feature)) < params.maxRank)
        params.maxRank = length(unique(terms.feature));
    end
    
    if (task.nItems <= 0)
        rules = [];
        return;
    end

    originalTask = task;
    if (params.prune)
        Check((params.pruneTestRatio > 0) && (params.pruneTestRatio < 1));
        [task, testTask] = SplitTask(task, 1 - params.pruneTestRatio);
    end
    
    if (params.postStabilize && ~params.prune)
        fprintf('RuleSetGenerator, WARNING: params.postStabilize = true, params.prune = false.');
    end
    
    nItems = task.nItems;
    nFeatures = task.nFeatures;
    nTerms = tsLength(terms);
  
    termsQuotas = CalibrateTermsQuotas(terms, params.nMaxTerms);
    termsCoverage = CalcTermsCoverage(terms, task);
    
    rules.terms = NaN(nTerms, params.maxRank);
    rules.features = false(nTerms, nFeatures);
    rules.target = NaN(nTerms, 1);
    for iTerm = 1:nTerms
        rules.terms(iTerm, 1) = iTerm;
        rules.features(iTerm, terms.feature(iTerm)) = true;
    end
    rules.coverage = termsCoverage;

    [rules, infos] = SetRulesBestClass(rules, rules.coverage, task, params);
    allrules = rules;
    
    topuniqueIds = selecttopunique(infos, rules.target, params.T1);
    lastRules = tsSelect(rules, topuniqueIds);
    
    for k = 2:params.maxRank
        if (params.verbose > 2)
            fprintf('RSG:k = %d of %d, #rules = %d\n', k, params.maxRank, tsLength(allrules))
        end
        
        nCurTerms = min(sum(termsQuotas), nTerms);
        nLastRules = tsLength(lastRules);
        nCurRules = nLastRules * nCurTerms;
        
        rules.terms = NaN(nCurRules, params.maxRank);
        rules.features = false(nCurRules, nFeatures);
        rules.target = NaN(nCurRules, 1);
        rules.coverage = false(nCurRules, nItems);
        
        idx = 0;
        for iRule = 1:nLastRules
            curTermsIds = CalibrateSelectTermsAccordingToQuotas(terms, termsQuotas);
            sourceRule = tsSelect(lastRules, iRule);
            
            for iTerm = curTermsIds'
                idx = idx + 1;
                if (sourceRule.features(terms.feature(iTerm)))
                    continue;
                end
                
                sourceRuleTermsCount = sum(~isnan(sourceRule.terms));
                rules.terms(idx, :) = sourceRule.terms;
                rules.terms(idx, sourceRuleTermsCount + 1) = iTerm;
                rules.features(idx, :) = sourceRule.features;
                rules.features(idx, terms.feature(iTerm)) = true;
                rules.coverage(idx, :) = sourceRule.coverage & termsCoverage(iTerm, :);
                rules.target(idx) = -1;
            end
        end
        
        rules = tsSelect(rules, rules.target == -1);            
        
        if (params.shuffle)
            % randomly shuffle rules to avoid preferences to first features
            rules = tsSelect(rules, randsample(tsLength(rules), tsLength(rules)));
        end
        
        [rules, infos] = SetRulesBestClass(rules, rules.coverage, task, params);
        allrules = tsConcat(allrules, rules);
        
        if (tsLength(rules) > 0)
            lastRules = tsSelect(rules, selecttopunique(infos, rules.target, params.T1));
        else
            lastRules = [];
        end
    end
    
    rules = allrules;
    rules = rmfield(rules, 'coverage');
    
    if (params.GatherFeatureUsage)
        debugInfo.usedFeatures = SortedMatrixAdd(debugInfo.usedFeatures, rules.features);
    end

    rules = SelectBestRules(rules, terms, task, params);
    
    if (params.stabilize)
        rules = RulesStabilizer(rules, terms, task, params);
    end
    
    if (params.prune)
        rules = RulesPruner( rules, terms, testTask, params );
    end
    
    if (params.postStabilize)
        rules = RulesStabilizer(rules, terms, originalTask, params);
    end
    
    rules = ConvertRulesToExplicitTermFormat(rules, terms);
end

function [indexes] = selecttopunique(values, target, top)
    % Selects "top" values from "values" - individually for each "target".
    indexes = [];
    s.values = values;
    s.target = target;
    s.indexes = (1:tsLength(s))';
    sg = tsGroup(s, 'target');
    for iTarget = 1:length(sg.Info)
        thisTargetValues = sg.Info{iTarget}.values;
        uvalues = unique(thisTargetValues);
        n = min(length(uvalues), top);
        [~, ids] = sort(uvalues, 'descend');
        uvalues = uvalues(ids(1:n));
        [~, ids] = ismember(uvalues, thisTargetValues);
        indexes = [indexes; sg.Info{iTarget}.indexes(ids)];                
    end
end

function [rules, infos] = SetRulesBestClass(rules, rulesCoverage, task, params)
    infos = CalcCoverageInfo(task, rulesCoverage, rules, params);    
    [infos, rules.target] = max(infos, [], 2);
end

function rules = SelectBestRules(rules, terms, task, params)
    totalRules = [];
    for target = unique(rules.target)'
        curRules = tsSelect(rules, rules.target == target);           
        infos = CalcRulesInfo(curRules, terms, task, params);
        [~, ids] = sort(infos, 'descend');
        ids = ids(1:min(length(ids), params.T2));
        bestRules = tsSelect(curRules, ids);
        totalRules = tsConcat(totalRules, bestRules);
    end

    rules = totalRules;
end
