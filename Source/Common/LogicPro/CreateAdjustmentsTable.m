function [adjTables, sets, setsHashes] = CreateAdjustmentsTable(task, params)
    params = SetDefault(params, 'max_features', 2);
    params = SetDefault(params, 'nTopRules', 500);
    params = SetDefault(params, 'testMode', false);
    params = SetDefault(params, 'parallel', false);
    params = SetDefault(params, 'verbose', false);
    params = SetDefault(params, 'fAdjust', @AdjustPNpnVoid);
    params = SetDefault(params, 'fInfo', @HInfoD);
    
    params = SetDefault(params, 'calcCV', false);
    params = SetDefault(params, 'nItersCV', 1000);
    params = SetDefault(params, 'calcHasse', false);
    params = SetDefault(params, 'calcHasseCluster', false);
    params = SetDefault(params, 'calcHasseClusterNoSize', false);
    params = SetDefault(params, 'calcBool', false);
    params = SetDefault(params, 'calcBoolCluster', false);
    params = SetDefault(params, 'calcBoolClusterNoSize', false);
    
    sets = CreateAllSets(task.nFeatures, params.max_features);
    
    if (params.testMode)
        %HACK-HACK, for testing.
        nTestModeSubSets = 100;
        if (nTestModeSubSets < size(sets, 1))
            sets = sets(randsample(size(sets, 1), nTestModeSubSets), :);
        end
    end
    
    terms = Calibrate(task);
    nSets = size(sets, 1);
    setsHashes = zeros(nSets, 1);
    for i = 1:nSets
        setsHashes(i, 1) = vectorHash(sets(i, :));
    end
    
    if (params.parallel)
        if (matlabpool('size') == 0)
            matlabpool
        end
        parfor i = 1:nSets
            [pKoefs, nKoefs] = RunIteration(sets(i, :), terms, task, params);
            pKoefsCell{i} = pKoefs;
            nKoefsCell{i} = nKoefs;
            if (params.verbose)
                fprintf('%d out of %d completed.\n', i, nSets)
            end
        end
    else
        pKoefsCell = cell(nSets, 1);
        nKoefsCell = cell(nSets, 1);
        for i = 1:nSets
            [pKoefs, nKoefs] = RunIteration(sets(i, :), terms, task, params);
            pKoefsCell{i} = pKoefs;
            nKoefsCell{i} = nKoefs;

            if (params.verbose)
                fprintf('%d out of %d completed.\n', i, nSets)
            end
        end
    end

    adjTables = [];
    names = fieldnames(pKoefsCell{1});
    for nameCell = names'
        name=nameCell{1};
        table = [];
        table.pKoef = zeros(nSets, task.nClasses);
        table.nKoef = zeros(nSets, task.nClasses);
        for i=1:nSets
            for j=1:task.nClasses
                table.pKoef(i, j) = pKoefsCell{i}.(name)(j);
                table.nKoef(i, j) = nKoefsCell{i}.(name)(j);
            end
        end 
        
        table.FeatureSets = sets;
        table.FeatureSetsHash = setsHashes;
        adjTables.(name) = table;
    end
end

function [pKoefs, nKoefs] = RunIteration(featureSet, terms, task, params)
    curTerms = tsSelect(terms, ismember(terms.feature, featureSet));

    localParams = params;
    localParams.maxRank = sum(~isnan(featureSet));
    
    params.verbose = false;
    curRules = RuleSetGeneratorTEMP(task, curTerms, params);
    [~, ~, ~, ~, errorVectorsAllClasses, coverage] = CalcRulesPNpn(curRules, curTerms, task, params);
    
    for iClass = 1:task.nClasses
        errorVectors = errorVectorsAllClasses(curRules.target == iClass, :);
        curRulesLocal = tsSelect(curRules, curRules.target == iClass);
        ev = sum(errorVectors,2);
        [~, idx] = sort(ev);
        idx = idx(1:min(length(idx), params.nTopRules));
        curRulesLocal = tsSelect(curRulesLocal, idx);
        errorVectors = errorVectors(idx,:);

        ell = floor(task.nItems/2);
        k = task.nItems - ell;

        if (params.calcHasse || params.calcHasseCluster || params.calcHasseClusterNoSize)
            graph = BuildHasseGraphOnEV(errorVectors);
            [P1, P11] = PrepareQEpsHasseByType(errorVectors, graph, task.target, curRulesLocal.target);

            if (params.calcHasse)
                % Hasse bound.
                pKoefs.Hasse(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P1, eps, ell, k), 0.5, 1);
                nKoefs.Hasse(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P11, eps, ell, k), 0.5, 1);
            end
               
            if (params.calcHasseCluster)
                % Hasse clustering via "NoisedSetOverfitting" set with account for cluster size.
                localParams.useClusterSize = true;
                [P1_CL, P11_CL] = RulesClustering(curRulesLocal, task, errorVectors, ell, P1, P11, localParams);
                pKoefs.HasseCluster(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P1_CL, eps, ell, k), 0.5, 1);
                nKoefs.HasseCluster(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P11_CL, eps, ell, k), 0.5, 1);
            end

            if (params.calcHasseClusterNoSize)
                % Hasse clustering via "NoisedSetOverfitting" set with no account for cluster size.
                localParams.useClusterSize = false;
                [P1_CL, P11_CL] = RulesClustering(curRulesLocal, task, errorVectors, ell, P1, P11, localParams);
                pKoefs.HasseClusterNoSize(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P1_CL, eps, ell, k), 0.5, 1);
                nKoefs.HasseClusterNoSize(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P11_CL, eps, ell, k), 0.5, 1);
            end
        end

        if (params.calcBool)
            % Bool bound.        
            [P1, P11] = PrepareQEpsHasseByType(errorVectors, [], task.target, curRulesLocal.target);
            pKoefs.Bool(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P1, eps, ell, k), 0.5, 1);
            nKoefs.Bool(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P11, eps, ell, k), 0.5, 1);        
        end
        
        if (params.calcBoolCluster)
            % Bool clustering via "NoisedSetOverfitting" set with account for cluster size.
            localParams.useClusterSize = true;
            [P1_CL, P11_CL] = RulesClustering(curRulesLocal, task, errorVectors, ell, P1, P11, localParams);
            pKoefs.BoolCluster(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P1_CL, eps, ell, k), 0.5, 1);
            nKoefs.BoolCluster(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P11_CL, eps, ell, k), 0.5, 1);
        end
        
        if (params.calcBoolClusterNoSize)
            % Bool clustering via "NoisedSetOverfitting" set with no account for cluster size.
            localParams.useClusterSize = false;
            [P1_CL, P11_CL] = RulesClustering(curRulesLocal, task, errorVectors, ell, P1, P11, localParams);
            pKoefs.BoolClusterNoSize(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P1_CL, eps, ell, k), 0.5, 1);
            nKoefs.BoolClusterNoSize(iClass) = bsearch(@(eps)CalcQEpsHasseByType(P11_CL, eps, ell, k), 0.5, 1);
        end
    end

    if (params.calcCV)
        pKoefs.CV = [];
        nKoefs.CV = [];
        for iClass = 1:task.nClasses
            curTargetClassRulesCoverage = coverage((curRules.target == iClass), :);
            [pKoefs.CV(iClass), nKoefs.CV(iClass)] = CV(curTargetClassRulesCoverage, task.target, iClass, params.nItersCV, params.nTopRules, params);
        end
    end
end

% creates all subsets of length less or equal to max_features.
function sets = CreateAllSets(nFeatures, max_features)
    sets = [];
    for i = 1:max_features
        curSet = CreateSets(nFeatures, i);
        curSetLen = size(curSet, 1);
        setsLen = size(sets, 1);
        sets((setsLen + 1) : (setsLen + curSetLen), 1:i) = curSet;
        sets((setsLen + 1) : (setsLen + curSetLen), (i+1) : max_features) = NaN;        
    end            
end

% creates all subsets of a given length
function sets = CreateSets(nFeatures, max_features)
    len = nchoosek(nFeatures, max_features);
    sets = zeros(len, max_features);
    vec = 1:max_features;
    sets(1, :) = vec;
    pos = 2;
    while(true)
        id = find(vec < nFeatures, 1, 'last' );
        
        if (isempty(id))
            break;
        end
        
        vec(id) = vec(id) + 1;
        for i = (id + 1) : max_features
            vec(i) = vec(i - 1) + 1;
        end

        if (vec(max_features) > nFeatures)
            continue;
        end
        
        sets(pos, :) = vec;
        pos = pos + 1;
    end    
end

function [pKoef, nKoef] = CV(coverage, targetVector, targetClass, nIters, nTopRules, params)
    correct = targetVector == targetClass;
    
    nItems = length(targetVector);
    nRules = size(coverage, 1);
    ell = floor(nItems / 2);
   
    pKoef = zeros(nIters, 1);
    nKoef = zeros(nIters, 1);

    correctM = false(nRules, nItems);
    correctM(:, correct) = true;
        
    P = ones(nRules, 1) * sum(correct);
    N = ones(nRules, 1) * sum(~correct);
    pVec = sum(coverage & correctM, 2);
    nVec = sum(coverage & ~correctM, 2);
    infosAll = params.fInfo(P, N, pVec, nVec);
    
    [~, ids] = sort(infosAll, 'descend');
    ids = ids(1:min(nTopRules, length(ids)));
    nRules = length(ids);
    correctM = correctM(ids, :);
    coverage = coverage(ids, :);    
    infosAll = infosAll(ids);
    
    ranks = zeros(nIters, 1);
    for i = 1:nIters
        trainMask = randsampleStratified(targetVector, ell);
        trainMaskM = false(nRules, nItems);
        trainMaskM(:, trainMask) = true;
        testMask = ~trainMask;

        PTrain = sum(correct(trainMask));
        NTrain = sum(~correct(trainMask));
        PTest = sum(correct(~trainMask));
        NTest = sum(~correct(~trainMask));
        
        pTrainVec = sum(trainMaskM & coverage & correctM, 2);
        nTrainVec = sum(trainMaskM & coverage & ~correctM, 2);
        
        infos = params.fInfo(PTrain, NTrain, pTrainVec, nTrainVec);
        
        [~, id] = max(infos);
        
        ranks(i) = sum(infosAll >= infosAll(id));
        
        pTest = sum(testMask & coverage(id, :) & correct', 2);
        nTest = sum(testMask & coverage(id, :) & ~correct', 2);
        pTrain = pTrainVec(id);
        nTrain = nTrainVec(id);
        
        pKoef(i) = pTrain - pTest;
        nKoef(i) = nTest - nTrain;
    end

    pKoef(isnan(pKoef)) = 0;
    nKoef(isnan(nKoef)) = 0;    

    pKoef = mean(pKoef);
    nKoef = mean(nKoef);
    
    pKoef(pKoef < 0) = 0;
    nKoef(nKoef < 0) = 0;
    
    pKoef = pKoef / ell;
    nKoef = nKoef / ell;
end
