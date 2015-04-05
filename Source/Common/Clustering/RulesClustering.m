function [P1, P11] = RulesClustering(rules, task, errorVectors, ell, P1, P11, params)
    if (~exist('params', 'var'))
        params = [];
    end
    
    params = SetDefault(params, 'useClusterSize', true);
        
    trainRatio = ell / task.nItems;    
    
    P1_tmp = cell(task.nClasses, 1);
    P11_tmp = cell(task.nClasses, 1);
    for targetClass = 1:task.nClasses
        curRules = rules.target == targetClass;
        errorVectors1 = errorVectors(curRules, :);
        errorVectors1(:, task.target ~= targetClass) = false;
        
        errorVectors11 = errorVectors(curRules, :);
        errorVectors11(:, task.target == targetClass) = false;
        
        P1_tmp{targetClass}  = RulesClusteringPerErrorType(P1(curRules),  errorVectors1, trainRatio, params);
        P11_tmp{targetClass} = RulesClusteringPerErrorType(P11(curRules), errorVectors11, trainRatio, params);
    end
    
    P1 = [];
    P11 = [];
    for targetClass = 1:task.nClasses
         P1 = AppendToStructArray(P1, P1_tmp{targetClass});
         P11 = AppendToStructArray(P11, P11_tmp{targetClass});
    end
end

function target = AppendToStructArray(target, source)
    if (isempty(target))
        target = source;
        return;
    end
    
    if (isempty(source))
        return;
    end
    
    targetLen = length(target);
    sourceLen = length(source);
    for i = 1:sourceLen
        target(targetLen + i) = source(i);
    end    
end

function Prepare = RulesClusteringPerErrorType(Prepare, errorVectors, trainRatio, params)
    % Perform rules clustering on errors of type I or type II.

    nRules = size(errorVectors, 1);
    if (nRules == 0)
        Prepare = [];
        return;
    end
    
    nItems = size(errorVectors, 2);
    
    L = nItems;
    ell = floor(L * trainRatio);
    k = L - ell;    
    
    rules.errorVectors = errorVectors;
    rules.errorCount = sum(rules.errorVectors, 2);
    rules.clusterId = (1:nRules)';
    
    tsDistances = CalculateDistances(rules);

    % Generic low and top error vectors for rules clusters.
    % low = intersect error vectors acros rules in the cluster
    % top = union error vectors acros rules in the cluster
    rulesGenericLow = rules.errorVectors;
    rulesGenericTop = rules.errorVectors;
    clusterSizes = ones(nRules, 1);


    s.m_etype=0;   %m1
    s.q_etype=0;   %q1
    s.q_total=0;   %q (total deficiency)
    s.mr_etype=0; 
    s.r_etype=0;
    s.d=0;
    Prepare_tmp = repmat(s, nRules, 1);
    
    for i = 1:nRules
        Prepare_tmp(i).m_etype = Prepare(i).m_etype;
        Prepare_tmp(i).q_etype = Prepare(i).q_etype;
        Prepare_tmp(i).q_total = Prepare(i).q_total;
        Prepare_tmp(i).mr_etype = 0;
        Prepare_tmp(i).r_etype = 0;
        Prepare_tmp(i).d = 1;

        La = L - Prepare_tmp(i).q_total;        
        Prepare_tmp(i).NSO = NoisedSetOverfittingPrepare( ...
            La, ell, Prepare_tmp(i).m_etype - Prepare_tmp(i).q_etype, Prepare_tmp(i).mr_etype, Prepare_tmp(i).r_etype, Prepare_tmp(i).d);
    end
    
    
    for idx = 1:tsLength(tsDistances)
        rule1id = tsDistances.i(idx);
        rule2id = tsDistances.j(idx);
        
        if (Prepare(rule1id).q_total ~= Prepare(rule2id).q_total)
            continue;
        end

        cluster1id = rules.clusterId(rule1id);
        cluster2id = rules.clusterId(rule2id);
        if (cluster1id == cluster2id)
            continue;
        end

        r1gen_low = rulesGenericLow(cluster1id, :);
        r2gen_low = rulesGenericLow(cluster2id, :);
        r1gen_top = rulesGenericTop(cluster1id, :);
        r2gen_top = rulesGenericTop(cluster2id, :);
        size1 = clusterSizes(cluster1id);
        size2 = clusterSizes(cluster2id);

        r12gen_low = (r1gen_low & r2gen_low);
        r12gen_top = (r1gen_top | r2gen_top);

        if (params.useClusterSize)
            d = size1 + size2;
        else
            d = -1;
        end
        
        m1 = sum(r12gen_low);
        m_r = sum(r12gen_top) - m1;
        
        if (m_r > (0.2 * L))
            % clusters can't be too large.
            continue;
        end
        
        r = (size1 * (Prepare_tmp(cluster1id).m_etype + Prepare_tmp(cluster1id).r_etype) + ...
             size2 * (Prepare_tmp(cluster2id).m_etype + Prepare_tmp(cluster2id).r_etype)) / (size1 + size2) - m1;
        q_total = min(Prepare_tmp(cluster1id).q_total, Prepare_tmp(cluster2id).q_total);
        q_etype = min(Prepare_tmp(cluster1id).q_etype, Prepare_tmp(cluster2id).q_etype);
        La = L - q_total;                

        PrepareNSO = NoisedSetOverfittingPrepare(La, ell, m1 - q_etype, m_r, ceil(r), d);
        medianaTogether = bsearch(@(eps)NoisedSetOverfittingCalc(PrepareNSO, eps), 0.5, 1, 0.005, 1); 
        medianaSeparate = bsearch(@(eps)(NoisedSetOverfittingCalc(Prepare_tmp(cluster1id).NSO, eps) + ...
                                         NoisedSetOverfittingCalc(Prepare_tmp(cluster2id).NSO, eps)), 0.5, 1, 0.005, 1);
        
        if (medianaTogether < medianaSeparate)
            % makes sense to combine clusters.
            rules.clusterId(rules.clusterId == cluster2id) = cluster1id;
            clusterSizes(cluster1id) = (size1 + size2);
            clusterSizes(cluster2id) = 0; 
            rulesGenericLow(cluster1id, :) = r12gen_low;
            rulesGenericTop(cluster1id, :) = r12gen_top;
            
            Prepare_tmp(cluster1id).m_etype = m1;
            Prepare_tmp(cluster1id).q_etype = q_etype;
            Prepare_tmp(cluster1id).q_total = q_total;
            Prepare_tmp(cluster1id).mr_etype = m_r;
            Prepare_tmp(cluster1id).r_etype = r;
            Prepare_tmp(cluster1id).d = d;
            Prepare_tmp(cluster1id).NSO = PrepareNSO;
        end
    end
    
    Prepare = Prepare_tmp(clusterSizes > 0);
end 

function values = Selector(structs, ids, fieldname)
    values = zeros(length(ids), 1);
    for i = 1:length(ids)
        values(i) = structs(ids(i)).(fieldname);
    end
end

function tsDistances = CalculateDistances(rules)
    nRules = tsLength(rules);
    tsDistances.i = zeros(nRules * nRules, 1);
    tsDistances.j = zeros(nRules * nRules, 1);
    tsDistances.rho = zeros(nRules * nRules, 1);

    q = 0; % index to fill up the tsDistances table.
    for i = 1:nRules
        ev1 = rules.errorVectors(i, :);
        for j=1:nRules        
            q = q + 1;
            tsDistances.i(q) = i;
            tsDistances.j(q) = j;

            if (i==j) 
                continue; 
            end;

            ev2 = rules.errorVectors(j, :);

            % keep distance set to 0 if algs are comparable.
            if (all(ev1 <= ev2) || all(ev2 <= ev1))
                continue;
            end

            tsDistances.rho(q) = sum(ev1 ~= ev2);
        end
    end

    tsDistances = tsRemove(tsDistances, tsDistances.rho == 0);
    tsDistances = tsSort(tsDistances, 'rho'); % greedy algorithm - merge clusters in order of increasing distance.
end

%===== Evaluation of theoretical and practical growth of eps@0.5
%m = 10;
%L = 100;
%ell = 50;
%k = 50;
%binKoef = CnkCreate(L+10);

%maxC = 20;
%mediana = zeros(maxC, 1);
%medianaTH = zeros(maxC, 1);
%for i=1:maxC
%    mediana(i) = bsearch(@(eps)(i*hhDistr(L, ell, m, ell/L*(m-eps*k), binKoef)), 0.5, 1, 0.0001);
%    medianaTH(i) = sqrt(2 * m * (L-m) / ell / k / (L-1)) * sqrt(log(2 * i / sqrt(2 * pi)));
%end
%plot(1:maxC, mediana, '.-r', 1:maxC, medianaTH, '.-b')
%=================================================================