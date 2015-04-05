function [P, N, p, n] = AdjustPNpn(P, N, p, n, rules, adjTable)
    if (isempty(rules))
        return;
    end
    
    nAdjTableFeature = size(adjTable.FeatureSets, 2);
    
    for i = 1:tsLength(rules)
        target = rules.target(i);
        features = find(rules.features(i, :));
        if (length(features) > nAdjTableFeature)
            p(i) = 0;
            n(i) = 0;
            continue;
        end
        
        features((length(features) + 1) : nAdjTableFeature) = NaN;
        featuresHash = vectorHash(features);
        
        id = find(featuresHash == adjTable.FeatureSetsHash);
        if (isempty(id))
            p(i) = 0;
            n(i) = 0;
            continue;
        end
        
        if (~isnan(target) && (target ~= -1))
            pKoef = adjTable.pKoef(id, target);
            nKoef = adjTable.nKoef(id, target);
        else
            pKoef = 0;
            nKoef = 0;
        end
        
        L = P(i) + N(i);
        p(i) = max(p(i) - L * pKoef, 0);
        n(i) = min(n(i) + L * nKoef, N(i));
    end
end