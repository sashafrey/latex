function [quotas, counts] = CalibrateTermsQuotas(terms, nMaxTerms)
    % Analizes, how many terms to select for each feature so that in total
    % we are limited by nMaxTerms
    
    nFeatures = max(terms.feature);
    counts = zeros(nFeatures, 1);
    for i=1:nFeatures
        counts(i) = sum(terms.feature == i);
    end
    origCounts = counts;
    
    if (sum(counts) <= nMaxTerms)
        quotas = counts;
        return;
    end

    % Sort according to frequencies
    [counts, ids] = sort(counts);    
    quotas = zeros(nFeatures, 1);
    cumulativeSum = 0;
    for i=1:nFeatures
        nextCumulativeSum = cumulativeSum + counts(i) * (nFeatures - i + 1);
        if (nextCumulativeSum <= nMaxTerms)
            cumulativeSum = nextCumulativeSum;
            quotas(i:end, :) = quotas(i:end, :) + counts(i);
            counts(i:end, :) = counts(i:end, :) - counts(i);            
            continue;
        end
        
        toSelect = floor((nMaxTerms - cumulativeSum) / (nFeatures - i + 1));
        toSelect = min(toSelect, counts(i));
        counts(i) = counts(i) - toSelect;
        quotas(i) = quotas(i) + toSelect;
        cumulativeSum = cumulativeSum + toSelect;        
    end
    
    %Restore original order
    quotas(ids) = quotas;
    counts = origCounts;
end