function termIds = CalibrateSelectTermsAccordingToQuotas(terms, quotas)
    nFeatures = max(terms.feature);
    Check(nFeatures == length(quotas));
    termIds = zeros(sum(quotas), 1);
    termIdsIterator = 0;
    for i = 1:nFeatures
        curTermsIds = find(terms.feature == i);
        nCurTerms = length(curTermsIds);
        if (quotas(i) < nCurTerms)
            curTermsIds = curTermsIds(randsample(nCurTerms, min(quotas(i), nCurTerms)));
        end     
        
        termIds((termIdsIterator + 1) : (termIdsIterator + length(curTermsIds))) = curTermsIds;
        termIdsIterator = termIdsIterator + length(curTermsIds);
    end
    
    termIds((termIdsIterator + 1) : end) = [];
end