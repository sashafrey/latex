function bound = getCombBound_ManyAlgs(algs, sourcesVects, L, ell, eps, boundType)
    bound = 0;
    for i = 1:algs.Count
        currAlgContrib = ...
            getCombBound_OneAlg(algs.Data(i), sourcesVects, L, ell, eps, boundType);
        if ~isnan(currAlgContrib)
            bound = bound + currAlgContrib;
        end
    end
end
