function bound = getSrwBoundEstimate(algs_rw, corrections, sources, L, l, ...
        eps, Vt, boundType)
    
    sourcesVects = zeros(sources.Count, L);
    for i = 1:sources.Count
        sourceIdx = sources.Data(i);
        sourcesVects(i, :) = algs_rw.Data(sourceIdx).errVect;
    end
    
    est_fw = 0;
    est_w = 0;
    
    for i = 1:algs_rw.Count
        est_fw = est_fw + ...
            getCombBound_OneAlg(algs_rw.Data(i), sourcesVects, L, l, eps, boundType) * ...
            corrections(i);
        est_w = est_w + corrections(i);
    end
    
    bound = Vt * est_fw / est_w;
end
