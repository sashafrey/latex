function bound = getLayeredBoundEstimate(algs_rw, corrections, sources, L, l, ...
        eps, boundType)
    
    sourcesVects = getSourcesVects(algs_rw, sources);
    
    meanLayerBounds_fw = zeros(L + 1, 1);
    meanLayerBounds_w = zeros(L + 1, 1);
    layerSizes = zeros(L + 1, 1);
    
    for i = 1:algs_rw.Count
        currLayerIdx = algs_rw.Data(i).errCnt + 1;
        
        meanLayerBounds_fw(currLayerIdx) = meanLayerBounds_fw(currLayerIdx) + ...
            getCombBound_OneAlg(algs_rw.Data(i), sourcesVects, L, l, eps, boundType) * ...
            corrections(i);
        meanLayerBounds_w(currLayerIdx) = ...
            meanLayerBounds_w(currLayerIdx) + corrections(i);
        
        currAlgDegree = algs_rw.Data(i).lowerNeighsCnt + ...
            algs_rw.Data(i).upperNeighsCnt;
        if currLayerIdx < L
            layerSizes(currLayerIdx + 1) = layerSizes(currLayerIdx + 1) + ...
                algs_rw.Data(i).upperNeighsCnt / currAlgDegree;
        end
        if currLayerIdx > 1
            layerSizes(currLayerIdx - 1) = layerSizes(currLayerIdx - 1) + ...
                algs_rw.Data(i).lowerNeighsCnt / currAlgDegree;
        end
    end

    meanLayerBounds = meanLayerBounds_fw ./ meanLayerBounds_w;
    
    layerSizes = layerSizes / algs_rw.Count;
    
    meanLayerBounds(isnan(meanLayerBounds)) = 0;
    meanLayerBounds(isinf(meanLayerBounds)) = 0;
    layerSizes(isnan(layerSizes)) = 0;
    layerSizes(isinf(layerSizes)) = 0;
    
    bound = sum(meanLayerBounds .* layerSizes);
end
