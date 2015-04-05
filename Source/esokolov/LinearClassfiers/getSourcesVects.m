function sourcesVects = getSourcesVects(algs, sources)
    if algs.Count >= 1
        L = length(algs.Data(1).errVect);
        sourcesVects = zeros(sources.Count, L);
        for i = 1:sources.Count
            sourceIdx = sources.Data(i);
            sourcesVects(i, :) = algs.Data(sourceIdx).errVect;
        end
        
        sourcesVects = unique(sourcesVects, 'rows');
    else
        sourcesVects = [];
    end
end
