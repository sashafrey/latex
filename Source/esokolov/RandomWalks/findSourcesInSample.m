function sources = findSourcesInSample(algs)
    sources = VectorCreate();
    
    for i = 1:algs.Count
        isSource = true;
        
        errVect_i = algs.Data(i).errVect;
        for j = 1:algs.Count
            if i == j
                continue;
            end
            
            errVect_j = algs.Data(j).errVect;
            
            if sum(errVect_i ~= errVect_j) == 0
                continue;
            end
            
            if sum(errVect_j <= errVect_i) == length(errVect_i)
                % нашли алгоритм лучше a_i
                isSource = false;
                break;
            end
        end
        
        if isSource
            sources = VectorAdd(sources, i);
        end
    end
end
