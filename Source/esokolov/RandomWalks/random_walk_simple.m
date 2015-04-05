function [algs, corrections] = random_walk_simple(X, Y, ...
    initialVertices, iterCnt, maxLevel, getNeighboursFuncHandle, display, ...
    isLazy)

    if length(initialVertices) > 1
        idx = randi(length(initialVertices));
        vertex = initialVertices(idx);
    else
        vertex = initialVertices;
    end

    L = size(X, 1);
    d = size(X, 2) - 1;
    
    algs = StructVectorCreate(vertex);
    corrections = zeros(iterCnt, 1);
    
    % [vertex, neighbours] = getNeighboursLC_rw(vertex, X, Y);
    [vertex, neighbours] = getNeighboursFuncHandle(vertex, X, Y);
    algs = StructVectorAdd(algs, vertex);

    for currIter = 2:iterCnt
        if ~isLazy || (rand(1) < 0.5)
            newAlgIdx = randi(neighbours.Count);
            vertex = neighbours.Data(newAlgIdx);

            neighbours.Count = 0;
            vertex_old = vertex;
            while neighbours.Count == 0
                vertex = vertex_old;

                [vertex, neighbours] = getNeighboursFuncHandle(vertex, X, Y);

                % если это вершина из уровня maxLevel, то нужно обрезать все ребра,
                % идующие из нее вверх
                for j = neighbours.Count:-1:1
                    if (neighbours.Data(j).errCnt > maxLevel)
                        neighbours = StructVectorDel(neighbours, j);
                    end
                end
            end
        end
        
        algs = StructVectorAdd(algs, vertex);
        
        if (display)
            fprintf('%d %d\n', currIter, vertex.errCnt);
        end
    end
    
    for i = 1:iterCnt
        corrections(i) = 1 / (algs.Data(i).upperNeighsCnt + ...
            algs.Data(i).lowerNeighsCnt);
    end
end

