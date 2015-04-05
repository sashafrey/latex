function [algs, corrections] = random_walk_fs(X, Y, ...
    initialVertices, iterCnt, maxLevel, getNeighboursFuncHandle, display, ...
    isLazy)

    L = size(X, 1);
    d = size(X, 2) - 1;
    nVertices = length(initialVertices);
    
    algs = StructVectorCreate(initialVertices(1));
        
    vertices = cell(nVertices, 1);
    neighbours = cell(nVertices, 1);
    neighCnt = zeros(nVertices, 1);
    
    for i = 1:nVertices
        [vertices{i}, neighbours{i}] = ...
            getNeighboursFuncHandle(initialVertices(i), X, Y);
        neighCnt(i) = neighbours{i}.Count;
        algs = StructVectorAdd(algs, vertices{i});
    end

    for currIter = 1:iterCnt
        probs = neighCnt ./ sum(neighCnt);
        r = rand(1);
        t = 0;
        for i = 1:nVertices
            t = t + probs(i);
            if (r <= t)
                idx = i;
                break;
            end
        end
        
        if (display)
            fprintf('%d %d %d\n', currIter, vertices{idx}.errCnt, idx);
        end
        
        if ~isLazy || (rand(1) < 0.5)
            newAlgIdx = randi(neighCnt(idx));

            neighbours{idx}.Count = 0;
            vertex_raw = neighbours{idx}.Data(newAlgIdx);
            while neighbours{idx}.Count == 0
                vertices{idx} = vertex_raw;

                % иногда каким-то чудом новая вершина оказывается где-то
                % высоко в графе - надо с этим бороться
                if vertices{idx}.errCnt > maxLevel
                    break;
                end

                [vertices{idx}, neighbours{idx}] = ...
                    getNeighboursFuncHandle(vertices{idx}, X, Y);

                % если это вершина из уровня maxLevel, то нужно обрезать все ребра,
                % идующие из нее вверх
                for j = neighbours{idx}.Count:-1:1
                    if (neighbours{idx}.Data(j).errCnt > maxLevel)
                        neighbours{idx} = StructVectorDel(neighbours{idx}, j);
                    end
                end
            end

            neighCnt(idx) = neighbours{idx}.Count;
        end
        
        algs = StructVectorAdd(algs, vertices{idx});
    end
    
    corrections = zeros(algs.Count, 1);
    for i = 1:algs.Count
        corrections(i) = 1 / (algs.Data(i).upperNeighsCnt + ...
            algs.Data(i).lowerNeighsCnt);
    end
end

