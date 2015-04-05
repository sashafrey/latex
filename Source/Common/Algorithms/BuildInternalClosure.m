function [algset, edges] = BuildInternalClosure(algset, edges)
    nAlgs = algset.Count;
    tasks = QueueCreate();
    for i=1:nAlgs
        children = GetChildren(edges, i);
        for child1 = children
            for child2 = children
                if (child1 < child2) 
                    tasks = QueuePush(tasks, [i child1 child2]);
                end
            end
        end
    end
    
    while ~QueueIsEmpty(tasks)
        %fprintf('TasksQueue size : %d \r', QueueCount(tasks))
        [tasks, value] = QueuePop(tasks);
        a = value(1); b = value(2); c = value(3);
        if (~EdgesContains(edges, a, b)) continue; end;
        if (~EdgesContains(edges, a, c)) continue; end;
        bAlg = AlgsetGet(algset, b);
        cAlg = AlgsetGet(algset, c);
        bcAlg = bAlg & cAlg;
        if (AlgsetContains(algset, bcAlg)) continue; end;
        
        % Update algset
        [algset, bcId] = AlgsetAdd(algset, bcAlg);
        
        % Update edges
        children = CreateChildrenEdges(bcId, a, b, c, algset, edges);
        parents = CreateParentsEdges(bcId, a, b, c, algset, edges);
        
        for child = children
            edges = EdgesAdd(edges, bcId, child);
        end
        
        for parent = parents
            edges = EdgesAdd(edges, parent, bcId);
        end
        
        for child = children
            for parent = parents
                if (EdgesContains(edges, parent, child))
                    edges = EdgesRemove(edges, parent, child);
                end
            end                
        end
        
        % Update task queue
        for child1 = children
            for child2 = children
                if (child1 < child2)
                    tasks = QueuePush(tasks, [bcId, child1, child2]);
                end                
            end
        end
        
        for parent = parents
            for child = GetChildren(edges, parent)
                if (child ~= bcId)
                    tasks = QueuePush(tasks, [parent, child, bcId]);
                end
            end
        end        
    end
end

function children = CreateChildrenEdges(bcId, a, b, c, algset, edges)
    children = [b c];
    bcAlg = AlgsetGet(algset, bcId);
    candidates = GetAllChildren(edges, a);
    candidates(ismember(candidates, GetAllChildren(edges, b))) = [];
    candidates(ismember(candidates, GetAllChildren(edges, c))) = [];
    while (~isempty(candidates))
        candidate = candidates(1);
        candidates(1) = [];
        candidateAlg = AlgsetGet(algset, candidate);
        if (all(candidateAlg >= bcAlg) && (any(candidateAlg ~= bcAlg)))
            candidates(ismember(candidates, GetAllChildren(edges, candidate))) = [];
            replaceId = [];
            ignore = false;
            for child = children
                childAlg = AlgsetGet(algset, child);
                if (all(candidateAlg >= childAlg) && (any(candidateAlg ~= childAlg)))
                    ignore = true;
                    break;
                end
                if (all(candidateAlg <= childAlg) && (any(candidateAlg ~= childAlg)))
                    replaceId = [replaceId, child]; 
                end
            end
            
            if (ignore) 
                continue; 
            end;
            
            children(ismember(children, replaceId)) = [];             
            children = [children, candidate];
        end        
    end
end

function parents = CreateParentsEdges(bcId, a, b, c, algset, edges)
    parents = a;
    bcAlg = AlgsetGet(algset, bcId);
    candidates = intersect(GetAllParents(edges, b), GetAllParents(edges, c));
    candidates(ismember(candidates, GetAllParents(edges, a))) = [];
    while (~isempty(candidates))
        candidate = candidates(1);
        candidates(1) = [];
        candidateAlg = AlgsetGet(algset, candidate);
        if (all(candidateAlg <= bcAlg) && (any(candidateAlg ~= bcAlg)))
            candidates(ismember(candidates, GetAllParents(edges, candidate))) = [];
            replaceId = [];
            ignore = false;
            for parent = parents
                parentAlg = AlgsetGet(algset, parent);
                if (all(candidateAlg <= parentAlg) && (any(candidateAlg ~= parentAlg)))
                    ignore = true;
                    break;
                end
                if (all(candidateAlg >= parentAlg) && (any(candidateAlg ~= parentAlg)))
                    replaceId = [replaceId, parent];                    
                end
            end
            
            if (ignore) 
                continue; 
            end;
            
            parents(ismember(parents, replaceId)) = []; 
            parents = [parents, candidate];
        end        
    end
end