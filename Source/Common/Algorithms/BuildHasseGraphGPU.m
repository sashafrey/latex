function edges = BuildHasseGraphGPU(algset)
    %assumes that all algs in algset are unique.
    nAlgs = algset.Count;

    nUINTS = size(algset.Data, 2);
    u = repmat(bitcmp(uint32(0)), 1, nUINTS);

    algsetGPU = gpuArray(PackLogicals32(AlgsetGet(algset, 1:algset.Count)));
    edgesGPU = gpuArray(false(nAlgs, nAlgs));
    for i=gpuArray(1:nAlgs)
        for j=gpuArray(1:nAlgs)
            edgesGPU(i, j) = min(bitor(algsetGPU(j, :), bitxor(algsetGPU(i, :), u)) == u) == 1;     
        end
    end 
    
    edgesCPU = gather(edgesGPU);
    for i=1:nAlgs
        edgesCPU(i, i) = 0;        
    end
    
    edges = EdgesCreate(nAlgs);    
    for i=1:nAlgs
        for j = find(edgesCPU(i, :))
            edges = EdgesAdd(edges, i, j);
        end
    end

    %transitive reduction
    edgesToRemove = EdgesCreate(nAlgs);
    for i=1:nAlgs
        %fprintf('Stage2 : %d of %d\r', i, nAlgs)
        iChildren = GetChildren(edges, i);
        if (isempty(iChildren))
            continue;
        end
        for iChild = iChildren
            iGrandChildren = GetChildren(edges, iChild);
            for iGrandChild = iGrandChildren
                if (EdgesContains(edges, i, iGrandChild))
                    edgesToRemove = EdgesAdd(edgesToRemove, i, iGrandChild);
                end
            end
        end            
    end

    for i=1:nAlgs
        edges.Children{i}(ismember(edges.Children{i}, edgesToRemove.Children{i})) = [];
        edges.Parents{i}(ismember(edges.Parents{i}, edgesToRemove.Parents{i})) = [];
    end
end
