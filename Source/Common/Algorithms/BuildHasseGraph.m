function edges = BuildHasseGraph(algset)
    nAlgs = algset.Count;
    edges = EdgesCreate(nAlgs);

    nUINTS = size(algset.Data, 2);
    u = repmat(bitcmp(uint64(0)), 1, nUINTS);
    for i=1:nAlgs        
        %fprintf('Stage1 : %d of %d\r', i, nAlgs)
        algI = algset.Data(i, :);
        for j=1:nAlgs
            if (i == j) 
                continue; 
            end;
            
            algJ = algset.Data(j, :);
            iIsLessThanJ = all(bitor(algJ, bitxor(algI, u)) == u);
            if (~iIsLessThanJ) 
                continue;
            end
            
            if any(algI ~= algJ)
                edges = EdgesAdd(edges, i, j);
            end
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
