function graph = BuildHasseGraphSlow(rawalgset)
    nAlgs = size(rawalgset, 1);
    
    rawalgset1 = RawAlgSetRemoveDuplicates(rawalgset);
    if (size(rawalgset1, 1) ~= nAlgs) 
        throw(MException('InvalidArgument', 'Remove duplicated algorithms from algset before building HasseGraph'));
    end    
        
    graph.V = cell(nAlgs, 1);
    for i=1:nAlgs
        % fprintf('Stage1 : %d of %d\r', i, nAlgs)
        algI = rawalgset(i, :);
        outEdges = zeros(0);
        for j=1:nAlgs
            if (i == j) 
                continue; 
            end;
            
            algJ = rawalgset(j, :);
            if all(algI <= algJ) && any(algI ~= algJ)
                outEdges = [outEdges, j];
            end
        end
        
        graph.V{i} = outEdges;
    end 
    
    %transitive reduction
    graph.ToRemove = cell(nAlgs, 1);
    for i=1:nAlgs
        % fprintf('Stage2 : %d of %d\r', i, nAlgs)
        iOutEdges = graph.V{i};
        graph.ToRemove{i} = false(1, length(iOutEdges));
        for j = iOutEdges
            jOutEdges = graph.V{j};
            for k = jOutEdges
                graph.ToRemove{i}(iOutEdges == k) = true;
            end
        end            
    end
    
    for i=1:nAlgs
        graph.V{i}(graph.ToRemove{i}) = [];
    end
end

